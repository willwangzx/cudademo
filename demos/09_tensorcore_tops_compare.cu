#include "cuda_helpers.cuh"
#include "cpuinfo.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

// This benchmark focuses on throughput differences between general FP32 GEMM and
// Tensor Core friendly low-precision GEMM. The reported dense TOPS are measured
// GEMM throughput, while the "2:4 sparsity" line is only an effective-counting view
// to help relate benchmark results to NVIDIA's AI TOPS marketing numbers.
constexpr int kMatrixSize = 4096;
constexpr int kBenchmarkIterations = 50;
constexpr size_t kLtWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

template <typename T>
struct StorageType {
  static constexpr size_t packing = 1;
  using type = T;
};

template <>
struct StorageType<__nv_fp4_e2m1> {
  // NVFP4 values are stored packed as fp4x2, so two logical scalars share one storage slot.
  static constexpr size_t packing = 2;
  using type = __nv_fp4x2_e2m1;
};

const char* cublas_status_to_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

inline void cublas_check(cublasStatus_t status, const char* call, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "[cuBLAS ERROR] " << file << ":" << line << '\n'
              << "  call : " << call << '\n'
              << "  error: " << cublas_status_to_string(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUBLAS_CHECK(call) cublas_check((call), #call, __FILE__, __LINE__)

double compute_tops(double total_ops, double elapsed_ms) {
  const double elapsed_seconds = elapsed_ms / 1.0e3;
  return total_ops / elapsed_seconds / 1.0e12;
}

double compute_sparse_effective_tops(double dense_tops) {
  // NVIDIA AI TOPS numbers are often quoted with 2:4 sparsity enabled, which doubles
  // the effective operation count for the same dense matmul wall time.
  return dense_tops * 2.0;
}

double compute_matmul_ops(int m, int k, int n, int iterations) {
  return 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n) *
         static_cast<double>(iterations);
}

double estimate_gpu_peak_fp32_tflops(int device = 0) {
  const unsigned int cuda_cores = try_get_cuda_core_count(device);
  if (cuda_cores == 0) {
    return 0.0;
  }

  int core_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, device));

  const double core_clock_hz = static_cast<double>(core_clock_khz) * 1.0e3;
  return static_cast<double>(cuda_cores) * core_clock_hz * 2.0 / 1.0e12;
}

float compute_max_error(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_error = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    max_error = std::max(max_error, std::abs(lhs[i] - rhs[i]));
  }
  return max_error;
}

size_t round_up(size_t value, size_t granularity) {
  return granularity * ((value + (granularity - 1)) / granularity);
}

template <typename T>
T ceil_div(T value, size_t divisor) {
  return (value + static_cast<T>(divisor - 1)) / static_cast<T>(divisor);
}

template <typename T>
constexpr size_t packed_element_count(size_t scalar_count) {
  return ceil_div(scalar_count, StorageType<T>::packing);
}

size_t get_scale_tensor_size(int rows, int cols, cublasLtMatmulMatrixScale_t scale_mode) {
  // cuBLASLt low-precision paths expect auxiliary scale tensors whose shapes depend on
  // the scaling mode. This helper mirrors the layout rules from NVIDIA's samples/docs.
  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) {
    return 1;
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ||
      scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
    const size_t scale_width =
        scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 32 : 16;
    constexpr size_t kScaleBlockCols = 32;
    constexpr size_t kScaleBlockRows = 4;
    constexpr size_t kScaleBlockInner = 4;

    const size_t block_rows = kScaleBlockInner * scale_width;
    const size_t block_cols = kScaleBlockCols * kScaleBlockRows;
    const size_t scale_rows = round_up(static_cast<size_t>(rows), block_rows) / scale_width;
    const size_t scale_cols = round_up(static_cast<size_t>(cols), block_cols);
    return scale_rows * scale_cols;
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F) {
    return static_cast<size_t>(cols);
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F) {
    return ceil_div(rows, static_cast<size_t>(128)) * static_cast<size_t>(cols);
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F) {
    return round_up(ceil_div(rows, static_cast<size_t>(128)), static_cast<size_t>(4)) *
           ceil_div(cols, static_cast<size_t>(128));
  }

  return 0;
}

std::vector<__nv_fp8_e4m3> quantize_to_fp8(const std::vector<float>& input) {
  // Simple host-side quantization is enough for a throughput demo because we mainly care
  // about exercising the Tensor Core path, not building a production quantization pipeline.
  std::vector<__nv_fp8_e4m3> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = __nv_fp8_e4m3(input[i]);
  }
  return output;
}

std::vector<__nv_fp4x2_e2m1> quantize_to_nvfp4x2(const std::vector<float>& input) {
  std::vector<__nv_fp4x2_e2m1> output(packed_element_count<__nv_fp4_e2m1>(input.size()));
  for (size_t storage_idx = 0; storage_idx < output.size(); ++storage_idx) {
    const size_t element_idx = storage_idx * 2;
    // Pad the last odd element with zero because fp4 storage is packed in pairs.
    const float left = input[element_idx];
    const float right = element_idx + 1 < input.size() ? input[element_idx + 1] : 0.0f;
    const float2 packed = {left, right};
    output[storage_idx] = __nv_fp4x2_e2m1(packed);
  }
  return output;
}

struct LtPathResult {
  bool supported = false;
  float elapsed_ms = 0.0f;
  double dense_tops = 0.0;
  std::string note;
};

// FP8 and NVFP4 are configured through cuBLASLt rather than plain cuBLAS because they need
// richer descriptors: explicit scale tensors, output amax, and heuristic algorithm selection.
class Fp8LtRunner {
 public:
  bool initialize(cublasLtHandle_t handle,
                  const __nv_fp8_e4m3* d_left,
                  const __nv_fp8_e4m3* d_right,
                  const __nv_bfloat16* d_c,
                  __nv_fp8_e4m3* d_d,
                  const float* d_left_scale,
                  const float* d_right_scale,
                  const float* d_c_scale,
                  const float* d_d_scale,
                  float* d_amax_d,
                  int m,
                  int n,
                  int k,
                  void* workspace,
                  size_t workspace_size,
                  std::string* error) {
    cleanup();
    handle_ = handle;
    d_left_ = d_left;
    d_right_ = d_right;
    d_c_ = d_c;
    d_d_ = d_d;
    workspace_ = workspace;
    workspace_size_ = workspace_size;

    auto fail = [&](const char* what, cublasStatus_t status) {
      if (error != nullptr) {
        *error = std::string(what) + ": " + cublas_status_to_string(status);
      }
      cleanup();
      return false;
    };

    const cublasOperation_t transa = CUBLAS_OP_N;
    const cublasOperation_t transb = CUBLAS_OP_N;

    cublasStatus_t status =
        cublasLtMatmulDescCreate(&operation_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatmulDescCreate", status);
    }

    status = cublasLtMatmulDescSetAttribute(operation_desc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transa, sizeof(transa));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_TRANSA", status);
    }

    status = cublasLtMatmulDescSetAttribute(operation_desc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transb, sizeof(transb));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_TRANSB", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_left_scale,
        sizeof(d_left_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_A_SCALE_POINTER", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_right_scale,
        sizeof(d_right_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_B_SCALE_POINTER", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &d_c_scale,
        sizeof(d_c_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_C_SCALE_POINTER", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_d_scale,
        sizeof(d_d_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_D_SCALE_POINTER", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_amax_d,
        sizeof(d_amax_d));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_AMAX_D_POINTER", status);
    }

    // Like the cuBLAS row-major trick, these descriptors are intentionally laid out as
    // the transposed view of row-major matrices so the math still corresponds to C = A * B.
    status = cublasLtMatrixLayoutCreate(&left_desc_, CUDA_R_8F_E4M3, n, k, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(left)", status);
    }

    status = cublasLtMatrixLayoutCreate(&right_desc_, CUDA_R_8F_E4M3, k, m, k);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(right)", status);
    }

    status = cublasLtMatrixLayoutCreate(&c_desc_, CUDA_R_16BF, n, m, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(C)", status);
    }

    status = cublasLtMatrixLayoutCreate(&d_desc_, CUDA_R_8F_E4M3, n, m, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(D)", status);
    }

    status = cublasLtMatmulPreferenceCreate(&preference_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatmulPreferenceCreate", status);
    }

    status = cublasLtMatmulPreferenceSetAttribute(
        preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_,
        sizeof(workspace_size_));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES", status);
    }

    // Ask cuBLASLt for one valid algorithm. If none is returned, this precision path is
    // not available for the current GPU / driver / toolkit combination.
    int returned_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(handle_, operation_desc_, left_desc_, right_desc_,
                                            c_desc_, d_desc_, preference_, 1, &heuristic_,
                                            &returned_results);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatmulAlgoGetHeuristic(FP8)", status);
    }
    if (returned_results == 0) {
      return fail("cublasLtMatmulAlgoGetHeuristic(FP8)", CUBLAS_STATUS_NOT_SUPPORTED);
    }

    initialized_ = true;
    return true;
  }

  cublasStatus_t run(cudaStream_t stream = 0) const {
    if (!initialized_) {
      return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    return cublasLtMatmul(handle_, operation_desc_, &alpha_, d_left_, left_desc_, d_right_,
                          right_desc_, &beta_, d_c_, c_desc_, d_d_, d_desc_,
                          &heuristic_.algo, workspace_, workspace_size_, stream);
  }

  ~Fp8LtRunner() { cleanup(); }

 private:
  void cleanup() {
    if (preference_ != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference_);
      preference_ = nullptr;
    }
    if (d_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(d_desc_);
      d_desc_ = nullptr;
    }
    if (c_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc_);
      c_desc_ = nullptr;
    }
    if (right_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(right_desc_);
      right_desc_ = nullptr;
    }
    if (left_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(left_desc_);
      left_desc_ = nullptr;
    }
    if (operation_desc_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_desc_);
      operation_desc_ = nullptr;
    }
    initialized_ = false;
  }

  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t operation_desc_ = nullptr;
  cublasLtMatrixLayout_t left_desc_ = nullptr;
  cublasLtMatrixLayout_t right_desc_ = nullptr;
  cublasLtMatrixLayout_t c_desc_ = nullptr;
  cublasLtMatrixLayout_t d_desc_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic_{};
  const __nv_fp8_e4m3* d_left_ = nullptr;
  const __nv_fp8_e4m3* d_right_ = nullptr;
  const __nv_bfloat16* d_c_ = nullptr;
  __nv_fp8_e4m3* d_d_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
  float alpha_ = 1.0f;
  float beta_ = 0.0f;
  bool initialized_ = false;
};

class Nvfp4LtRunner {
 public:
  bool initialize(cublasLtHandle_t handle,
                  const __nv_fp4x2_e2m1* d_left,
                  const __nv_fp4x2_e2m1* d_right,
                  const __nv_bfloat16* d_c,
                  __nv_fp4x2_e2m1* d_d,
                  const __nv_fp8_e4m3* d_left_scale,
                  const __nv_fp8_e4m3* d_right_scale,
                  const float* d_d_scale,
                  __nv_fp8_e4m3* d_d_out_scale,
                  int m,
                  int n,
                  int k,
                  void* workspace,
                  size_t workspace_size,
                  std::string* error) {
    cleanup();
    handle_ = handle;
    d_left_ = d_left;
    d_right_ = d_right;
    d_c_ = d_c;
    d_d_ = d_d;
    workspace_ = workspace;
    workspace_size_ = workspace_size;

    auto fail = [&](const char* what, cublasStatus_t status) {
      if (error != nullptr) {
        *error = std::string(what) + ": " + cublas_status_to_string(status);
      }
      cleanup();
      return false;
    };

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    const cublasLtMatmulMatrixScale_t block_scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    const cublasLtMatmulMatrixScale_t scalar_scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

    cublasStatus_t status =
        cublasLtMatmulDescCreate(&operation_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatmulDescCreate", status);
    }

    status = cublasLtMatmulDescSetAttribute(operation_desc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transa, sizeof(transa));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_TRANSA", status);
    }

    status = cublasLtMatmulDescSetAttribute(operation_desc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transb, sizeof(transb));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_TRANSB", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &block_scale_mode,
        sizeof(block_scale_mode));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_A_SCALE_MODE", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &block_scale_mode,
        sizeof(block_scale_mode));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_B_SCALE_MODE", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scalar_scale_mode,
        sizeof(scalar_scale_mode));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_D_SCALE_MODE", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &block_scale_mode,
        sizeof(block_scale_mode));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_left_scale,
        sizeof(d_left_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_A_SCALE_POINTER", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_right_scale,
        sizeof(d_right_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_B_SCALE_POINTER", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_d_scale,
        sizeof(d_d_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_D_SCALE_POINTER", status);
    }

    status = cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_d_out_scale,
        sizeof(d_d_out_scale));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER", status);
    }

    // NVFP4 follows the official cuBLASLt sample layout more closely than the FP8 path:
    // the A operand is exposed with an explicit transpose and block-scale metadata.
    const uint64_t rows_a = transa == CUBLAS_OP_N ? static_cast<uint64_t>(m)
                                                  : static_cast<uint64_t>(k);
    const uint64_t cols_a = transa == CUBLAS_OP_N ? static_cast<uint64_t>(k)
                                                  : static_cast<uint64_t>(m);
    const int64_t lda = transa == CUBLAS_OP_N ? m : k;
    const uint64_t rows_b = transb == CUBLAS_OP_N ? static_cast<uint64_t>(k)
                                                  : static_cast<uint64_t>(n);
    const uint64_t cols_b = transb == CUBLAS_OP_N ? static_cast<uint64_t>(n)
                                                  : static_cast<uint64_t>(k);
    const int64_t ldb = transb == CUBLAS_OP_N ? k : n;

    status = cublasLtMatrixLayoutCreate(&left_desc_, CUDA_R_4F_E2M1, rows_a, cols_a, lda);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(left)", status);
    }

    status = cublasLtMatrixLayoutCreate(&right_desc_, CUDA_R_4F_E2M1, rows_b, cols_b, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(right)", status);
    }

    status = cublasLtMatrixLayoutCreate(&c_desc_, CUDA_R_16BF, m, n, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(C)", status);
    }

    status = cublasLtMatrixLayoutCreate(&d_desc_, CUDA_R_4F_E2M1, m, n, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatrixLayoutCreate(D)", status);
    }

    status = cublasLtMatmulPreferenceCreate(&preference_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatmulPreferenceCreate", status);
    }

    status = cublasLtMatmulPreferenceSetAttribute(
        preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_,
        sizeof(workspace_size_));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES", status);
    }

    int returned_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(handle_, operation_desc_, left_desc_, right_desc_,
                                            c_desc_, d_desc_, preference_, 1, &heuristic_,
                                            &returned_results);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail("cublasLtMatmulAlgoGetHeuristic(NVFP4)", status);
    }
    if (returned_results == 0) {
      return fail("cublasLtMatmulAlgoGetHeuristic(NVFP4)", CUBLAS_STATUS_NOT_SUPPORTED);
    }

    initialized_ = true;
    return true;
  }

  cublasStatus_t run(cudaStream_t stream = 0) const {
    if (!initialized_) {
      return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    return cublasLtMatmul(handle_, operation_desc_, &alpha_, d_left_, left_desc_, d_right_,
                          right_desc_, &beta_, d_c_, c_desc_, d_d_, d_desc_,
                          &heuristic_.algo, workspace_, workspace_size_, stream);
  }

  ~Nvfp4LtRunner() { cleanup(); }

 private:
  void cleanup() {
    if (preference_ != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference_);
      preference_ = nullptr;
    }
    if (d_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(d_desc_);
      d_desc_ = nullptr;
    }
    if (c_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc_);
      c_desc_ = nullptr;
    }
    if (right_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(right_desc_);
      right_desc_ = nullptr;
    }
    if (left_desc_ != nullptr) {
      cublasLtMatrixLayoutDestroy(left_desc_);
      left_desc_ = nullptr;
    }
    if (operation_desc_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_desc_);
      operation_desc_ = nullptr;
    }
    initialized_ = false;
  }

  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t operation_desc_ = nullptr;
  cublasLtMatrixLayout_t left_desc_ = nullptr;
  cublasLtMatrixLayout_t right_desc_ = nullptr;
  cublasLtMatrixLayout_t c_desc_ = nullptr;
  cublasLtMatrixLayout_t d_desc_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic_{};
  const __nv_fp4x2_e2m1* d_left_ = nullptr;
  const __nv_fp4x2_e2m1* d_right_ = nullptr;
  const __nv_bfloat16* d_c_ = nullptr;
  __nv_fp4x2_e2m1* d_d_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
  float alpha_ = 1.0f;
  float beta_ = 0.0f;
  bool initialized_ = false;
};

void run_fp32_pedantic(cublasHandle_t handle,
                       const float* d_a,
                       const float* d_b,
                       float* d_c,
                       int m,
                       int k,
                       int n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Pedantic FP32 disables fast reduced-precision modes and serves as the general-purpose
  // floating-point baseline for this file.
  CUBLAS_CHECK(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            d_b,
                            CUDA_R_32F,
                            n,
                            d_a,
                            CUDA_R_32F,
                            k,
                            &beta,
                            d_c,
                            CUDA_R_32F,
                            n,
                            CUBLAS_COMPUTE_32F_PEDANTIC,
                            CUBLAS_GEMM_DEFAULT));
}

void run_tf32(cublasHandle_t handle,
              const float* d_a,
              const float* d_b,
              float* d_c,
              int m,
              int k,
              int n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // TF32 keeps FP32 inputs/outputs but routes the multiply path through Tensor Cores.
  CUBLAS_CHECK(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            d_b,
                            CUDA_R_32F,
                            n,
                            d_a,
                            CUDA_R_32F,
                            k,
                            &beta,
                            d_c,
                            CUDA_R_32F,
                            n,
                            CUBLAS_COMPUTE_32F_FAST_TF32,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void run_fp16(cublasHandle_t handle,
              const __half* d_a,
              const __half* d_b,
              float* d_c,
              int m,
              int k,
              int n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // FP16 inputs with FP32 accumulation are a common dense-Tensor-Core training/inference mix.
  CUBLAS_CHECK(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            d_b,
                            CUDA_R_16F,
                            n,
                            d_a,
                            CUDA_R_16F,
                            k,
                            &beta,
                            d_c,
                            CUDA_R_32F,
                            n,
                            CUBLAS_COMPUTE_32F_FAST_16F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void run_int8(cublasHandle_t handle,
              const int8_t* d_a,
              const int8_t* d_b,
              int32_t* d_c,
              int m,
              int k,
              int n) {
  const int32_t alpha = 1;
  const int32_t beta = 0;

  // INT8 uses integer accumulation, so the output buffer is int32 rather than float.
  CUBLAS_CHECK(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            d_b,
                            CUDA_R_8I,
                            n,
                            d_a,
                            CUDA_R_8I,
                            k,
                            &beta,
                            d_c,
                            CUDA_R_32I,
                            n,
                            CUBLAS_COMPUTE_32I,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <typename LaunchFn>
float benchmark_ms(LaunchFn&& launch, int iterations) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warm up once before recording so first-use overhead does not distort the steady-state path.
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < iterations; ++iter) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms;
}

int32_t compute_int8_sample_reference(const std::vector<int8_t>& a,
                                      const std::vector<int8_t>& b,
                                      int k,
                                      int n,
                                      int row,
                                      int col) {
  int32_t value = 0;
  for (int inner = 0; inner < k; ++inner) {
    value += static_cast<int32_t>(a[row * k + inner]) *
             static_cast<int32_t>(b[inner * n + col]);
  }
  return value;
}

}  // namespace

int main() {
  constexpr int kM = kMatrixSize;
  constexpr int kK = kMatrixSize;
  constexpr int kN = kMatrixSize;

  // All paths benchmark the same GEMM shape so the throughput numbers stay directly comparable.
  const size_t float_elements = static_cast<size_t>(kM) * kK;
  const size_t float_c_elements = static_cast<size_t>(kM) * kN;
  const size_t bytes_f32_a = float_elements * sizeof(float);
  const size_t bytes_f32_b = static_cast<size_t>(kK) * kN * sizeof(float);
  const size_t bytes_f32_c = float_c_elements * sizeof(float);
  const size_t bytes_f16_a = float_elements * sizeof(__half);
  const size_t bytes_f16_b = static_cast<size_t>(kK) * kN * sizeof(__half);
  const size_t bytes_i8_a = float_elements * sizeof(int8_t);
  const size_t bytes_i8_b = static_cast<size_t>(kK) * kN * sizeof(int8_t);
  const size_t bytes_i32_c = float_c_elements * sizeof(int32_t);
  const size_t bytes_fp8_a = float_elements * sizeof(__nv_fp8_e4m3);
  const size_t bytes_fp8_b = static_cast<size_t>(kK) * kN * sizeof(__nv_fp8_e4m3);
  const size_t bytes_fp8_c = float_c_elements * sizeof(__nv_fp8_e4m3);
  const size_t bytes_bf16_c = float_c_elements * sizeof(__nv_bfloat16);
  const size_t bytes_nvfp4_a =
      packed_element_count<__nv_fp4_e2m1>(float_elements) * sizeof(__nv_fp4x2_e2m1);
  const size_t bytes_nvfp4_b = packed_element_count<__nv_fp4_e2m1>(static_cast<size_t>(kK) * kN) *
                               sizeof(__nv_fp4x2_e2m1);
  const size_t bytes_nvfp4_c =
      packed_element_count<__nv_fp4_e2m1>(float_c_elements) * sizeof(__nv_fp4x2_e2m1);
  // NVFP4 uses per-block scales, so its auxiliary metadata footprint depends on matrix shape.
  const size_t nvfp4_left_scale_count =
      get_scale_tensor_size(kN, kK, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
  const size_t nvfp4_right_scale_count =
      get_scale_tensor_size(kK, kM, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
  const size_t nvfp4_output_scale_count =
      get_scale_tensor_size(kN, kM, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);

  std::vector<float> h_a_f32(float_elements);
  std::vector<float> h_b_f32(static_cast<size_t>(kK) * kN);
  std::vector<__half> h_a_f16(float_elements);
  std::vector<__half> h_b_f16(static_cast<size_t>(kK) * kN);
  std::vector<int8_t> h_a_i8(float_elements);
  std::vector<int8_t> h_b_i8(static_cast<size_t>(kK) * kN);
  std::vector<float> h_c_fp32(float_c_elements, 0.0f);
  std::vector<float> h_c_tf32(float_c_elements, 0.0f);
  std::vector<float> h_c_fp16(float_c_elements, 0.0f);
  std::vector<int32_t> h_c_int8(float_c_elements, 0);

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kK; ++col) {
      const float value =
          -1.0f + static_cast<float>((row * 7 + col * 3) % 97) / 48.0f;
      h_a_f32[row * kK + col] = value;
      h_a_f16[row * kK + col] = __float2half_rn(value);
      h_a_i8[row * kK + col] = static_cast<int8_t>((row * 5 + col * 3) % 15 - 7);
    }
  }

  for (int row = 0; row < kK; ++row) {
    for (int col = 0; col < kN; ++col) {
      const float value =
          -0.75f + static_cast<float>((row * 11 + col * 2) % 89) / 56.0f;
      h_b_f32[row * kN + col] = value;
      h_b_f16[row * kN + col] = __float2half_rn(value);
      h_b_i8[row * kN + col] = static_cast<int8_t>((row * 2 + col * 5) % 15 - 7);
    }
  }

  const std::vector<__nv_fp8_e4m3> h_a_fp8 = quantize_to_fp8(h_a_f32);
  const std::vector<__nv_fp8_e4m3> h_b_fp8 = quantize_to_fp8(h_b_f32);
  const std::vector<__nv_fp4x2_e2m1> h_a_nvfp4 = quantize_to_nvfp4x2(h_a_f32);
  const std::vector<__nv_fp4x2_e2m1> h_b_nvfp4 = quantize_to_nvfp4x2(h_b_f32);
  const std::vector<__nv_fp8_e4m3> h_nvfp4_left_scale(nvfp4_left_scale_count,
                                                       __nv_fp8_e4m3(1.0f));
  const std::vector<__nv_fp8_e4m3> h_nvfp4_right_scale(nvfp4_right_scale_count,
                                                        __nv_fp8_e4m3(1.0f));
  const std::vector<__nv_fp8_e4m3> h_nvfp4_output_scale(nvfp4_output_scale_count,
                                                         __nv_fp8_e4m3(1.0f));

  print_cpu_info();
  print_device_summary();

  float* d_a_f32 = nullptr;
  float* d_b_f32 = nullptr;
  float* d_c_f32 = nullptr;
  __half* d_a_f16 = nullptr;
  __half* d_b_f16 = nullptr;
  int8_t* d_a_i8 = nullptr;
  int8_t* d_b_i8 = nullptr;
  int32_t* d_c_i32 = nullptr;
  __nv_fp8_e4m3* d_a_fp8 = nullptr;
  __nv_fp8_e4m3* d_b_fp8 = nullptr;
  __nv_fp8_e4m3* d_d_fp8 = nullptr;
  __nv_fp4x2_e2m1* d_a_nvfp4 = nullptr;
  __nv_fp4x2_e2m1* d_b_nvfp4 = nullptr;
  __nv_fp4x2_e2m1* d_d_nvfp4 = nullptr;
  __nv_bfloat16* d_c_bf16 = nullptr;
  float* d_fp8_left_scale = nullptr;
  float* d_fp8_right_scale = nullptr;
  float* d_fp8_c_scale = nullptr;
  float* d_fp8_d_scale = nullptr;
  float* d_fp8_amax = nullptr;
  __nv_fp8_e4m3* d_nvfp4_left_scale = nullptr;
  __nv_fp8_e4m3* d_nvfp4_right_scale = nullptr;
  __nv_fp8_e4m3* d_nvfp4_output_scale = nullptr;
  float* d_nvfp4_d_scale = nullptr;
  void* d_lt_workspace = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a_f32, bytes_f32_a));
  CUDA_CHECK(cudaMalloc(&d_b_f32, bytes_f32_b));
  CUDA_CHECK(cudaMalloc(&d_c_f32, bytes_f32_c));
  CUDA_CHECK(cudaMalloc(&d_a_f16, bytes_f16_a));
  CUDA_CHECK(cudaMalloc(&d_b_f16, bytes_f16_b));
  CUDA_CHECK(cudaMalloc(&d_a_i8, bytes_i8_a));
  CUDA_CHECK(cudaMalloc(&d_b_i8, bytes_i8_b));
  CUDA_CHECK(cudaMalloc(&d_c_i32, bytes_i32_c));
  CUDA_CHECK(cudaMalloc(&d_a_fp8, bytes_fp8_a));
  CUDA_CHECK(cudaMalloc(&d_b_fp8, bytes_fp8_b));
  CUDA_CHECK(cudaMalloc(&d_d_fp8, bytes_fp8_c));
  CUDA_CHECK(cudaMalloc(&d_a_nvfp4, bytes_nvfp4_a));
  CUDA_CHECK(cudaMalloc(&d_b_nvfp4, bytes_nvfp4_b));
  CUDA_CHECK(cudaMalloc(&d_d_nvfp4, bytes_nvfp4_c));
  CUDA_CHECK(cudaMalloc(&d_c_bf16, bytes_bf16_c));
  CUDA_CHECK(cudaMalloc(&d_fp8_left_scale, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fp8_right_scale, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fp8_c_scale, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fp8_d_scale, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fp8_amax, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_nvfp4_left_scale,
                        nvfp4_left_scale_count * sizeof(__nv_fp8_e4m3)));
  CUDA_CHECK(cudaMalloc(&d_nvfp4_right_scale,
                        nvfp4_right_scale_count * sizeof(__nv_fp8_e4m3)));
  CUDA_CHECK(cudaMalloc(&d_nvfp4_output_scale,
                        nvfp4_output_scale_count * sizeof(__nv_fp8_e4m3)));
  CUDA_CHECK(cudaMalloc(&d_nvfp4_d_scale, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lt_workspace, kLtWorkspaceBytes));

  CUDA_CHECK(cudaMemcpy(d_a_f32, h_a_f32.data(), bytes_f32_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_f32, h_b_f32.data(), bytes_f32_b, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_a_f16, h_a_f16.data(), bytes_f16_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_f16, h_b_f16.data(), bytes_f16_b, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_a_i8, h_a_i8.data(), bytes_i8_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_i8, h_b_i8.data(), bytes_i8_b, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_a_fp8, h_a_fp8.data(), bytes_fp8_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_fp8, h_b_fp8.data(), bytes_fp8_b, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_a_nvfp4, h_a_nvfp4.data(), bytes_nvfp4_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_nvfp4, h_b_nvfp4.data(), bytes_nvfp4_b, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nvfp4_left_scale, h_nvfp4_left_scale.data(),
                        nvfp4_left_scale_count * sizeof(__nv_fp8_e4m3),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nvfp4_right_scale, h_nvfp4_right_scale.data(),
                        nvfp4_right_scale_count * sizeof(__nv_fp8_e4m3),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nvfp4_output_scale, h_nvfp4_output_scale.data(),
                        nvfp4_output_scale_count * sizeof(__nv_fp8_e4m3),
                        cudaMemcpyHostToDevice));

  const float one_scale = 1.0f;
  const float zero_value = 0.0f;
  CUDA_CHECK(
      cudaMemcpy(d_fp8_left_scale, &one_scale, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_fp8_right_scale, &one_scale, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fp8_c_scale, &one_scale, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fp8_d_scale, &one_scale, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fp8_amax, &zero_value, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_nvfp4_d_scale, &one_scale, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c_bf16, 0, bytes_bf16_c));

  cublasHandle_t handle = nullptr;
  cublasLtHandle_t lt_handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasLtCreate(&lt_handle));

  // We count GEMM as 2 * M * N * K operations so TFLOPS/TOPS are directly comparable across
  // all precisions, even though the internal hardware pipelines are very different.
  const double total_ops = compute_matmul_ops(kM, kK, kN, kBenchmarkIterations);

  const float fp32_ms = benchmark_ms(
      [&] { run_fp32_pedantic(handle, d_a_f32, d_b_f32, d_c_f32, kM, kK, kN); },
      kBenchmarkIterations);
  CUDA_CHECK(cudaMemcpy(h_c_fp32.data(), d_c_f32, bytes_f32_c, cudaMemcpyDeviceToHost));

  const float tf32_ms =
      benchmark_ms([&] { run_tf32(handle, d_a_f32, d_b_f32, d_c_f32, kM, kK, kN); },
                   kBenchmarkIterations);
  CUDA_CHECK(cudaMemcpy(h_c_tf32.data(), d_c_f32, bytes_f32_c, cudaMemcpyDeviceToHost));

  const float fp16_ms =
      benchmark_ms([&] { run_fp16(handle, d_a_f16, d_b_f16, d_c_f32, kM, kK, kN); },
                   kBenchmarkIterations);
  CUDA_CHECK(cudaMemcpy(h_c_fp16.data(), d_c_f32, bytes_f32_c, cudaMemcpyDeviceToHost));

  const float int8_ms =
      benchmark_ms([&] { run_int8(handle, d_a_i8, d_b_i8, d_c_i32, kM, kK, kN); },
                   kBenchmarkIterations);
  CUDA_CHECK(cudaMemcpy(h_c_int8.data(), d_c_i32, bytes_i32_c, cudaMemcpyDeviceToHost));

  LtPathResult fp8_result;
  LtPathResult nvfp4_result;
  float fp8_amax = 0.0f;
  Fp8LtRunner fp8_runner;
  // FP8/NVFP4 are optional paths. If initialization fails, we report "skipped" rather than
  // aborting the whole demo so the rest of the benchmark still provides value.
  if (fp8_runner.initialize(lt_handle, d_b_fp8, d_a_fp8, d_c_bf16, d_d_fp8,
                            d_fp8_left_scale, d_fp8_right_scale, d_fp8_c_scale,
                            d_fp8_d_scale, d_fp8_amax, kM, kN, kK, d_lt_workspace,
                            kLtWorkspaceBytes, &fp8_result.note)) {
    fp8_result.supported = true;
    fp8_result.elapsed_ms = benchmark_ms([&] { CUBLAS_CHECK(fp8_runner.run()); },
                                         kBenchmarkIterations);
    fp8_result.dense_tops = compute_tops(total_ops, fp8_result.elapsed_ms);
    CUDA_CHECK(cudaMemcpy(&fp8_amax, d_fp8_amax, sizeof(float), cudaMemcpyDeviceToHost));
  }

  Nvfp4LtRunner nvfp4_runner;
  if (nvfp4_runner.initialize(lt_handle, d_a_nvfp4, d_b_nvfp4, d_c_bf16, d_d_nvfp4,
                              d_nvfp4_left_scale, d_nvfp4_right_scale, d_nvfp4_d_scale,
                              d_nvfp4_output_scale, kM, kN, kK, d_lt_workspace,
                              kLtWorkspaceBytes, &nvfp4_result.note)) {
    nvfp4_result.supported = true;
    nvfp4_result.elapsed_ms = benchmark_ms([&] { CUBLAS_CHECK(nvfp4_runner.run()); },
                                           kBenchmarkIterations);
    nvfp4_result.dense_tops = compute_tops(total_ops, nvfp4_result.elapsed_ms);
  }

  const double fp32_tflops = compute_tops(total_ops, fp32_ms);
  const double tf32_tflops = compute_tops(total_ops, tf32_ms);
  const double fp16_tflops = compute_tops(total_ops, fp16_ms);
  const double int8_tops = compute_tops(total_ops, int8_ms);
  const double gpu_peak_fp32_tflops = estimate_gpu_peak_fp32_tflops();

  const float tf32_max_error = compute_max_error(h_c_fp32, h_c_tf32);
  const float fp16_max_error = compute_max_error(h_c_fp32, h_c_fp16);
  const int sample_row = 7;
  const int sample_col = 11;
  const int32_t int8_sample_reference =
      compute_int8_sample_reference(h_a_i8, h_b_i8, kK, kN, sample_row, sample_col);
  const int32_t int8_sample_result = h_c_int8[sample_row * kN + sample_col];

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Benchmark: Tensor Core throughput compare\n";
  std::cout << "Matrix shape: A(" << kM << " x " << kK << "), B(" << kK << " x " << kN
            << "), C(" << kM << " x " << kN << ")\n";
  std::cout << "Iterations per path: " << kBenchmarkIterations << '\n';
  std::cout << "Dense MAC counting: 2 * M * N * K operations\n";
  std::cout << "FP32 cuBLAS time: " << fp32_ms << " ms"
            << " (avg " << fp32_ms / kBenchmarkIterations << " ms)\n";
  std::cout << "FP32 cuBLAS throughput: " << fp32_tflops << " TFLOP/s\n";
  std::cout << "TF32 Tensor Core time: " << tf32_ms << " ms"
            << " (avg " << tf32_ms / kBenchmarkIterations << " ms)\n";
  std::cout << "TF32 Tensor Core throughput: " << tf32_tflops << " TFLOP/s"
            << " (" << tf32_tflops / fp32_tflops << "x vs FP32)\n";
  std::cout << "FP16 Tensor Core time: " << fp16_ms << " ms"
            << " (avg " << fp16_ms / kBenchmarkIterations << " ms)\n";
  std::cout << "FP16 Tensor Core throughput: " << fp16_tflops << " TFLOP/s"
            << " (" << fp16_tflops / fp32_tflops << "x vs FP32)\n";
  std::cout << "INT8 Tensor Core time: " << int8_ms << " ms"
            << " (avg " << int8_ms / kBenchmarkIterations << " ms)\n";
  std::cout << "INT8 Tensor Core throughput: " << int8_tops << " TOP/s"
            << " (" << int8_tops / fp32_tflops << "x vs FP32)\n";
  std::cout << "INT8 Tensor Core effective throughput (2:4 sparsity): "
            << compute_sparse_effective_tops(int8_tops) << " TOP/s\n";
  if (fp8_result.supported) {
    std::cout << "FP8 Tensor Core time: " << fp8_result.elapsed_ms << " ms"
              << " (avg " << fp8_result.elapsed_ms / kBenchmarkIterations << " ms)\n";
    std::cout << "FP8 Tensor Core dense throughput: " << fp8_result.dense_tops << " TOP/s"
              << " (" << fp8_result.dense_tops / fp32_tflops << "x vs FP32)\n";
    std::cout << "FP8 Tensor Core effective throughput (2:4 sparsity): "
              << compute_sparse_effective_tops(fp8_result.dense_tops) << " TOP/s\n";
    std::cout << "FP8 output amax after benchmark: " << fp8_amax << '\n';
  } else {
    std::cout << "FP8 Tensor Core path: skipped";
    if (!fp8_result.note.empty()) {
      std::cout << " (" << fp8_result.note << ")";
    }
    std::cout << '\n';
  }
  if (nvfp4_result.supported) {
    std::cout << "NVFP4 Tensor Core time: " << nvfp4_result.elapsed_ms << " ms"
              << " (avg " << nvfp4_result.elapsed_ms / kBenchmarkIterations << " ms)\n";
    std::cout << "NVFP4 Tensor Core dense throughput: " << nvfp4_result.dense_tops
              << " TOP/s"
              << " (" << nvfp4_result.dense_tops / fp32_tflops << "x vs FP32)\n";
    std::cout << "NVFP4 Tensor Core effective throughput (2:4 sparsity): "
              << compute_sparse_effective_tops(nvfp4_result.dense_tops) << " TOP/s\n";
  } else {
    std::cout << "NVFP4 Tensor Core path: skipped";
    if (!nvfp4_result.note.empty()) {
      std::cout << " (" << nvfp4_result.note << ")";
    }
    std::cout << '\n';
  }
  if (gpu_peak_fp32_tflops > 0.0) {
    std::cout << "Estimated GPU FP32 peak (CUDA-reported clock): " << gpu_peak_fp32_tflops
              << " TFLOP/s\n";
    std::cout << "FP32 utilization vs reported peak: "
              << (fp32_tflops / gpu_peak_fp32_tflops) * 100.0 << " %\n";
  }
  std::cout << "TF32 max error vs FP32: " << tf32_max_error << '\n';
  std::cout << "FP16 max error vs FP32: " << fp16_max_error << '\n';
  std::cout << "INT8 sample check: C[" << sample_row << ", " << sample_col
            << "] = " << int8_sample_result << " (expected " << int8_sample_reference
            << ")\n";
  std::cout << "Note: dense TOPS are measured GEMM throughput.\n";
  std::cout << "Note: the 2:4 sparse line is a simple dense * 2 effective-counting view, "
               "which is the closest benchmark-side bridge to NVIDIA AI TOPS marketing.\n";
  std::cout << "Note: FP8 and NVFP4 output values depend on their associated scale tensors, "
               "so this demo treats both mainly as throughput paths.\n";

  CUBLAS_CHECK(cublasLtDestroy(lt_handle));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_a_f32));
  CUDA_CHECK(cudaFree(d_b_f32));
  CUDA_CHECK(cudaFree(d_c_f32));
  CUDA_CHECK(cudaFree(d_a_f16));
  CUDA_CHECK(cudaFree(d_b_f16));
  CUDA_CHECK(cudaFree(d_a_i8));
  CUDA_CHECK(cudaFree(d_b_i8));
  CUDA_CHECK(cudaFree(d_c_i32));
  CUDA_CHECK(cudaFree(d_a_fp8));
  CUDA_CHECK(cudaFree(d_b_fp8));
  CUDA_CHECK(cudaFree(d_d_fp8));
  CUDA_CHECK(cudaFree(d_a_nvfp4));
  CUDA_CHECK(cudaFree(d_b_nvfp4));
  CUDA_CHECK(cudaFree(d_d_nvfp4));
  CUDA_CHECK(cudaFree(d_c_bf16));
  CUDA_CHECK(cudaFree(d_fp8_left_scale));
  CUDA_CHECK(cudaFree(d_fp8_right_scale));
  CUDA_CHECK(cudaFree(d_fp8_c_scale));
  CUDA_CHECK(cudaFree(d_fp8_d_scale));
  CUDA_CHECK(cudaFree(d_fp8_amax));
  CUDA_CHECK(cudaFree(d_nvfp4_left_scale));
  CUDA_CHECK(cudaFree(d_nvfp4_right_scale));
  CUDA_CHECK(cudaFree(d_nvfp4_output_scale));
  CUDA_CHECK(cudaFree(d_nvfp4_d_scale));
  CUDA_CHECK(cudaFree(d_lt_workspace));
  return 0;
}
