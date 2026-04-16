#include "cuda_helpers.cuh"
#include "cpuinfo.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include <cublas_v2.h>

namespace {

constexpr int kTile = 16;
constexpr int kMatrixSize = 2048;
constexpr int kCpuBenchmarkIterations = 2;
constexpr int kGpuBenchmarkIterations = 20;

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

__global__ void tiled_matmul(const float* a,
                             const float* b,
                             float* c,
                             int m,
                             int k,
                             int n) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  const int tiles = div_up(k, kTile);

  float value = 0.0f;

  for (int tile = 0; tile < tiles; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();

    for (int i = 0; i < kTile; ++i) {
      value += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = value;
  }
}

void matmul_cpu_rows(const std::vector<float>& a,
                     const std::vector<float>& b,
                     std::vector<float>& c,
                     int k,
                     int n,
                     int row_begin,
                     int row_end) {
  for (int row = row_begin; row < row_end; ++row) {
    float* c_row = c.data() + static_cast<size_t>(row) * n;
    const float* a_row = a.data() + static_cast<size_t>(row) * k;

    std::fill(c_row, c_row + n, 0.0f);

    for (int inner = 0; inner < k; ++inner) {
      const float a_value = a_row[inner];
      const float* b_row = b.data() + static_cast<size_t>(inner) * n;

      for (int col = 0; col < n; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

unsigned int get_cpu_worker_count(int rows) {
  unsigned int workers = getPhysicalCoreCount();
  if (workers == 0) {
    workers = std::thread::hardware_concurrency();
  }
  if (workers == 0) {
    workers = 1;
  }

  return std::min(workers, static_cast<unsigned int>(std::max(rows, 1)));
}

void matmul_cpu_parallel(const std::vector<float>& a,
                         const std::vector<float>& b,
                         std::vector<float>& c,
                         int m,
                         int k,
                         int n,
                         unsigned int worker_count) {
  if (worker_count <= 1) {
    matmul_cpu_rows(a, b, c, k, n, 0, m);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  const int base_rows = m / static_cast<int>(worker_count);
  const int remainder = m % static_cast<int>(worker_count);

  int row_begin = 0;
  for (unsigned int worker = 0; worker < worker_count; ++worker) {
    const int rows_for_worker =
        base_rows + (worker < static_cast<unsigned int>(remainder) ? 1 : 0);
    const int row_end = row_begin + rows_for_worker;

    workers.emplace_back(
        matmul_cpu_rows, std::cref(a), std::cref(b), std::ref(c), k, n, row_begin, row_end);

    row_begin = row_end;
  }

  for (std::thread& worker : workers) {
    worker.join();
  }
}

double compute_gflops(double total_flops, double elapsed_ms) {
  const double elapsed_seconds = elapsed_ms / 1.0e3;
  return total_flops / elapsed_seconds / 1.0e9;
}

double compute_matmul_flops(int m, int k, int n, int iterations) {
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

void run_cublas_fp32(cublasHandle_t handle,
                     const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int m,
                     int k,
                     int n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // cuBLAS is column-major by default. For row-major C = A * B, compute C^T = B^T * A^T.
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

}  // namespace

int main() {
  constexpr int kM = kMatrixSize;
  constexpr int kK = kMatrixSize;
  constexpr int kN = kMatrixSize;

  const size_t bytes_a = static_cast<size_t>(kM) * kK * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(kK) * kN * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(kM) * kN * sizeof(float);

  std::vector<float> h_a(kM * kK);
  std::vector<float> h_b(kK * kN);
  std::vector<float> h_cpu(kM * kN, 0.0f);
  std::vector<float> h_gpu_custom(kM * kN, 0.0f);
  std::vector<float> h_gpu_cublas(kM * kN, 0.0f);

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kK; ++col) {
      h_a[row * kK + col] = 0.25f + static_cast<float>((row + col) % 11) * 0.125f;
    }
  }
  for (int row = 0; row < kK; ++row) {
    for (int col = 0; col < kN; ++col) {
      h_b[row * kN + col] = 0.5f + static_cast<float>((row * 3 + col) % 13) * 0.0625f;
    }
  }

  print_cpu_info();
  print_device_summary();

  const unsigned int cpu_workers = get_cpu_worker_count(kM);

  matmul_cpu_parallel(h_a, h_b, h_cpu, kM, kK, kN, cpu_workers);

  const auto cpu_start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < kCpuBenchmarkIterations; ++iter) {
    matmul_cpu_parallel(h_a, h_b, h_cpu, kM, kK, kN, cpu_workers);
  }
  const auto cpu_stop = std::chrono::steady_clock::now();
  const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c_custom = nullptr;
  float* d_c_cublas = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
  CUDA_CHECK(cudaMalloc(&d_c_custom, bytes_c));
  CUDA_CHECK(cudaMalloc(&d_c_cublas, bytes_c));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));

  const dim3 block(kTile, kTile);
  const dim3 grid(div_up(kN, kTile), div_up(kM, kTile));

  tiled_matmul<<<grid, block>>>(d_a, d_b, d_c_custom, kM, kK, kN);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasHandle_t cublas_handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH));
  run_cublas_fp32(cublas_handle, d_a, d_b, d_c_cublas, kM, kK, kN);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t custom_start{};
  cudaEvent_t custom_stop{};
  cudaEvent_t cublas_start{};
  cudaEvent_t cublas_stop{};
  CUDA_CHECK(cudaEventCreate(&custom_start));
  CUDA_CHECK(cudaEventCreate(&custom_stop));
  CUDA_CHECK(cudaEventCreate(&cublas_start));
  CUDA_CHECK(cudaEventCreate(&cublas_stop));

  CUDA_CHECK(cudaEventRecord(custom_start));
  for (int iter = 0; iter < kGpuBenchmarkIterations; ++iter) {
    tiled_matmul<<<grid, block>>>(d_a, d_b, d_c_custom, kM, kK, kN);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(custom_stop));
  CUDA_CHECK(cudaEventSynchronize(custom_stop));

  float custom_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&custom_kernel_ms, custom_start, custom_stop));

  CUDA_CHECK(cudaEventRecord(cublas_start));
  for (int iter = 0; iter < kGpuBenchmarkIterations; ++iter) {
    run_cublas_fp32(cublas_handle, d_a, d_b, d_c_cublas, kM, kK, kN);
  }
  CUDA_CHECK(cudaEventRecord(cublas_stop));
  CUDA_CHECK(cudaEventSynchronize(cublas_stop));

  float cublas_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cublas_kernel_ms, cublas_start, cublas_stop));

  CUDA_CHECK(cudaMemcpy(h_gpu_custom.data(), d_c_custom, bytes_c, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_gpu_cublas.data(), d_c_cublas, bytes_c, cudaMemcpyDeviceToHost));

  const double cpu_total_flops = compute_matmul_flops(kM, kK, kN, kCpuBenchmarkIterations);
  const double gpu_total_flops = compute_matmul_flops(kM, kK, kN, kGpuBenchmarkIterations);
  const double cpu_gflops = compute_gflops(cpu_total_flops, cpu_ms);
  const double custom_gflops = compute_gflops(gpu_total_flops, custom_kernel_ms);
  const double cublas_gflops = compute_gflops(gpu_total_flops, cublas_kernel_ms);
  const double cpu_tflops = cpu_gflops / 1.0e3;
  const double custom_tflops = custom_gflops / 1.0e3;
  const double cublas_tflops = cublas_gflops / 1.0e3;
  const double gpu_peak_tflops = estimate_gpu_peak_fp32_tflops();
  const double custom_peak_utilization =
      gpu_peak_tflops > 0.0 ? (custom_tflops / gpu_peak_tflops) * 100.0 : 0.0;
  const double cublas_peak_utilization =
      gpu_peak_tflops > 0.0 ? (cublas_tflops / gpu_peak_tflops) * 100.0 : 0.0;
  const double arithmetic_intensity =
      (2.0 * static_cast<double>(kM) * static_cast<double>(kK) * static_cast<double>(kN)) /
      (static_cast<double>(kM * kK + kK * kN + kM * kN) * sizeof(float));
  const float custom_max_error = compute_max_error(h_cpu, h_gpu_custom);
  const float cublas_max_error = compute_max_error(h_cpu, h_gpu_cublas);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Benchmark: FP32 GEMM throughput compare\n";
  std::cout << "Matrix shape: A(" << kM << " x " << kK << "), B(" << kK << " x " << kN
            << "), C(" << kM << " x " << kN << ")\n";
  std::cout << "Approx arithmetic intensity: " << arithmetic_intensity << " FLOP/byte\n";
  std::cout << "CPU workers: " << cpu_workers << '\n';
  std::cout << "CPU benchmark iterations: " << kCpuBenchmarkIterations << '\n';
  std::cout << "GPU benchmark iterations: " << kGpuBenchmarkIterations << '\n';
  std::cout << "CPU total time: " << cpu_ms << " ms"
            << " (avg " << cpu_ms / kCpuBenchmarkIterations << " ms)\n";
  std::cout << "CPU throughput: " << cpu_gflops << " GFLOP/s"
            << " (" << cpu_tflops << " TFLOP/s)\n";
  std::cout << "Custom tiled kernel time: " << custom_kernel_ms << " ms"
            << " (avg " << custom_kernel_ms / kGpuBenchmarkIterations << " ms)\n";
  std::cout << "Custom tiled kernel throughput: " << custom_gflops << " GFLOP/s"
            << " (" << custom_tflops << " TFLOP/s)\n";
  std::cout << "cuBLAS FP32 SGEMM time: " << cublas_kernel_ms << " ms"
            << " (avg " << cublas_kernel_ms / kGpuBenchmarkIterations << " ms)\n";
  std::cout << "cuBLAS FP32 SGEMM throughput: " << cublas_gflops << " GFLOP/s"
            << " (" << cublas_tflops << " TFLOP/s)\n";
  if (gpu_peak_tflops > 0.0) {
    std::cout << "Estimated GPU FP32 peak (CUDA-reported clock): " << gpu_peak_tflops
              << " TFLOP/s\n";
    std::cout << "Custom kernel utilization vs reported peak: " << custom_peak_utilization
              << " %\n";
    std::cout << "cuBLAS utilization vs reported peak: " << cublas_peak_utilization
              << " %\n";
  }
  std::cout << "Custom kernel speedup vs CPU: " << cpu_ms / custom_kernel_ms << "x\n";
  std::cout << "cuBLAS speedup vs CPU: " << cpu_ms / cublas_kernel_ms << "x\n";
  std::cout << "Custom kernel max error: " << custom_max_error << '\n';
  std::cout << "cuBLAS max error: " << cublas_max_error << '\n';
  std::cout << "Sample output: C[7, 11] = " << h_gpu_cublas[7 * kN + 11] << '\n';

  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaEventDestroy(custom_start));
  CUDA_CHECK(cudaEventDestroy(custom_stop));
  CUDA_CHECK(cudaEventDestroy(cublas_start));
  CUDA_CHECK(cudaEventDestroy(cublas_stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c_custom));
  CUDA_CHECK(cudaFree(d_c_cublas));
  return 0;
}
