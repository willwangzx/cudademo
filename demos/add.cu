#include "cuda_helpers.cuh"
#include "cpuinfo.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

__global__ void add_kernel(const float* x, float* y, int n) {
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = global_id; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

void add_cpu(const std::vector<float>& x, std::vector<float>& y) {
  for (int i = 0; i < static_cast<int>(y.size()); ++i) {
    y[i] = x[i] + y[i];
  }
}

double compute_gflops(size_t num_elements,
                      int iterations,
                      double flops_per_element,
                      double elapsed_ms) {
  const double total_flops =
      static_cast<double>(num_elements) * static_cast<double>(iterations) * flops_per_element;
  const double elapsed_seconds = elapsed_ms / 1.0e3;
  return total_flops / elapsed_seconds / 1.0e9;
}

double compute_effective_bandwidth_gb_s(size_t num_elements,
                                        int iterations,
                                        double bytes_per_element,
                                        double elapsed_ms) {
  const double total_bytes =
      static_cast<double>(num_elements) * static_cast<double>(iterations) * bytes_per_element;
  const double elapsed_seconds = elapsed_ms / 1.0e3;
  return total_bytes / elapsed_seconds / 1.0e9;
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

int main() {
  constexpr int kNumElements = 1 << 24;
  constexpr int kThreadsPerBlock = 1024;
  constexpr int kIterations = 10;
  constexpr double kFlopsPerElement = 1.0;  // y = x + y
  constexpr double kKernelBytesPerElement =
      3.0 * sizeof(float);  // read x/y, write y

  const size_t bytes = static_cast<size_t>(kNumElements) * sizeof(float);

  std::vector<float> h_x(kNumElements);
  std::vector<float> h_y_initial(kNumElements);
  std::vector<float> h_cpu_iterated(kNumElements);
  std::vector<float> h_cpu_single_pass(kNumElements);
  std::vector<float> h_gpu_iterated(kNumElements);
  std::vector<float> h_gpu_total(kNumElements);

  print_cpu_info();
  print_device_summary();

  for (int i = 0; i < kNumElements; ++i) {
    h_x[i] = 0.5f * static_cast<float>(i % 100);
    h_y_initial[i] = 1.0f + 0.25f * static_cast<float>(i % 7);
  }

  float* d_x = nullptr;
  float* d_y = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y_initial.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = div_up(kNumElements, kThreadsPerBlock);

  // Warm up once so first-launch overhead does not skew the measured timings.
  add_kernel<<<blocks, kThreadsPerBlock>>>(d_x, d_y, kNumElements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(d_y, h_y_initial.data(), bytes, cudaMemcpyHostToDevice));

  h_cpu_iterated = h_y_initial;
  const auto cpu_start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < kIterations; ++iter) {
    add_cpu(h_x, h_cpu_iterated);
  }
  const auto cpu_stop = std::chrono::steady_clock::now();
  const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

  h_cpu_single_pass = h_y_initial;
  add_cpu(h_x, h_cpu_single_pass);

  cudaEvent_t kernel_start{};
  cudaEvent_t kernel_stop{};
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  CUDA_CHECK(cudaMemcpy(d_y, h_y_initial.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(kernel_start));
  for (int iter = 0; iter < kIterations; ++iter) {
    add_kernel<<<blocks, kThreadsPerBlock>>>(d_x, d_y, kNumElements);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  float gpu_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, kernel_start, kernel_stop));
  CUDA_CHECK(cudaMemcpy(h_gpu_iterated.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  const auto gpu_total_start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < kIterations; ++iter) {
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y_initial.data(), bytes, cudaMemcpyHostToDevice));
    add_kernel<<<blocks, kThreadsPerBlock>>>(d_x, d_y, kNumElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_gpu_total.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  }
  const auto gpu_total_stop = std::chrono::steady_clock::now();
  const double gpu_total_ms =
      std::chrono::duration<double, std::milli>(gpu_total_stop - gpu_total_start).count();

  const double cpu_gflops =
      compute_gflops(kNumElements, kIterations, kFlopsPerElement, cpu_ms);
  const double gpu_kernel_gflops =
      compute_gflops(kNumElements, kIterations, kFlopsPerElement, gpu_kernel_ms);
  const double gpu_total_gflops =
      compute_gflops(kNumElements, kIterations, kFlopsPerElement, gpu_total_ms);

  const double cpu_bandwidth_gb_s = compute_effective_bandwidth_gb_s(
      kNumElements, kIterations, kKernelBytesPerElement, cpu_ms);
  const double gpu_kernel_bandwidth_gb_s = compute_effective_bandwidth_gb_s(
      kNumElements, kIterations, kKernelBytesPerElement, gpu_kernel_ms);

  const double gpu_peak_tflops = estimate_gpu_peak_fp32_tflops();
  const double cpu_tflops = cpu_gflops / 1.0e3;
  const double gpu_kernel_tflops = gpu_kernel_gflops / 1.0e3;
  const double gpu_total_tflops = gpu_total_gflops / 1.0e3;
  const double arithmetic_intensity = kFlopsPerElement / kKernelBytesPerElement;
  const double gpu_peak_utilization =
      gpu_peak_tflops > 0.0 ? (gpu_kernel_tflops / gpu_peak_tflops) * 100.0 : 0.0;

  const float iterated_max_error = compute_max_error(h_cpu_iterated, h_gpu_iterated);
  const float end_to_end_max_error = compute_max_error(h_cpu_single_pass, h_gpu_total);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Operation: y = x + y (in place)\n";
  std::cout << "Elements: " << kNumElements << '\n';
  std::cout << "Iterations: " << kIterations << '\n';
  std::cout << "Arithmetic intensity: " << arithmetic_intensity << " FLOP/byte\n";
  std::cout << "CPU total time (single-thread): " << cpu_ms << " ms"
            << " (avg " << cpu_ms / kIterations << " ms)\n";
  std::cout << "CPU throughput: " << cpu_gflops << " GFLOP/s"
            << " (" << cpu_tflops << " TFLOP/s)\n";
  std::cout << "CPU effective bandwidth: " << cpu_bandwidth_gb_s << " GB/s\n";
  std::cout << "GPU kernel time: " << gpu_kernel_ms << " ms"
            << " (avg " << gpu_kernel_ms / kIterations << " ms)\n";
  std::cout << "GPU kernel throughput: " << gpu_kernel_gflops << " GFLOP/s"
            << " (" << gpu_kernel_tflops << " TFLOP/s)\n";
  std::cout << "GPU effective bandwidth: " << gpu_kernel_bandwidth_gb_s << " GB/s\n";
  std::cout << "GPU total time (H2D + kernel + D2H): " << gpu_total_ms << " ms"
            << " (avg " << gpu_total_ms / kIterations << " ms)\n";
  std::cout << "GPU end-to-end throughput: " << gpu_total_gflops << " GFLOP/s"
            << " (" << gpu_total_tflops << " TFLOP/s)\n";
  if (gpu_peak_tflops > 0.0) {
    std::cout << "Estimated GPU FP32 peak (FMA-based): " << gpu_peak_tflops << " TFLOP/s\n";
    std::cout << "Kernel peak utilization: " << gpu_peak_utilization << " %\n";
  }
  std::cout << "Kernel-only speedup vs CPU: " << cpu_ms / gpu_kernel_ms << "x\n";
  std::cout << "End-to-end speedup vs CPU: " << cpu_ms / gpu_total_ms << "x\n";
  std::cout << "Iterated kernel max error: " << iterated_max_error << '\n';
  std::cout << "Single-pass end-to-end max error: " << end_to_end_max_error << '\n';
  std::cout << "Sample after " << kIterations << " iterations: y[0] = " << h_gpu_iterated[0]
            << ", y[1] = " << h_gpu_iterated[1]
            << ", y[2] = " << h_gpu_iterated[2] << '\n';

  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
