#include "cuda_helpers.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

#if CUDART_VERSION >= 12020
cudaError_t prefetch_to_device(const void* ptr, size_t bytes, int device, cudaStream_t stream = 0) {
  const cudaMemLocation location{cudaMemLocationTypeDevice, device};
  return cudaMemPrefetchAsync(ptr, bytes, location, 0, stream);
}

cudaError_t prefetch_to_host(const void* ptr, size_t bytes, cudaStream_t stream = 0) {
  const cudaMemLocation location{cudaMemLocationTypeHost, 0};
  return cudaMemPrefetchAsync(ptr, bytes, location, 0, stream);
}
#else
cudaError_t prefetch_to_device(const void* ptr, size_t bytes, int device, cudaStream_t stream = 0) {
  return cudaMemPrefetchAsync(ptr, bytes, device, stream);
}

cudaError_t prefetch_to_host(const void* ptr, size_t bytes, cudaStream_t stream = 0) {
  return cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream);
}
#endif

}  // namespace

// SAXPY 是线性代数中的经典小例子：
// y = alpha * x + y
// 它很适合用来演示“逐元素并行”以及 Unified Memory 的用法。
__global__ void saxpy(float alpha, const float* x, float* y, int n) {
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = global_id; i < n; i += stride) {
    y[i] = alpha * x[i] + y[i];
  }
}

int main() {
  constexpr int kNumElements = 1 << 22;
  constexpr int kThreadsPerBlock = 256;
  const size_t bytes = static_cast<size_t>(kNumElements) * sizeof(float);
  constexpr float kAlpha = 2.5f;

  float* x = nullptr;
  float* y = nullptr;

  // 这里不再分别写 cudaMalloc + cudaMemcpy，而是使用 Unified Memory。
  // 这样分配出来的内存对 CPU 和 GPU 都“可见”，便于学习和原型开发。
  CUDA_CHECK(cudaMallocManaged(&x, bytes));
  CUDA_CHECK(cudaMallocManaged(&y, bytes));

  // 直接在 CPU 上初始化托管内存。
  for (int i = 0; i < kNumElements; ++i) {
    x[i] = static_cast<float>(i % 17);
    y[i] = 1.0f;
  }

  // 查询当前设备，以及它是否支持更完整的 managed memory 并发访问能力。
  // 不同平台上 Unified Memory 的行为可能略有差异，所以这里做一次能力探测。
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int concurrent_managed_access = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&concurrent_managed_access,
                                    cudaDevAttrConcurrentManagedAccess,
                                    device));

  bool used_prefetch = false;

  // 如果平台支持，就主动把数据预取到 GPU。
  // 为什么要预取：
  // 1. 可以减少第一次访问时的页面迁移开销
  // 2. 能帮助你理解 Unified Memory 背后的“数据迁移”概念
  // 如果平台不支持，就回退到按需迁移，程序依然可以正常运行。
  if (concurrent_managed_access != 0) {
    const cudaError_t prefetch_x = prefetch_to_device(x, bytes, device);
    const cudaError_t prefetch_y = prefetch_to_device(y, bytes, device);
    if (prefetch_x == cudaSuccess && prefetch_y == cudaSuccess) {
      used_prefetch = true;
    } else {
      std::cout << "Prefetch is not available on this setup, fallback to on-demand migration.\n";
    }
  } else {
    std::cout << "Device does not report concurrent managed access, fallback to on-demand migration.\n";
  }

  const int blocks = div_up(kNumElements, kThreadsPerBlock);
  saxpy<<<blocks, kThreadsPerBlock>>>(kAlpha, x, y, kNumElements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 如果之前成功预取到 GPU，这里再尝试把结果预取回 CPU。
  // 这样 CPU 后面读取 y 时，数据更可能已经在主机侧可用。
  if (used_prefetch) {
    const cudaError_t prefetch_back = prefetch_to_host(y, bytes);
    if (prefetch_back != cudaSuccess) {
      std::cout << "Host prefetch skipped: " << cudaGetErrorString(prefetch_back) << '\n';
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // 即使使用 Unified Memory，也照样应该在 CPU 上做结果校验。
  float max_error = 0.0f;
  for (int i = 0; i < kNumElements; ++i) {
    const float expected = kAlpha * static_cast<float>(i % 17) + 1.0f;
    max_error = std::max(max_error, std::abs(y[i] - expected));
  }

  std::cout << "Unified memory SAXPY finished.\n";
  std::cout << "Max error: " << max_error << '\n';
  std::cout << "Sample output: y[123] = " << y[123] << '\n';

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  return 0;
}

/*
代码逻辑梳理
1. 用 cudaMallocManaged 分配一块 CPU 和 GPU 都能访问的托管内存。
2. 在 CPU 上直接初始化 x 和 y，不再手写 cudaMemcpy。
3. 查询设备是否支持更完整的 Unified Memory 访问能力，并尝试做 prefetch。
4. 启动 SAXPY kernel，让 GPU 并行执行 y = alpha * x + y。
5. 如果平台支持，就把结果再预取回 CPU；如果不支持，就依靠按需迁移。
6. 最后在 CPU 上验证结果正确性，并释放托管内存。
*/
