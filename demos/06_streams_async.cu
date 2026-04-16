#include "cuda_helpers.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

// 这个 kernel 和前面的 vector add 很像，
// 区别在于它只负责“单个 chunk”的计算。
// 后面我们会用多个 stream 并行处理多个 chunk。
__global__ void vector_add_chunk(const float* a, const float* b, float* c, int n) {
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = global_id; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int kNumStreams = 4;
  constexpr int kChunkSize = 1 << 20;
  constexpr int kThreadsPerBlock = 256;
  constexpr int kNumElements = kNumStreams * kChunkSize;
  const size_t chunk_bytes = static_cast<size_t>(kChunkSize) * sizeof(float);

  float* h_a = nullptr;
  float* h_b = nullptr;
  float* h_c = nullptr;

  // 用 cudaMallocHost 分配 pinned memory（固定页内存）。
  // 为什么这里要用它：
  // 1. 异步 memcpy 更常和 pinned memory 配合
  // 2. 更有机会让数据传输和计算发生重叠
  CUDA_CHECK(cudaMallocHost(&h_a, kNumElements * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_b, kNumElements * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_c, kNumElements * sizeof(float)));

  for (int i = 0; i < kNumElements; ++i) {
    h_a[i] = static_cast<float>(i % 11);
    h_b[i] = 3.0f + static_cast<float>(i % 5);
    h_c[i] = 0.0f;
  }

  // 每个 stream 都维护自己的一套 device 缓冲区。
  // 这样每个 chunk 可以在自己的 stream 上独立排队执行。
  cudaStream_t streams[kNumStreams];
  float* d_a[kNumStreams]{};
  float* d_b[kNumStreams]{};
  float* d_c[kNumStreams]{};

  for (int s = 0; s < kNumStreams; ++s) {
    CUDA_CHECK(cudaStreamCreate(&streams[s]));
    CUDA_CHECK(cudaMalloc(&d_a[s], chunk_bytes));
    CUDA_CHECK(cudaMalloc(&d_b[s], chunk_bytes));
    CUDA_CHECK(cudaMalloc(&d_c[s], chunk_bytes));
  }

  const int blocks = div_up(kChunkSize, kThreadsPerBlock);
  const auto start = std::chrono::steady_clock::now();

  // 把总数据切成多个 chunk，每个 chunk 交给一个 stream。
  // 每个 stream 上都按“拷入 -> 计算 -> 拷回”的顺序排队。
  // 不同 stream 之间则有机会并发执行，从而重叠传输和计算。
  for (int s = 0; s < kNumStreams; ++s) {
    const int offset = s * kChunkSize;

    CUDA_CHECK(cudaMemcpyAsync(d_a[s], h_a + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(d_b[s], h_b + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[s]));
    vector_add_chunk<<<blocks, kThreadsPerBlock, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s], kChunkSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c[s], chunk_bytes, cudaMemcpyDeviceToHost, streams[s]));
  }

  // 虽然每个 stream 内是有序的，但主机端还是要等所有 stream 都完成，
  // 才能安全读取最终结果。
  for (int s = 0; s < kNumStreams; ++s) {
    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
  }

  const auto stop = std::chrono::steady_clock::now();
  const double elapsed_ms =
    std::chrono::duration<double, std::milli>(stop - start).count();

  // 把所有 chunk 拼回来的结果在 CPU 上统一校验。
  float max_error = 0.0f;
  for (int i = 0; i < kNumElements; ++i) {
    const float expected = h_a[i] + h_b[i];
    max_error = std::max(max_error, std::abs(h_c[i] - expected));
  }

  std::cout << "Async stream pipeline finished in " << elapsed_ms << " ms\n";
  std::cout << "Streams used: " << kNumStreams << '\n';
  std::cout << "Max error   : " << max_error << '\n';
  std::cout << "Sample output: c[42] = " << h_c[42] << '\n';

  for (int s = 0; s < kNumStreams; ++s) {
    CUDA_CHECK(cudaFree(d_a[s]));
    CUDA_CHECK(cudaFree(d_b[s]));
    CUDA_CHECK(cudaFree(d_c[s]));
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }

  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  CUDA_CHECK(cudaFreeHost(h_c));
  return 0;
}

/*
代码逻辑梳理
1. 用 pinned memory 在 CPU 端准备输入和输出缓冲区，为异步拷贝做准备。
2. 把总数据切成多个 chunk，并为每个 stream 准备独立的 device 缓冲区。
3. 在每个 stream 上依次排队：HostToDevice 拷贝、kernel 计算、DeviceToHost 拷贝。
4. 同一个 stream 内操作保持顺序，不同 stream 之间则有机会并发执行。
5. 主机端等待所有 stream 完成后，再统一检查结果是否正确。
6. 最后释放所有 stream、显存和 pinned memory 资源。
*/
