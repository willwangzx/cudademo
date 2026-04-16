#include "cuda_helpers.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// 这是一个最经典的“逐元素并行” kernel。
// 每个线程负责若干个元素，把 a[i] + b[i] 写到 c[i]。
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  // 当前线程的全局编号。
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  // stride 等于整个 grid 里一共有多少线程。
  // 用它可以让一个线程在处理完第一个元素后，继续跳着处理后面的元素。
  const int stride = blockDim.x * gridDim.x;

  // 这就是常见的 grid-stride loop。
  // 为什么这样写：
  // 1. 不要求“线程总数”必须刚好等于“数据总数”
  // 2. 当数据量更大时，一个线程也能处理多个位置
  // 3. 这是 CUDA 中非常通用的写法
  for (int i = global_id; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  // 一共处理多少个元素。
  constexpr int kNumElements = 1 << 20;

  // 每个 block 放多少线程。
  // 256 是一个很常见的起点，后续你可以尝试改成 128 或 512 比较效果。
  constexpr int kThreadsPerBlock = 256;

  // 总字节数 = 元素个数 * 每个元素大小。
  const size_t bytes = static_cast<size_t>(kNumElements) * sizeof(float);

  // h_ 前缀表示 host，也就是 CPU 内存中的数据。
  std::vector<float> h_a(kNumElements);
  std::vector<float> h_b(kNumElements);
  std::vector<float> h_c(kNumElements);

  // 在 CPU 上先准备输入数据。
  // 这么做是为了后面把输入拷到 GPU 上，并且还能在 CPU 端计算期望结果做校验。
  for (int i = 0; i < kNumElements; ++i) {
    h_a[i] = 0.5f * static_cast<float>(i % 100);
    h_b[i] = 1.0f + 0.25f * static_cast<float>(i % 7);
  }

  // d_ 前缀表示 device，也就是 GPU 显存中的数据指针。
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  // 在 GPU 上申请三段显存：
  // 输入 a、输入 b、输出 c。
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // 把 CPU 上准备好的输入数据拷到 GPU。
  // GPU kernel 不能直接访问 std::vector 管理的这块 host 内存。
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // 计算需要多少个 block 才能覆盖全部元素。
  // div_up 的作用是“向上取整”，避免最后剩下的元素没人处理。
  const int blocks = div_up(kNumElements, kThreadsPerBlock);

  // 用 event 测量 kernel 的执行时间。
  // 这里测的是 GPU 计算阶段，不包含最后拷回主机的时间。
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  vector_add<<<blocks, kThreadsPerBlock>>>(d_a, d_b, d_c, kNumElements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  // 计算完成后，把结果从 GPU 拷回 CPU。
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  // 在 CPU 上重新计算期望值，确认 GPU 结果正确。
  float max_error = 0.0f;
  for (int i = 0; i < kNumElements; ++i) {
    const float expected = h_a[i] + h_b[i];
    max_error = std::max(max_error, std::abs(h_c[i] - expected));
  }

  std::cout << "Vector add finished in " << elapsed_ms << " ms\n";
  std::cout << "Max error: " << max_error << '\n';
  std::cout << "Sample output:\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "  c[" << i << "] = " << h_c[i] << '\n';
  }

  // 释放 event 和显存资源。
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}

/*
代码逻辑梳理
1. 先在 CPU 上创建并初始化两个输入向量 h_a、h_b。
2. 在 GPU 上用 cudaMalloc 申请三块显存 d_a、d_b、d_c。
3. 用 cudaMemcpy 把输入数据从 CPU 拷到 GPU。
4. 根据数据规模和 block 大小计算 blocks，然后启动 vector_add kernel。
5. kernel 中每个线程通过 grid-stride loop 处理自己负责的若干元素。
6. 计算结束后，把 d_c 拷回 h_c，并在 CPU 上逐元素校验结果是否正确。
7. 最后输出耗时和样例结果，并释放所有资源。
*/
