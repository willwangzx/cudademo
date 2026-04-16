#include "cuda_helpers.cuh"

#include <cmath>
#include <iostream>
#include <vector>

// reduction 是 CUDA 里非常经典的协作型模式：
// 把很多元素一步步合并成更少的部分和，最后得到总和。
__global__ void reduce_sum(const float* input, float* partial_sums, int n) {
  // extern __shared__ 表示这块 shared memory 的大小在 kernel 启动时再指定。
  // 它只在当前 block 内可见，访问速度通常比 global memory 更快。
  extern __shared__ float shared[];

  const unsigned int tid = threadIdx.x;

  // 这里让每个线程先读两个元素，减少后续参与归约的线程数。
  // 一个 block 一次会覆盖 2 * blockDim.x 个输入元素。
  const unsigned int start = 2 * blockIdx.x * blockDim.x + tid;

  float sum = 0.0f;
  if (start < static_cast<unsigned int>(n)) {
    sum += input[start];
  }
  if (start + blockDim.x < static_cast<unsigned int>(n)) {
    sum += input[start + blockDim.x];
  }

  // 先把每个线程的局部和写进 shared memory。
  shared[tid] = sum;
  __syncthreads();

  // 下面是典型的块内树形归约。
  // 每一轮都把活跃线程数减半，直到 block 内只剩下 shared[0] 保存总和。
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  // 一个 block 只输出一个部分和，交给 CPU 或下一轮 reduction 继续处理。
  if (tid == 0) {
    partial_sums[blockIdx.x] = shared[0];
  }
}

int main() {
  constexpr int kNumElements = 1 << 22;
  constexpr int kThreadsPerBlock = 256;

  // 因为每个线程会先加载两个元素，所以 block 数要按 2 * blockDim 来算。
  const int blocks = div_up(kNumElements, kThreadsPerBlock * 2);
  const size_t input_bytes = static_cast<size_t>(kNumElements) * sizeof(float);
  const size_t partial_bytes = static_cast<size_t>(blocks) * sizeof(float);

  // 用全 1 数据初始化，这样最终和应该等于元素个数，便于验证。
  std::vector<float> h_input(kNumElements, 1.0f);
  std::vector<float> h_partial(blocks, 0.0f);

  float* d_input = nullptr;
  float* d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_partial, partial_bytes));

  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

  // 第三个启动参数是动态 shared memory 的字节数。
  // 这里给每个线程预留一个 float 大小的空间。
  reduce_sum<<<blocks, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(d_input,
                                                                              d_partial,
                                                                              kNumElements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // GPU 只完成了“每个 block 一个部分和”。
  // 为了让代码更容易读，这里把部分和拷回 CPU 再做最后一轮累加。
  CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, partial_bytes, cudaMemcpyDeviceToHost));

  double gpu_sum = 0.0;
  for (float value : h_partial) {
    gpu_sum += value;
  }

  const double expected = static_cast<double>(kNumElements);
  std::cout << "Shared-memory reduction finished.\n";
  std::cout << "Expected sum: " << expected << '\n';
  std::cout << "GPU sum      : " << gpu_sum << '\n';
  std::cout << "Abs error    : " << std::abs(gpu_sum - expected) << '\n';

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_partial));
  return 0;
}

/*
代码逻辑梳理
1. CPU 先准备一大段输入数据，并全部初始化为 1，方便检查总和。
2. GPU kernel 中每个线程先从 global memory 读取两个元素，得到局部和。
3. 各线程把局部和写入 shared memory，并通过 __syncthreads() 同步。
4. block 内用树形归约不断把 shared memory 中的数据两两相加。
5. 每个 block 最终只输出一个 partial sum 到 d_partial。
6. 把所有 partial sum 拷回 CPU，再做最后一轮累加，得到完整总和。
*/
