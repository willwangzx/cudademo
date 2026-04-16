#include "cuda_helpers.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
// tile 大小决定：
// 1. 每个 block 负责输出矩阵中的一个 kTile x kTile 小块
// 2. 每轮从 A 和 B 中搬进 shared memory 的数据规模
constexpr int kTile = 16;
}

// 这是一个“基于 shared memory 的 tiled 矩阵乘法”示例。
// 目标是让你理解：shared memory 的价值不只是快，而是可以复用数据。
__global__ void tiled_matmul(const float* a,
                             const float* b,
                             float* c,
                             int m,
                             int k,
                             int n) {
  // 每个 block 都会把 A 和 B 的一个 tile 搬进 shared memory。
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];

  // 当前线程负责输出矩阵 C 中哪个元素。
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  float value = 0.0f;

  // 沿着 K 维度一段一段地处理。
  // 每处理一段，就把 A 和 B 对应的小块装入 shared memory。
  const int tiles = div_up(k, kTile);

  for (int tile = 0; tile < tiles; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;

    // block 内线程协作加载 A 的一个 tile。
    // 超出边界的位置补 0，这样非整齐尺寸也能安全工作。
    tile_a[threadIdx.y][threadIdx.x] =
      (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;

    // block 内线程协作加载 B 的一个 tile。
    tile_b[threadIdx.y][threadIdx.x] =
      (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    // 确保 tile 已经全部装载完成，再开始计算。
    __syncthreads();

    // 使用 shared memory 中的数据做一小段乘加。
    // 这样一份 tile 数据会被 block 内多个线程重复使用，提高数据复用率。
    for (int i = 0; i < kTile; ++i) {
      value += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }

    // 确保本轮计算全部结束，再进入下一轮 tile 装载。
    __syncthreads();
  }

  // 最后把每个线程负责的那个输出元素写回 global memory。
  if (row < m && col < n) {
    c[row * n + col] = value;
  }
}

int main() {
  constexpr int kM = 128;
  constexpr int kK = 256;
  constexpr int kN = 192;

  const size_t bytes_a = static_cast<size_t>(kM) * kK * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(kK) * kN * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(kM) * kN * sizeof(float);

  std::vector<float> h_a(kM * kK);
  std::vector<float> h_b(kK * kN);
  std::vector<float> h_c(kM * kN, 0.0f);
  std::vector<float> h_reference(kM * kN, 0.0f);

  // 初始化输入矩阵。
  // 这里用简单公式生成数据，而不是随机数，方便调试和复现。
  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kK; ++col) {
      h_a[row * kK + col] = static_cast<float>((row + col) % 7);
    }
  }
  for (int row = 0; row < kK; ++row) {
    for (int col = 0; col < kN; ++col) {
      h_b[row * kN + col] = static_cast<float>((row * 2 + col) % 5);
    }
  }

  // 先在 CPU 上算一个参考答案。
  // 为什么值得这样做：
  // 1. 矩阵乘法逻辑更复杂，先有对照更安心
  // 2. 后面你继续优化 kernel 时，也能快速验证结果有没有被搞错
  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      float sum = 0.0f;
      for (int inner = 0; inner < kK; ++inner) {
        sum += h_a[row * kK + inner] * h_b[inner * kN + col];
      }
      h_reference[row * kN + col] = sum;
    }
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
  CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));

  // 一个 block 对应输出矩阵中的一个 16x16 小块。
  const dim3 block(kTile, kTile);
  const dim3 grid(div_up(kN, kTile), div_up(kM, kTile));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  tiled_matmul<<<grid, block>>>(d_a, d_b, d_c, kM, kK, kN);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));

  // 把 GPU 结果和 CPU 参考答案逐元素比较。
  float max_error = 0.0f;
  for (int i = 0; i < kM * kN; ++i) {
    max_error = std::max(max_error, std::abs(h_c[i] - h_reference[i]));
  }

  std::cout << "Tiled matrix multiplication finished in " << elapsed_ms << " ms\n";
  std::cout << "Grid = (" << grid.x << ", " << grid.y << "), "
            << "Block = (" << block.x << ", " << block.y << ")\n";
  std::cout << "Max error: " << max_error << '\n';
  std::cout << "Sample output: c[3, 5] = " << h_c[3 * kN + 5] << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}

/*
代码逻辑梳理
1. CPU 先准备矩阵 A、B，并在 CPU 上计算一份朴素矩阵乘法参考答案。
2. 把 A、B 拷到 GPU，准备让 GPU 计算输出矩阵 C。
3. GPU 中一个 block 负责 C 的一个 kTile x kTile 输出小块。
4. 每轮循环把 A 和 B 在 K 维上的一段 tile 协作加载到 shared memory。
5. block 内线程重复使用 shared memory 中的数据做乘加，减少重复全局访存。
6. 所有 tile 处理完后，每个线程把自己负责的 C[row, col] 写回显存。
7. 最后把结果拷回 CPU，并和参考答案比较，确认优化没有破坏正确性。
*/
