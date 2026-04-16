#include "cuda_helpers.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// 这个 kernel 演示二维线程布局的基本写法。
// 当问题天然是“行 x 列”的结构时，用二维 block/grid 会更直观。
__global__ void matrix_add_2d(const float* a, const float* b, float* c, int width, int height) {
  // 先根据 block 和线程号计算当前线程负责的二维坐标 (x, y)。
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // 因为 grid 往往会“向上取整”覆盖整个矩阵边界，
  // 所以最后一圈 block 中可能有一些线程超出合法范围。
  // 这里必须做边界判断，避免越界访问。
  if (x < width && y < height) {
    // 虽然逻辑上是二维矩阵，但内存里依然按一维数组连续存放。
    // 所以访问元素时要把 (x, y) 映射回线性下标。
    const int index = y * width + x;
    c[index] = a[index] + b[index];
  }
}

int main() {
  constexpr int kWidth = 1024;
  constexpr int kHeight = 768;

  // 这里选一个 16x16 的 block，是矩阵类问题中很常见的起点。
  constexpr dim3 kBlockSize(16, 16);

  // 用向上取整的方式计算 grid 大小，保证全部元素都能被覆盖到。
  const dim3 grid_size(div_up(kWidth, static_cast<int>(kBlockSize.x)),
                       div_up(kHeight, static_cast<int>(kBlockSize.y)));
  const int count = kWidth * kHeight;
  const size_t bytes = static_cast<size_t>(count) * sizeof(float);

  std::vector<float> h_a(count);
  std::vector<float> h_b(count);
  std::vector<float> h_c(count);

  // 在 CPU 上构造一个容易验证的输入：
  // a[row, col] = row
  // b[row, col] = 0.5 * col
  // 这样结果就很容易人工推出来。
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      const int index = row * kWidth + col;
      h_a[index] = static_cast<float>(row);
      h_b[index] = 0.5f * static_cast<float>(col);
    }
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // 用二维 grid 和二维 block 启动 kernel。
  matrix_add_2d<<<grid_size, kBlockSize>>>(d_a, d_b, d_c, kWidth, kHeight);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  // 在 CPU 上重新按相同公式检查结果。
  float max_error = 0.0f;
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      const int index = row * kWidth + col;
      const float expected = static_cast<float>(row) + 0.5f * static_cast<float>(col);
      max_error = std::max(max_error, std::abs(h_c[index] - expected));
    }
  }

  std::cout << "2D matrix add finished.\n";
  std::cout << "Grid = (" << grid_size.x << ", " << grid_size.y << "), "
            << "Block = (" << kBlockSize.x << ", " << kBlockSize.y << ")\n";
  std::cout << "Max error: " << max_error << '\n';
  std::cout << "Sample output: c[10, 20] = " << h_c[10 * kWidth + 20] << '\n';

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}

/*
代码逻辑梳理
1. 在 CPU 上创建两个二维矩阵，但在内存里仍按一维数组连续存储。
2. 选定 16x16 的 block，并用向上取整计算二维 grid 尺寸。
3. 把矩阵数据从 CPU 拷到 GPU。
4. kernel 中每个线程先求出自己的 (x, y)，再映射回线性下标 index。
5. 线程只在合法边界内执行加法，避免最后一圈 block 越界。
6. 计算结束后把结果拷回 CPU，并逐元素验证矩阵加法是否正确。
*/
