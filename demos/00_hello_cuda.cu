#include "cuda_helpers.cuh"

#include <iostream>

// __global__ 表示这是一个 kernel:
// 1. 由 CPU 端代码启动
// 2. 真正执行位置在 GPU 上
// 3. 会有很多线程同时执行这段函数体
__global__ void hello_from_gpu() {
  // 每个线程都能拿到自己所在的 block 编号和 block 内线程编号。
  // 这一行把它们拼成一个“全局线程编号”，这是后续几乎所有 CUDA demo
  // 都会反复用到的索引公式。
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  // 这里用 printf 只是为了观察 GPU 上到底启动了多少线程。
  // 注意：每个线程都会执行一次，所以输出会有很多行。
  printf("Hello from GPU:  global_id=%d * %d = %d\n",
         blockIdx.x,
         threadIdx.x,
         global_id);
}

int main() {
  // 先打印当前 GPU 的基本信息，帮助我们确认程序正在使用哪张卡。
  print_device_summary();

  // 这里决定 kernel 的启动规模：
  // kBlocks = 启动多少个线程块
  // kThreadsPerBlock = 每个线程块中放多少个线程
  // 总线程数 = kBlocks * kThreadsPerBlock
  constexpr int kBlocks = 4;
  constexpr int kThreadsPerBlock = 16;

  std::cout << "\nLaunching hello_from_gpu<<<" << kBlocks << ", " << kThreadsPerBlock << ">>> ...\n";

  // <<<grid, block>>> 是 CUDA 最有代表性的启动语法。
  // 这里表示：
  // 1. 启动 kBlocks 个 block
  // 2. 每个 block 有 kThreadsPerBlock 个线程
  hello_from_gpu<<<kBlocks, kThreadsPerBlock>>>();

  // 先检查“启动 kernel 这件事”有没有立刻出错，例如参数非法。
  CUDA_CHECK(cudaGetLastError());

  // kernel 启动通常是异步的。
  // 如果不在这里等待，CPU 可能先结束 main，GPU 还没来得及把输出刷出来。
  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "Kernel finished successfully.\n";
  return 0;
}

/*
代码逻辑梳理
1. CPU 端 main() 先打印当前 GPU 信息，确认运行环境。
2. main() 设定 grid 和 block 的大小，决定总共会启动多少个 GPU 线程。
3. CPU 用 <<<grid, block>>> 语法启动 hello_from_gpu kernel。
4. GPU 上的每个线程都会计算自己的 global_id，并各自执行一次 printf。
5. CPU 通过 cudaDeviceSynchronize() 等待 GPU 执行完成，再结束程序。
*/
