# CUDA 学习清单

这份清单的目标不是“看过”，而是“做过、跑过、改过、能讲明白”。

## 一、准备阶段

- [ ] 会写基础 C++：函数、引用、指针、`std::vector`
- [ ] 能看懂简单的 CMake 项目
- [ ] 知道 CPU 串行循环和并行循环的区别
- [ ] 已安装 CUDA Toolkit，并确认 `nvcc --version` 可用
- [ ] 已确认本机有 NVIDIA GPU，且 `nvidia-smi` 可用

## 二、CUDA 基础概念

- [ ] 能解释 host / device / kernel
- [ ] 能解释 `__global__`、`__device__`、`__host__`
- [ ] 能解释 `threadIdx`、`blockIdx`、`blockDim`、`gridDim`
- [ ] 能根据数据规模计算 grid 和 block
- [ ] 能写出一维索引和二维索引的映射公式
- [ ] 能解释为什么很多 CUDA kernel 需要边界判断
- [ ] 能解释 grid-stride loop 是干什么的

## 三、内存管理

- [ ] 会用 `cudaMalloc` / `cudaFree`
- [ ] 会用 `cudaMemcpy`
- [ ] 知道 Host 内存和 Device 内存不是同一块
- [ ] 知道 `cudaMallocManaged` 是什么
- [ ] 知道 pinned memory 的基本用途
- [ ] 了解 shared memory、global memory、register 的区别
- [ ] 能解释 coalesced memory access 的基本含义

## 四、线程协作与同步

- [ ] 知道 block 内线程可以通过 shared memory 协作
- [ ] 知道 `__syncthreads()` 的作用和限制
- [ ] 能独立写出块内 reduction 的基本结构
- [ ] 知道 warp 的概念
- [ ] 知道分支发散会影响执行效率

## 五、性能优化入门

- [ ] 能解释为什么 block size 会影响性能
- [ ] 能解释 occupancy 是什么
- [ ] 能解释为什么矩阵乘法常用 tiling
- [ ] 能解释“数据复用”为什么重要
- [ ] 能解释计算和传输重叠的基本思路
- [ ] 知道 stream 是做什么的
- [ ] 能区分“先保证正确”与“再做优化”这两个阶段

## 六、工具链与调试

- [ ] 会看 `cudaGetLastError()` / `cudaDeviceSynchronize()` 的报错
- [ ] 会使用 `printf` 做基础 kernel 调试
- [ ] 知道 `compute-sanitizer` 可以帮助查什么问题
- [ ] 至少知道 Nsight Systems 和 Nsight Compute 是干什么的
- [ ] 会比较 CPU 结果与 GPU 结果验证正确性

## 七、推荐实践顺序

- [ ] 跑通 `00_hello_cuda`
- [ ] 跑通 `01_vector_add`
- [ ] 跑通 `02_matrix_add_2d`
- [ ] 跑通 `03_unified_memory`
- [ ] 跑通 `04_reduction_shared`
- [ ] 跑通 `05_tiled_matmul`
- [ ] 跑通 `06_streams_async`

## 八、每个 demo 跑完后你都应该做的事

- [ ] 自己解释这段 kernel 的索引公式
- [ ] 自己画出 grid / block / thread 的结构图
- [ ] 改一个参数并重新运行
- [ ] 人工构造一个错误输入，验证程序是否还能正确处理
- [ ] 写出 CPU 对照版本
- [ ] 打印 3 到 5 个样例结果做 sanity check

## 九、进阶专题

- [ ] 原子操作 `atomicAdd`
- [ ] 前缀和 scan
- [ ] 直方图 histogram
- [ ] 卷积 convolution
- [ ] warp-level primitive
- [ ] Tensor Core / WMMA
- [ ] 多 GPU 基础概念
- [ ] Thrust / CUB / cuBLAS / cuDNN 的定位

## 十、你可以用来检验自己是否真的学会了

如果下面这些问题你能用自己的话讲清楚，说明你已经入门成功：

- [ ] 为什么 CUDA 程序一般要先在 CPU 上准备数据，再把数据拷到 GPU
- [ ] 为什么 kernel 通常不能假设“线程数刚好等于数据量”
- [ ] 为什么 shared memory 能加速某些算法
- [ ] 为什么矩阵乘法适合做 tiling
- [ ] 为什么异步拷贝通常要配合 pinned memory
- [ ] 为什么“结果正确”永远优先于“跑得更快”

## 十一、建议你完成的阶段性项目

- [ ] 用 CUDA 写一个图像反色 / 灰度化程序
- [ ] 用 CUDA 写一个 blur 或 box filter
- [ ] 用 CUDA 实现直方图统计
- [ ] 用 CUDA 实现 prefix sum
- [ ] 对矩阵乘法做一次“朴素版 -> tile 版”的优化对比
- [ ] 写一篇自己的学习笔记，总结你对 block、warp、shared memory 的理解
