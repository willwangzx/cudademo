# CUDA Demo 学习项目

这个仓库按“边写 demo 边学 CUDA”的思路组织。你只要顺着 demo 顺序跑一遍，再按每个 demo 后面的建议自己改一改，进步会比只看概念快很多。

## Prerequisites
- NVIDIA GPU（建议支持 Compute Capability 3.0 及以上的设备）
- CUDA Toolkit（建议版本 11.0 及以上）[https://developer.nvidia.com/cuda-downloads]
- C++ 编译器（支持 C++11 或更高版本）
- CMake（建议版本 3.10 及以上）

## 你会学到什么

- CUDA 程序的基本结构：host 代码、device 代码、kernel launch
- 线程组织方式：`threadIdx`、`blockIdx`、`blockDim`、`gridDim`
- 显存管理：`cudaMalloc`、`cudaMemcpy`、`cudaMallocManaged`
- 2D 网格与矩阵问题建模
- shared memory、`__syncthreads()`、块内归约
- tile 优化与矩阵乘法
- stream、异步拷贝、固定页内存

## 运行环境

这个仓库现在支持下面几种构建场景：

- Windows + Visual Studio + CUDA Toolkit：构建全部 CUDA demos
- Linux / WSL + `cmake` + `nvcc` + GCC/Clang：构建全部 CUDA demos
- 没有 CUDA 编译器的机器：仍然可以完成 CMake 配置与构建，此时会生成一个 host-only 的 `cuda_demo_info` 提示程序

当前这台机器已经验证过以下工具可用：

- CUDA Toolkit `13.2`
- `nvcc`
- NVIDIA GPU

## 构建方式

Windows 下推荐直接用仓库里的脚本，它会自动找到本机 CUDA Toolkit 并传给 CMake：

```powershell
.\build.ps1
```

脚本默认把构建结果放到 `build/`。如果你之前执行过不带 CUDA toolset 的 `cmake -S . -B build`，脚本也会自动重置这份失配缓存后再重新配置。

Linux / WSL 下推荐用新增的 Bash 脚本：

```bash
chmod +x ./build.sh
./build.sh
```

如果你想指定单独的构建目录或配置，可以传参数：

```bash
./build.sh build-linux Release
```

手动执行也可以，推荐这样写：

```powershell
cmake -S . -B build -T cuda="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
cmake --build build --config Release
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

如果你直接执行 `cmake -S . -B build`，在某些 Windows 环境里可能会看到 `No CUDA toolset found`。这不是代码问题，而是 CMake 没自动找到 Visual Studio 的 CUDA toolset。

如果当前机器没有 CUDA 编译器，CMake 不会再直接失败，而是会构建一个 `cuda_demo_info` 可执行文件，明确提示你当前只是 host-only 模式。

如果你希望“没有 CUDA 就立刻报错”，可以手动打开严格模式：

```bash
cmake -S . -B build -DCUDA_DEMO_REQUIRE_CUDA=ON
```

编译完成后，可执行文件通常在：

- `build/bin/Release/`（Visual Studio 这类多配置生成器）
- `build/bin/`（Ninja / NMake 这类单配置生成器）

如果你手动把二进制目录设成了别的名字，比如 `build-vs-cuda/`，那运行路径也要跟着变成对应目录。

## Demo 顺序

| 顺序 | 文件 | 核心主题 | 学完后你应该掌握 |
|---|---|---|---|
| 0 | `demos/00_hello_cuda.cu` | 第一个 kernel | 知道 GPU kernel 是怎么启动的 |
| 1 | `demos/01_vector_add.cu` | 一维并行 + 显存拷贝 | 会写最基础的数据并行程序 |
| 2 | `demos/02_matrix_add_2d.cu` | 二维网格 | 会把矩阵问题映射到线程块 |
| 3 | `demos/03_unified_memory.cu` | Unified Memory | 理解托管内存和预取 |
| 4 | `demos/04_reduction_shared.cu` | shared memory + 归约 | 理解线程协作和同步 |
| 5 | `demos/05_tiled_matmul.cu` | tile 优化 | 理解数据复用与共享内存价值 |
| 6 | `demos/06_streams_async.cu` | stream + 异步 pipeline | 理解拷贝与计算重叠的思路 |
| 7 | `demos/07_bucket_sort.cu` | histogram + scan + scatter + bucket sort | 练习把多个 CUDA 小模块串成完整 pipeline |

## 建议运行命令

```powershell
.\build\bin\Release\demo00_hello.exe
.\build\bin\Release\demo01_vector_add.exe
.\build\bin\Release\demo02_matrix_add_2d.exe
.\build\bin\Release\demo03_unified_memory.exe
.\build\bin\Release\demo04_reduction_shared.exe
.\build\bin\Release\demo05_tiled_matmul.exe
.\build\bin\Release\demo06_streams_async.exe
.\build\bin\Release\demo07_bucket_sort.exe
```

如果你的生成器不是 Visual Studio，也可能在下面这个目录：

```powershell
.\build\bin\demo00_hello.exe
```

如果你之前是手动构建到 `build-vs-cuda/`，那就运行：

```powershell
.\build-vs-cuda\bin\Release\demo00_hello.exe
```

Linux / WSL 上通常是：

```bash
./build/bin/demo00_hello
./build/bin/demo07_bucket_sort
```

如果当前机器没有 CUDA 编译器，那么构建完成后会看到：

```bash
./build/bin/cuda_demo_info
```

## 每个 demo 怎么学

### Demo 0: 第一个 CUDA 程序

看点：

- `__global__` 是什么
- `<<<grid, block>>>` 是什么
- 为什么要 `cudaDeviceSynchronize()`

建议你自己改：

- 改 block 数量和线程数量，观察输出
- 只打印前几个线程，理解全局线程编号

### Demo 1: Vector Add

看点：

- Host 数据如何拷贝到 Device
- grid-stride loop 为什么常用
- 为什么 kernel 里通常要做边界判断或 stride 循环

建议你自己改：

- 把 `float` 改成 `int`
- 把加法改成 `c[i] = 2 * a[i] + b[i]`
- 试试不同 block size，比如 `128`、`256`、`512`

### Demo 2: 2D Matrix Add

看点：

- `dim3` 的用法
- `(x, y)` 坐标如何映射回一维数组
- 为什么图像、矩阵、卷积类问题经常用二维 block

建议你自己改：

- 改成矩阵减法或阈值处理
- 尝试把 block size 改成 `32 x 8`、`8 x 32`

### Demo 3: Unified Memory

看点：

- `cudaMallocManaged` 和 `cudaMalloc` 的区别
- `cudaMemPrefetchAsync` 的作用
- 什么时候 Unified Memory 适合学习和原型开发

建议你自己改：

- 删掉 prefetch，再运行一次看看
- 把 SAXPY 改成别的逐元素运算

### Demo 4: Shared Memory Reduction

看点：

- shared memory 的生命周期
- `__syncthreads()` 为什么必须成对理解
- 归约为什么是 CUDA 高频题型

建议你自己改：

- 把输入改成非全 1 数据
- 尝试自己写 warp-level 优化版本

### Demo 5: Tiled Matrix Multiplication

看点：

- 为什么朴素矩阵乘法访问显存代价高
- tile 如何减少重复访存
- shared memory 不只是“快”，更重要的是“复用”

建议你自己改：

- 先自己写一个不使用 shared memory 的朴素版本
- 比较朴素版和 tile 版的耗时
- 修改矩阵尺寸，观察是否都能正确运行

### Demo 6: Streams + Async

看点：

- stream 的基本作用
- `cudaMemcpyAsync` 为什么经常和 pinned memory 一起出现
- 什么叫“传输和计算重叠”

建议你自己改：

- 把 stream 数改成 `2` 或 `8`
- 比较同步版本和异步版本的时间

## 推荐学习节奏

### 第 1 周

- 跑通 Demo 0, 1, 2
- 重点理解线程模型和 kernel launch

### 第 2 周

- 跑通 Demo 3, 4
- 重点理解内存模型、同步、shared memory

### 第 3 周

- 跑通 Demo 5
- 能独立解释 tile 优化为什么更快

### 第 4 周

- 跑通 Demo 6
- 开始接触 profile 工具和性能分析

## 下一步建议

学完这套 demo 后，推荐继续做下面 4 个小项目：

- 图像灰度化 / 模糊
- 直方图统计
- 前缀和 scan
- 简化版卷积或 GEMM 优化

详细学习清单见 [LEARNING_CHECKLIST.md](LEARNING_CHECKLIST.md)。
