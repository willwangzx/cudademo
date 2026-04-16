#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

inline void cuda_check(cudaError_t error, const char* call, const char* file, int line) {
  if (error != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << file << ":" << line << '\n'
              << "  call : " << call << '\n'
              << "  error: " << cudaGetErrorString(error) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

__host__ __device__ inline int div_up(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

using nvmlDevice_t = struct nvmlDevice_st*;
using nvmlReturn_t = int;

constexpr nvmlReturn_t kNvmlSuccess = 0;

using nvmlInitFn = nvmlReturn_t (*)();
using nvmlShutdownFn = nvmlReturn_t (*)();
using nvmlDeviceGetHandleByPciBusIdFn =
    nvmlReturn_t (*)(const char* pci_bus_id, nvmlDevice_t* device);
using nvmlDeviceGetNumGpuCoresFn =
    nvmlReturn_t (*)(nvmlDevice_t device, unsigned int* num_cores);

#if defined(_WIN32)
using NvmlModuleHandle = HMODULE;

inline NvmlModuleHandle load_nvml_library(const char* path) {
  return LoadLibraryA(path);
}

inline void* load_nvml_symbol(NvmlModuleHandle module, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(module, name));
}

inline void close_nvml_library(NvmlModuleHandle module) {
  if (module != nullptr) {
    FreeLibrary(module);
  }
}
#else
using NvmlModuleHandle = void*;

inline NvmlModuleHandle load_nvml_library(const char* path) {
  return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
}

inline void* load_nvml_symbol(NvmlModuleHandle module, const char* name) {
  return dlsym(module, name);
}

inline void close_nvml_library(NvmlModuleHandle module) {
  if (module != nullptr) {
    dlclose(module);
  }
}
#endif

inline NvmlModuleHandle try_load_nvml_module() {
#if defined(_WIN32)
  if (NvmlModuleHandle module = load_nvml_library("nvml.dll")) {
    return module;
  }

  char system_dir[MAX_PATH] = {};
  const UINT system_dir_length = GetSystemDirectoryA(system_dir, MAX_PATH);
  if (system_dir_length > 0 && system_dir_length < MAX_PATH) {
    std::string nvml_path(system_dir, system_dir_length);
    nvml_path += "\\nvml.dll";
    if (NvmlModuleHandle module = load_nvml_library(nvml_path.c_str())) {
      return module;
    }
  }

  char program_files[MAX_PATH] = {};
  const DWORD program_files_length =
      GetEnvironmentVariableA("ProgramW6432", program_files, MAX_PATH);
  if (program_files_length > 0 && program_files_length < MAX_PATH) {
    std::string nvml_path(program_files, program_files_length);
    nvml_path += "\\NVIDIA Corporation\\NVSMI\\nvml.dll";
    return load_nvml_library(nvml_path.c_str());
  }

  return nullptr;
#else
  static constexpr const char* kCandidates[] = {
      "libnvidia-ml.so.1",
      "libnvidia-ml.so",
      "/usr/lib/wsl/lib/libnvidia-ml.so.1",
      "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
      "/usr/lib64/libnvidia-ml.so.1",
  };

  for (const char* candidate : kCandidates) {
    if (NvmlModuleHandle module = load_nvml_library(candidate)) {
      return module;
    }
  }

  return nullptr;
#endif
}

inline unsigned int try_get_cuda_core_count(int device) {
  char pci_bus_id[32] = {};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), device));

  NvmlModuleHandle nvml_module = try_load_nvml_module();
  if (nvml_module == nullptr) {
    return 0;
  }

  const auto nvml_init =
      reinterpret_cast<nvmlInitFn>(load_nvml_symbol(nvml_module, "nvmlInit_v2"));
  const auto nvml_shutdown =
      reinterpret_cast<nvmlShutdownFn>(load_nvml_symbol(nvml_module, "nvmlShutdown"));
  const auto nvml_get_handle_by_pci_bus_id = reinterpret_cast<nvmlDeviceGetHandleByPciBusIdFn>(
      load_nvml_symbol(nvml_module, "nvmlDeviceGetHandleByPciBusId_v2"));
  const auto nvml_get_num_gpu_cores = reinterpret_cast<nvmlDeviceGetNumGpuCoresFn>(
      load_nvml_symbol(nvml_module, "nvmlDeviceGetNumGpuCores"));

  if (nvml_init == nullptr || nvml_shutdown == nullptr ||
      nvml_get_handle_by_pci_bus_id == nullptr || nvml_get_num_gpu_cores == nullptr) {
    close_nvml_library(nvml_module);
    return 0;
  }

  if (nvml_init() != kNvmlSuccess) {
    close_nvml_library(nvml_module);
    return 0;
  }

  nvmlDevice_t nvml_device = nullptr;
  if (nvml_get_handle_by_pci_bus_id(pci_bus_id, &nvml_device) != kNvmlSuccess) {
    nvml_shutdown();
    close_nvml_library(nvml_module);
    return 0;
  }

  unsigned int cuda_cores = 0;
  const nvmlReturn_t core_result = nvml_get_num_gpu_cores(nvml_device, &cuda_cores);
  nvml_shutdown();
  close_nvml_library(nvml_module);

  if (core_result != kNvmlSuccess) {
    return 0;
  }

  return cuda_cores;
}

inline void print_device_summary(int device = 0) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No CUDA device found.\n";
    std::exit(EXIT_FAILURE);
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int memory_clock_khz = 0;
  int memory_bus_width_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));

  const double memory_bandwidth_gb_s =
      2.0 * static_cast<double>(memory_clock_khz) *
      (static_cast<double>(memory_bus_width_bits) / 8.0) / 1.0e6;
  const unsigned int cuda_cores = try_get_cuda_core_count(device);

  const auto old_flags = std::cout.flags();
  const auto old_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2);

  std::cout << "GPU: " << prop.name << '\n'
            << "  global memory      : "
            << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << " GB\n"
            << "  SM count           : " << prop.multiProcessorCount << '\n';

  if (cuda_cores != 0) {
    std::cout << "  CUDA cores         : " << cuda_cores << '\n';
  }

  std::cout << "  memory bandwidth   : " << memory_bandwidth_gb_s << " GB/s\n"
            << "  max threads / block: " << prop.maxThreadsPerBlock << '\n';

  std::cout.flags(old_flags);
  std::cout.precision(old_precision);
}
