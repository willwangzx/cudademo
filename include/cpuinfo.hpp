#pragma once

#include <cctype>
#include <cstring>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
#include <cpuid.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <filesystem>
#include <fstream>
#include <set>
#endif

inline bool cpuid_supported() {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  return true;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
  return true;
#else
  return false;
#endif
}

inline bool read_cpuid(unsigned int leaf, unsigned int subleaf, int info[4]) {
  if (!cpuid_supported()) {
    return false;
  }

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
  return true;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
  const unsigned int group = leaf & 0x80000000u;
  if (__get_cpuid_max(group, nullptr) < leaf) {
    return false;
  }

  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);

  info[0] = static_cast<int>(eax);
  info[1] = static_cast<int>(ebx);
  info[2] = static_cast<int>(ecx);
  info[3] = static_cast<int>(edx);
  return true;
#else
  (void)leaf;
  (void)subleaf;
  (void)info;
  return false;
#endif
}

inline std::string trim_ascii_whitespace(std::string value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.erase(value.begin());
  }

  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.pop_back();
  }

  return value;
}

inline std::string getCpuVendor() {
  int info[4] = {};
  if (!read_cpuid(0, 0, info)) {
    return "Unknown vendor";
  }

  char vendor[13] = {};
  std::memcpy(vendor + 0, &info[1], 4);
  std::memcpy(vendor + 4, &info[3], 4);
  std::memcpy(vendor + 8, &info[2], 4);
  return vendor;
}

inline std::string getCpuBrand() {
  int info[4] = {};
  if (!read_cpuid(0x80000000u, 0, info) ||
      static_cast<unsigned int>(info[0]) < 0x80000004u) {
    return "Unknown CPU";
  }

  int brand_info[12] = {};
  if (!read_cpuid(0x80000002u, 0, brand_info + 0) ||
      !read_cpuid(0x80000003u, 0, brand_info + 4) ||
      !read_cpuid(0x80000004u, 0, brand_info + 8)) {
    return "Unknown CPU";
  }

  char brand[49] = {};
  std::memcpy(brand, brand_info, sizeof(brand_info));
  return trim_ascii_whitespace(brand);
}

#if defined(__linux__)
inline bool is_cpu_directory_name(const std::string& name) {
  if (name.size() <= 3 || name.rfind("cpu", 0) != 0) {
    return false;
  }

  for (size_t i = 3; i < name.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(name[i])) == 0) {
      return false;
    }
  }

  return true;
}

inline std::string read_first_line(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string line;
  if (!input.is_open() || !std::getline(input, line)) {
    return {};
  }

  return trim_ascii_whitespace(line);
}
#endif

inline unsigned int getPhysicalCoreCount() {
#if defined(_WIN32)
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
  if (length == 0) {
    return 0;
  }

  std::vector<unsigned char> buffer(length);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
          &length)) {
    return 0;
  }

  DWORD cores = 0;
  DWORD offset = 0;
  while (offset < length) {
    const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
        buffer.data() + offset);
    if (info->Relationship == RelationProcessorCore) {
      ++cores;
    }
    offset += info->Size;
  }
  return static_cast<unsigned int>(cores);
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
  int cores = 0;
  size_t size = sizeof(cores);
  if (sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) == 0 && cores > 0) {
    return static_cast<unsigned int>(cores);
  }
  return 0;
#elif defined(__linux__)
  namespace fs = std::filesystem;

  std::error_code error;
  const fs::path cpu_root("/sys/devices/system/cpu");
  std::set<std::string> unique_cores;

  if (fs::exists(cpu_root, error)) {
    for (const auto& entry : fs::directory_iterator(
             cpu_root, fs::directory_options::skip_permission_denied, error)) {
      if (error) {
        break;
      }

      if (!entry.is_directory(error) || error) {
        error.clear();
        continue;
      }

      const std::string directory_name = entry.path().filename().string();
      if (!is_cpu_directory_name(directory_name)) {
        continue;
      }

      const fs::path topology_dir = entry.path() / "topology";
      const std::string core_id = read_first_line(topology_dir / "core_id");
      if (core_id.empty()) {
        continue;
      }

      std::string package_id = read_first_line(topology_dir / "physical_package_id");
      if (package_id.empty()) {
        package_id = "0";
      }

      unique_cores.insert(package_id + ":" + core_id);
    }
  }

  if (!unique_cores.empty()) {
    return static_cast<unsigned int>(unique_cores.size());
  }
#endif

  return std::thread::hardware_concurrency();
}

inline void print_cpu_info() {
  std::cout << "CPU Name: " << getCpuBrand() << '\n';
  const unsigned int physical_cores = getPhysicalCoreCount();
  if (physical_cores != 0) {
    std::cout << "Physical Cores: " << physical_cores << '\n';
  }

  const unsigned int logical_threads = std::thread::hardware_concurrency();
  if (logical_threads != 0) {
    std::cout << "Logical Threads: " << logical_threads << '\n';
  }
}
