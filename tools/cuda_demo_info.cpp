#include "cpuinfo.hpp"

#include <iostream>

int main() {
  print_cpu_info();
  std::cout << "CUDA demos were skipped because no CUDA compiler/toolkit was detected during CMake configure.\n";
  std::cout << "Set CUDACXX or CMAKE_CUDA_COMPILER and reconfigure to build the GPU demos on this machine.\n";
  return 0;
}
