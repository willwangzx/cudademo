cmake_minimum_required(VERSION 3.24)

project(cudademo LANGUAGES CXX)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(CUDA_DEMO_REQUIRE_CUDA
  "Fail configuration instead of building the host-only fallback target when CUDA is unavailable."
  OFF)

add_library(cuda_demo_common INTERFACE)
target_include_directories(cuda_demo_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(cuda_demo_common INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /W4>
  )
else()
  target_compile_options(cuda_demo_common INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  )
endif()

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  if(MSVC)
    target_compile_options(cuda_demo_common INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
  else()
    target_compile_options(cuda_demo_common INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
  endif()

  function(add_cuda_demo target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE cuda_demo_common)
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endfunction()

  add_cuda_demo(demo00_hello demos/00_hello_cuda.cu)
  add_cuda_demo(demo01_vector_add demos/01_vector_add.cu)
  add_cuda_demo(demo02_matrix_add_2d demos/02_matrix_add_2d.cu)
  add_cuda_demo(demo03_unified_memory demos/03_unified_memory.cu)
  add_cuda_demo(demo04_reduction_shared demos/04_reduction_shared.cu)
  add_cuda_demo(demo05_tiled_matmul demos/05_tiled_matmul.cu)
  add_cuda_demo(demo06_streams_async demos/06_streams_async.cu)
  add_cuda_demo(demo07_bucket_sort demos/07_bucket_sort.cu)
  add_cuda_demo(demo08_fp32_gemm_compare demos/08_fp32_gemm_compare.cu)
  add_cuda_demo(demo09_tensorcore_tops_compare demos/09_tensorcore_tops_compare.cu)
  target_link_libraries(demo08_fp32_gemm_compare PRIVATE CUDA::cublas)
  target_link_libraries(demo09_tensorcore_tops_compare PRIVATE CUDA::cublas CUDA::cublasLt)

  message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")
else()
  if(CUDA_DEMO_REQUIRE_CUDA)
    message(FATAL_ERROR
      "CUDA compiler not found. Install the CUDA toolkit or set CUDACXX/CMAKE_CUDA_COMPILER, "
      "or configure with -DCUDA_DEMO_REQUIRE_CUDA=OFF to build the host-only fallback target.")
  endif()

  add_executable(cuda_demo_info tools/cuda_demo_info.cpp)
  target_link_libraries(cuda_demo_info PRIVATE cuda_demo_common)

  message(WARNING
    "CUDA compiler not found. Skipping CUDA demo executables and building 'cuda_demo_info' instead.")
endif()
