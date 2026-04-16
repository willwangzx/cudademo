#include "cuda_helpers.cuh"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kNumElements = 1 << 20;
constexpr int kValueRange = 4096;
constexpr int kBucketWidth = 16;
constexpr int kNumBuckets = kValueRange / kBucketWidth;
constexpr int kThreadsPerBlock = 256;

static_assert(kValueRange % kBucketWidth == 0, "bucket width must divide value range");

__global__ void bucket_histogram(const int* input, int* bucket_counts, int n) {
  __shared__ int local_counts[kNumBuckets];

  for (int bucket = threadIdx.x; bucket < kNumBuckets; bucket += blockDim.x) {
    local_counts[bucket] = 0;
  }
  __syncthreads();

  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = global_id; i < n; i += stride) {
    const int value = input[i];
    const int bucket = value / kBucketWidth;
    atomicAdd(&local_counts[bucket], 1);
  }
  __syncthreads();

  for (int bucket = threadIdx.x; bucket < kNumBuckets; bucket += blockDim.x) {
    atomicAdd(&bucket_counts[bucket], local_counts[bucket]);
  }
}

__global__ void exclusive_scan_small(const int* counts, int* offsets) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int running_total = 0;
  for (int bucket = 0; bucket < kNumBuckets; ++bucket) {
    offsets[bucket] = running_total;
    running_total += counts[bucket];
  }
}

__global__ void scatter_to_buckets(const int* input,
                                   int* bucket_write_positions,
                                   int* bucketed_values,
                                   int n) {
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = global_id; i < n; i += stride) {
    const int value = input[i];
    const int bucket = value / kBucketWidth;
    const int slot = atomicAdd(&bucket_write_positions[bucket], 1);
    bucketed_values[slot] = value;
  }
}

__global__ void sort_each_bucket(int* bucketed_values,
                                 const int* bucket_counts,
                                 const int* bucket_offsets) {
  const int bucket = blockIdx.x;
  if (bucket >= kNumBuckets) {
    return;
  }

  const int start = bucket_offsets[bucket];
  const int count = bucket_counts[bucket];
  const int bucket_base_value = bucket * kBucketWidth;

  __shared__ int local_value_counts[kBucketWidth];
  __shared__ int local_value_offsets[kBucketWidth];

  for (int local_value = threadIdx.x; local_value < kBucketWidth; local_value += blockDim.x) {
    local_value_counts[local_value] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const int value = bucketed_values[start + i];
    const int local_value = value - bucket_base_value;
    atomicAdd(&local_value_counts[local_value], 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int running_total = 0;
    for (int local_value = 0; local_value < kBucketWidth; ++local_value) {
      local_value_offsets[local_value] = running_total;
      running_total += local_value_counts[local_value];
    }
  }
  __syncthreads();

  for (int local_value = threadIdx.x; local_value < kBucketWidth; local_value += blockDim.x) {
    const int begin = local_value_offsets[local_value];
    const int end = begin + local_value_counts[local_value];
    const int value = bucket_base_value + local_value;

    for (int i = begin; i < end; ++i) {
      bucketed_values[start + i] = value;
    }
  }
}

}  // namespace

int main() {
  std::vector<int> h_input(kNumElements);
  std::vector<int> h_cpu_sorted(kNumElements);
  std::vector<int> h_gpu_sorted(kNumElements);
  std::vector<int> h_bucket_counts(kNumBuckets);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> distribution(0, kValueRange - 1);
  for (int& value : h_input) {
    value = distribution(rng);
  }

  h_cpu_sorted = h_input;

  const auto cpu_start = std::chrono::steady_clock::now();
  std::sort(h_cpu_sorted.begin(), h_cpu_sorted.end());
  const auto cpu_stop = std::chrono::steady_clock::now();
  const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

  int* d_input = nullptr;
  int* d_bucket_counts = nullptr;
  int* d_bucket_offsets = nullptr;
  int* d_bucket_write_positions = nullptr;
  int* d_bucketed_values = nullptr;

  const size_t input_bytes = static_cast<size_t>(kNumElements) * sizeof(int);
  const size_t bucket_bytes = static_cast<size_t>(kNumBuckets) * sizeof(int);

  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_bucket_counts, bucket_bytes));
  CUDA_CHECK(cudaMalloc(&d_bucket_offsets, bucket_bytes));
  CUDA_CHECK(cudaMalloc(&d_bucket_write_positions, bucket_bytes));
  CUDA_CHECK(cudaMalloc(&d_bucketed_values, input_bytes));

  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bucket_counts, 0, bucket_bytes));

  const int histogram_blocks = std::min(div_up(kNumElements, kThreadsPerBlock * 8), 128);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  bucket_histogram<<<histogram_blocks, kThreadsPerBlock>>>(d_input, d_bucket_counts, kNumElements);
  CUDA_CHECK(cudaGetLastError());

  exclusive_scan_small<<<1, 1>>>(d_bucket_counts, d_bucket_offsets);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(d_bucket_write_positions,
                        d_bucket_offsets,
                        bucket_bytes,
                        cudaMemcpyDeviceToDevice));

  scatter_to_buckets<<<histogram_blocks, kThreadsPerBlock>>>(
      d_input, d_bucket_write_positions, d_bucketed_values, kNumElements);
  CUDA_CHECK(cudaGetLastError());

  sort_each_bucket<<<kNumBuckets, kThreadsPerBlock>>>(
      d_bucketed_values, d_bucket_counts, d_bucket_offsets);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float gpu_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_gpu_sorted.data(), d_bucketed_values, input_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_bucket_counts.data(),
                        d_bucket_counts,
                        bucket_bytes,
                        cudaMemcpyDeviceToHost));

  const bool matches_cpu = (h_gpu_sorted == h_cpu_sorted);
  const bool nondecreasing =
      std::is_sorted(h_gpu_sorted.begin(), h_gpu_sorted.end());
  const int max_bucket_load =
      *std::max_element(h_bucket_counts.begin(), h_bucket_counts.end());

  print_device_summary();
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "CUDA bucket sort finished.\n";
  std::cout << "Elements         : " << kNumElements << '\n';
  std::cout << "Value range      : [0, " << (kValueRange - 1) << "]\n";
  std::cout << "Bucket width     : " << kBucketWidth << '\n';
  std::cout << "Bucket count     : " << kNumBuckets << '\n';
  std::cout << "Max bucket load  : " << max_bucket_load << '\n';
  std::cout << "CPU std::sort    : " << cpu_ms << " ms\n";
  std::cout << "GPU sort pipeline: " << gpu_kernel_ms << " ms\n";
  std::cout << "Sorted on GPU    : " << (nondecreasing ? "yes" : "no") << '\n';
  std::cout << "Matches CPU      : " << (matches_cpu ? "yes" : "no") << '\n';
  std::cout << "First 16 values  : ";
  for (int i = 0; i < 16; ++i) {
    std::cout << h_gpu_sorted[i] << (i + 1 == 16 ? '\n' : ' ');
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_bucket_counts));
  CUDA_CHECK(cudaFree(d_bucket_offsets));
  CUDA_CHECK(cudaFree(d_bucket_write_positions));
  CUDA_CHECK(cudaFree(d_bucketed_values));
  return (matches_cpu && nondecreasing) ? 0 : 1;
}

/*
This demo implements a simple CUDA bucket sort for bounded integers.

1. Split the value range into fixed-width buckets.
2. Count how many elements fall into each bucket on the GPU.
3. Prefix-sum the bucket counts to compute bucket offsets.
4. Scatter the input values into contiguous bucket ranges.
5. Sort each bucket in place with a tiny per-bucket counting sort.

The implementation is intentionally education-oriented:
- the bucket pass shows how to build a histogram and scatter in CUDA,
- the per-bucket sort keeps each bucket's local value range small,
- correctness is verified against CPU std::sort.
*/
