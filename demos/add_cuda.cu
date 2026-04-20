#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

__global__ void add(int n, const float* x, float* y) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int dif=blockDim.x*gridDim.x;
    for (; i < n; i+=dif) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    int N = 1 << 20;
    
    float* d_x;
    float* d_y;

    cudaMallocManaged(&d_x,N*sizeof(float));
    cudaMallocManaged(&d_y,N*sizeof(float));

    for (int i = 0; i < N; i++) {
        d_x[i] = 1.0f;
        d_y[i] = 2.0f;
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaMemLocation prefetch_location{};
    prefetch_location.type = cudaMemLocationTypeDevice;
    prefetch_location.id = device;

    cudaMemPrefetchAsync(d_x, N * sizeof(float), prefetch_location, 0, 0);
    cudaMemPrefetchAsync(d_y, N * sizeof(float), prefetch_location, 0, 0);

    int blocksize=1024;
    int numblocks=(N+blocksize-1)/blocksize;

    add<<<numblocks, blocksize>>>(N, d_x, d_y);
    cudaDeviceSynchronize();

    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = max(max_error, abs(d_y[i] - 3.0f));
    }

    cout << "Max error: " << max_error << endl;

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
