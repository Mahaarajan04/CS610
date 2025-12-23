#include <iostream>
#include <cuda_runtime.h>

#define N 128  // You can change this
#define BLOCK_SIZE 8  // You can change this

__global__ void stencil3d_naive(float *input, float *output, int n) {
    // Compute the 3D thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Skip boundary cells to avoid out-of-bounds access
    if (i >= 1 && i < n - 1 &&
        j >= 1 && j < n - 1 &&
        k >= 1 && k < n - 1) {
        
        int idx = i * n * n + j * n + k;

        float center = input[idx];
        float x_plus = input[(i + 1) * n * n + j * n + k];
        float x_minus = input[(i - 1) * n * n + j * n + k];
        float y_plus = input[i * n * n + (j + 1) * n + k];
        float y_minus = input[i * n * n + (j - 1) * n + k];
        float z_plus = input[i * n * n + j * n + (k + 1)];
        float z_minus = input[i * n * n + j * n + (k - 1)];

        output[idx] = (center + x_plus + x_minus + y_plus + y_minus + z_plus + z_minus) / 7.0f;
    }
}

// Host helper to initialize and run the kernel
int main() {
    int size = N * N * N;
    size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(bytes);
    float *h_output = (float *)malloc(bytes);

    // Initialize input
    for (int i = 0; i < size; i++) h_input[i] = static_cast<float>(i % 100);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    stencil3d_naive<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
