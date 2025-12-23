#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cout;
using std::endl;
using namespace std::chrono;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
// ------------------ CHECKER ------------------
void check_result(const double* a, const double* b, uint64_t n) {
    int diffs = 0;
    for (uint64_t i = 0; i < n * n * n; i++)
        if (fabs(a[i] - b[i]) > THRESHOLD)
            diffs++;
    if (diffs == 0)
        cout << "No differences found!\n";
    else
        cout << diffs << " differences found!\n";
}
const uint64_t N = 256;         // Grid dimension
const uint64_t NUM = N * N * N;
const size_t BYTES = NUM * sizeof(double);
#define BLOCK 8                // Change to 1, 2, 4, or 8 for experiments

// ------------------ NAIVE KERNEL ------------------
__global__ void naive_kernel(const double* input, double* output, uint64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
        output[i * n * n + j * n + k] = 0.8 * (
            input[(i - 1) * n * n + j * n + k] +
            input[(i + 1) * n * n + j * n + k] +
            input[i * n * n + (j - 1) * n + k] +
            input[i * n * n + (j + 1) * n + k] +
            input[i * n * n + j * n + (k - 1)] +
            input[i * n * n + j * n + (k + 1)]
        );
    }
}

// ------------------ SHARED MEMORY KERNEL ------------------
__global__ void shmem_kernel(const double* input, double* output, uint64_t n) {
    extern __shared__ double tile[];

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockDim.x, by = blockDim.y, bz = blockDim.z;

    int i = blockIdx.x * bx + tx;
    int j = blockIdx.y * by + ty;
    int k = blockIdx.z * bz + tz;

    int si = tx + 1;
    int sj = ty + 1;
    int sk = tz + 1;

    int pitch = bx + 2;
    int slice = pitch * (by + 2);
    int sidx = si + sj * pitch + sk * slice;

    if (i < n && j < n && k < n)
        tile[sidx] = input[i * n * n + j * n + k];

    if (tx == 0 && i > 0)
        tile[(si - 1) + sj * pitch + sk * slice] = input[(i - 1) * n * n + j * n + k];
    if (tx == bx - 1 && i < n - 1)
        tile[(si + 1) + sj * pitch + sk * slice] = input[(i + 1) * n * n + j * n + k];

    if (ty == 0 && j > 0)
        tile[si + (sj - 1) * pitch + sk * slice] = input[i * n * n + (j - 1) * n + k];
    if (ty == by - 1 && j < n - 1)
        tile[si + (sj + 1) * pitch + sk * slice] = input[i * n * n + (j + 1) * n + k];

    if (tz == 0 && k > 0)
        tile[si + sj * pitch + (sk - 1) * slice] = input[i * n * n + j * n + (k - 1)];
    if (tz == bz - 1 && k < n - 1)
        tile[si + sj * pitch + (sk + 1) * slice] = input[i * n * n + j * n + (k + 1)];

    __syncthreads();

    if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
        output[i * n * n + j * n + k] = 0.8 * (
            tile[(si - 1) + sj * pitch + sk * slice] +
            tile[(si + 1) + sj * pitch + sk * slice] +
            tile[si + (sj - 1) * pitch + sk * slice] +
            tile[si + (sj + 1) * pitch + sk * slice] +
            tile[si + sj * pitch + (sk - 1) * slice] +
            tile[si + sj * pitch + (sk + 1) * slice]
        );
    }
}
__global__ void opt_kernel(const double* input, double* output, uint64_t n) {

    extern __shared__ double tile[];

    // Permutation: threadIdx.x = k (fastest)
    int tx = threadIdx.x; // k
    int ty = threadIdx.y; // j
    int tz = threadIdx.z; // i

    int bx = blockDim.x, by = blockDim.y, bz = blockDim.z;

    int k = blockIdx.x * bx + tx;
    int j = blockIdx.y * by + ty;
    int i = blockIdx.z * bz + tz;

    // Shared memory coords
    int sk = tx + 1;
    int sj = ty + 1;
    int si = tz + 1;

    int pitch = bx + 2;
    int slice = pitch * (by + 2);
    int idx = sk + sj*pitch + si*slice;

    // Load center
    if (i < n && j < n && k < n)
        tile[idx] = input[i*n*n + j*n + k];

    // Halo loads
    if (tx == 0 && k > 0)
        tile[(sk - 1) + sj*pitch + si*slice] = input[i*n*n + j*n + (k - 1)];
    if (tx == bx - 1 && k < n - 1)
        tile[(sk + 1) + sj*pitch + si*slice] = input[i*n*n + j*n + (k + 1)];

    if (ty == 0 && j > 0)
        tile[sk + (sj - 1)*pitch + si*slice] = input[i*n*n + (j - 1)*n + k];
    if (ty == by - 1 && j < n - 1)
        tile[sk + (sj + 1)*pitch + si*slice] = input[i*n*n + (j + 1)*n + k];

    if (tz == 0 && i > 0)
        tile[sk + sj*pitch + (si - 1)*slice] = input[(i - 1)*n*n + j*n + k];
    if (tz == bz - 1 && i < n - 1)
        tile[sk + sj*pitch + (si + 1)*slice] = input[(i + 1)*n*n + j*n + k];

    __syncthreads();

    // Compute using shared memory (UNROLLED)
    if (i >= 1 && i < n - 1 &&
        j >= 1 && j < n - 1 &&
        k >= 1 && k < n - 1) {

        double v =
            tile[(sk - 1) + sj*pitch + si*slice] +
            tile[(sk + 1) + sj*pitch + si*slice] +
            tile[sk + (sj - 1)*pitch + si*slice] +
            tile[sk + (sj + 1)*pitch + si*slice] +
            tile[sk + sj*pitch + (si - 1)*slice] +
            tile[sk + sj*pitch + (si + 1)*slice];

        output[i*n*n + j*n + k] = 0.8 * v;
    }
}

void run_pinned_pipeline(const double* h_in, double* h_out, const double* h_ref, size_t size_bytes) {
    // Pinned host memory (already passed in)
    double *d_in, *d_out;
    cudaCheckError(cudaMalloc(&d_in, size_bytes));
    cudaCheckError(cudaMalloc(&d_out, size_bytes));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threads(BLOCK, BLOCK, BLOCK);
    dim3 blocks((N + BLOCK - 1)/BLOCK, (N + BLOCK - 1)/BLOCK, (N + BLOCK - 1)/BLOCK);
    size_t shmem = (BLOCK + 2)*(BLOCK + 2)*(BLOCK + 2)*sizeof(double);

    auto start_t = high_resolution_clock::now();

    // Async H2D
    cudaMemcpyAsync(d_in, h_in, size_bytes, cudaMemcpyHostToDevice, stream);

    // Kernel
    opt_kernel<<<blocks, threads, shmem, stream>>>(d_in, d_out, N);

    // Async D2H
    cudaMemcpyAsync(h_out, d_out, size_bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    auto end_t = high_resolution_clock::now();

    float time_ms = duration_cast<milliseconds>(end_t - start_t).count();
    cout << "\nV4 pinned overlapped time: " << time_ms << " ms\n";
    check_result(h_ref, h_out, N);

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_out);
}



// ------------------ CPU BASELINE ------------------
void stencil(const double* in, double* out) {
    for (uint64_t i = 1; i < N - 1; i++)
        for (uint64_t j = 1; j < N - 1; j++)
            for (uint64_t k = 1; k < N - 1; k++)
                out[i * N * N + j * N + k] = 0.8 * (
                    in[(i - 1) * N * N + j * N + k] +
                    in[(i + 1) * N * N + j * N + k] +
                    in[i * N * N + (j - 1) * N + k] +
                    in[i * N * N + (j + 1) * N + k] +
                    in[i * N * N + j * N + (k - 1)] +
                    in[i * N * N + j * N + (k + 1)]
                );
}


// ------------------ MAIN ------------------
int main() {
    const uint64_t NUM_ELEMS = N * N * N;
    const size_t SIZE_BYTES = NUM_ELEMS * sizeof(double);

    double* h_in = new double[NUM_ELEMS];
    double* h_cpu = new double[NUM_ELEMS];
    double* h_gpu = new double[NUM_ELEMS];

    for (uint64_t i = 0; i < NUM_ELEMS; i++)
        h_in[i] = rand();

    std::fill(h_cpu, h_cpu + NUM_ELEMS, 0);
    std::fill(h_gpu, h_gpu + NUM_ELEMS, 0);

    // CPU baseline
    auto cpu_start = high_resolution_clock::now();
    stencil(h_in, h_cpu);
    auto cpu_end = high_resolution_clock::now();
    cout << "CPU time: "
         << duration_cast<milliseconds>(cpu_end - cpu_start).count()
         << " ms\n";

    // GPU alloc
    double *d_in, *d_out;
    cudaCheckError(cudaMalloc(&d_in, SIZE_BYTES));
    cudaCheckError(cudaMalloc(&d_out, SIZE_BYTES));

    // Launch config
    dim3 threads(BLOCK, BLOCK, BLOCK);
    dim3 blocks((N + BLOCK - 1) / BLOCK,
                (N + BLOCK - 1) / BLOCK,
                (N + BLOCK - 1) / BLOCK);

    // Events for kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // ------------------ NAIVE ------------------
    auto naive_start = high_resolution_clock::now();
    cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, SIZE_BYTES);

    cudaEventRecord(start);
    naive_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost);
    auto naive_end = high_resolution_clock::now();

    float naive_ms;
    cudaEventElapsedTime(&naive_ms, start, stop);

    cout << "\nNaive kernel-only time: " << naive_ms << " ms\n";
    cout << "Naive end-to-end time : "
         << duration_cast<milliseconds>(naive_end - naive_start).count()
         << " ms\n";
    check_result(h_cpu, h_gpu, N);

    // ------------------ SHARED MEMORY ------------------
    auto shmem_start = high_resolution_clock::now();
    cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, SIZE_BYTES);

    size_t shmem_bytes = (BLOCK + 2)*(BLOCK + 2)*(BLOCK + 2) * sizeof(double);

    cudaEventRecord(start);
    shmem_kernel<<<blocks, threads, shmem_bytes>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost);
    auto shmem_end = high_resolution_clock::now();

    float shmem_ms;
    cudaEventElapsedTime(&shmem_ms, start, stop);

    cout << "\nShared kernel-only time: " << shmem_ms << " ms\n";
    cout << "Shared end-to-end time : "
         << duration_cast<milliseconds>(shmem_end - shmem_start).count()
         << " ms\n";
    check_result(h_cpu, h_gpu, N);
// ------------------ OPT MEMORY ------------------
    auto opt_start = high_resolution_clock::now();
    cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, SIZE_BYTES);

    size_t opt_bytes = (BLOCK + 2)*(BLOCK + 2)*(BLOCK + 2) * sizeof(double);

    cudaEventRecord(start);
    opt_kernel<<<blocks, threads, opt_bytes>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost);
    auto opt_end = high_resolution_clock::now();

    float opt_ms;
    cudaEventElapsedTime(&opt_ms, start, stop);

    cout << "\nOpt kernel-only time: " << opt_ms << " ms\n";
    cout << "Opt end-to-end time : "
         << duration_cast<milliseconds>(opt_end - opt_start).count()
         << " ms\n";
    check_result(h_cpu, h_gpu, N);


        // ----------------- V4: PINNED HOST MEMORY + OVERLAP ----------------
    double *h_in_pin, *h_out_pin;
    cudaCheckError(cudaHostAlloc((void**)&h_in_pin, BYTES, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&h_out_pin, BYTES, cudaHostAllocDefault));

    memcpy(h_in_pin, h_in, BYTES); // copy to pinned buffer

    run_pinned_pipeline(h_in_pin, h_out_pin, h_cpu, BYTES);

    cudaFreeHost(h_in_pin); cudaFreeHost(h_out_pin);

    // Cleanup
    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_cpu; delete[] h_gpu;
    cout<<"Block size is: "<<BLOCK<<endl;
    return 0;
}
