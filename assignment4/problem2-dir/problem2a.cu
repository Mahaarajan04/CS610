#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>

using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
#define pow 25
const uint64_t N = (1ULL << pow); 

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void block_scan_kernel(const uint32_t* input,
                                  uint32_t* output,
                                  uint32_t* block_sums,
                                  uint64_t n)
{
    extern __shared__ uint32_t sdata[];

    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;

    sdata[tid] = (gid < n ? input[gid] : 0);
    __syncthreads();

    for (uint32_t off = 1; off < blockDim.x; off <<= 1) {
        uint32_t val = (tid >= off ? sdata[tid - off] : 0);
        __syncthreads();
        sdata[tid] += val;
        __syncthreads();
    }

    if (gid < n) output[gid] = sdata[tid];
    if (tid == blockDim.x - 1) block_sums[blockIdx.x] = sdata[tid];
}

__global__ void add_offsets_kernel(uint32_t* output,
                                   const uint32_t* block_scanned,
                                   uint64_t n)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n || blockIdx.x == 0) return;

    output[gid] += block_scanned[blockIdx.x - 1];
}

void cpu_inclusive_scan(const uint32_t* in, uint32_t* out)
{
    out[0] = in[0];
    for (uint64_t i = 1; i < N; i++)
        out[i] = out[i - 1] + in[i];
}

void cpu_scan_block_sums(const uint32_t* in, uint32_t* out, int m)
{
    if (m <= 0) return;
    out[0] = in[0];
    for (int i = 1; i < m; i++)
        out[i] = out[i - 1] + in[i];
}

void check_result(const uint32_t* a, const uint32_t* b, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            cout << "Mismatch at index " << i << endl;
            assert(false);
        }
    }
    cout << "No differences found\n";
}

int main()
{
    uint32_t* h_input  = new uint32_t[N];
    uint32_t* h_ref    = new uint32_t[N];
    uint32_t* h_output = new uint32_t[N];

    for (uint64_t i = 0; i < N; i++) h_input[i] = 1;


    HRTimer cpu_t1 = HR::now();
    cpu_inclusive_scan(h_input, h_ref);
    HRTimer cpu_t2 = HR::now();
    double cpu_ms = duration_cast<milliseconds>(cpu_t2 - cpu_t1).count();
    uint32_t *d_input, *d_output;
    size_t bytes = N * sizeof(uint32_t);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    const int BLOCK = 1024;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    uint32_t *d_blk, *d_blk_scanned;
    cudaMalloc(&d_blk, GRID * sizeof(uint32_t));
    cudaMalloc(&d_blk_scanned, GRID * sizeof(uint32_t));

    uint32_t* h_blk = new uint32_t[GRID];
    uint32_t* h_blk_scanned = new uint32_t[GRID];

    cudaEvent_t k_start, k_end;
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_end);

    HRTimer t1 = HR::now();

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    block_scan_kernel<<<GRID, BLOCK, BLOCK * sizeof(uint32_t)>>>(
        d_input, d_output, d_blk, N);
    cudaCheckError(cudaGetLastError());

    cudaMemcpy(h_blk, d_blk, GRID * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    cpu_scan_block_sums(h_blk, h_blk_scanned, GRID);

    cudaMemcpy(d_blk_scanned, h_blk_scanned, GRID * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    add_offsets_kernel<<<GRID, BLOCK>>>(d_output, d_blk_scanned, N);
    cudaCheckError(cudaGetLastError());

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(k_start);

    block_scan_kernel<<<GRID, BLOCK, BLOCK * sizeof(uint32_t)>>>(
        d_input, d_output, d_blk, N);

    add_offsets_kernel<<<GRID, BLOCK>>>(d_output, d_blk_scanned, N);

    cudaEventRecord(k_end);
    cudaEventSynchronize(k_end);

    float kernel_only_ms = 0;
    cudaEventElapsedTime(&kernel_only_ms, k_start, k_end);
    HRTimer t2 = HR::now();
    double end_to_end_ms = duration_cast<milliseconds>(t2 - t1).count();

    check_result(h_ref, h_output, N);

    cout << "CPU time (ms):          " << cpu_ms << endl;
    cout << "Kernel-only time (ms):  " << kernel_only_ms << endl;
    cout << "End-to-end time (ms):   " << end_to_end_ms << endl;

    delete[] h_input;
    delete[] h_ref;
    delete[] h_output;
    delete[] h_blk;
    delete[] h_blk_scanned;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blk);
    cudaFree(d_blk_scanned);

    cudaEventDestroy(k_start);
    cudaEventDestroy(k_end);
    cout<<"Ran for 2^ "<<pow<<endl;
    return 0;
}
