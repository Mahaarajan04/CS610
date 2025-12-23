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

#define POW 25
const uint64_t N = (1ULL << POW);

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// -----------------------------------------------------------------------------
// Kernel 1: Per-block inclusive scan + store block sums (uint64_t)
// -----------------------------------------------------------------------------
__global__ void block_scan_kernel(const uint32_t* input,
                                  uint64_t* output,
                                  uint64_t* block_sums,
                                  uint64_t n)
{
    extern __shared__ uint64_t sdata[];

    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;

    sdata[tid] = (gid < n ? (uint64_t)input[gid] : 0ULL);
    __syncthreads();

    for (uint32_t off = 1; off < blockDim.x; off <<= 1) {
        uint64_t val = (tid >= off ? sdata[tid - off] : 0ULL);
        __syncthreads();
        sdata[tid] += val;
        __syncthreads();
    }

    if (gid < n)
        output[gid] = sdata[tid];

    if (tid == blockDim.x - 1)
        block_sums[blockIdx.x] = sdata[tid];
}


// -----------------------------------------------------------------------------
// Kernel 2: add offsets from scanned block_sums
// -----------------------------------------------------------------------------
__global__ void add_offsets_kernel(uint64_t* output,
                                   const uint64_t* block_scanned,
                                   uint64_t n)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n || blockIdx.x == 0) return;

    output[gid] += block_scanned[blockIdx.x - 1];
}


// -----------------------------------------------------------------------------
// CPU Reference Scan (uint64_t)
// -----------------------------------------------------------------------------
void cpu_inclusive_scan(const uint32_t* in, uint64_t* out)
{
    out[0] = in[0];
    for (uint64_t i = 1; i < N; i++)
        out[i] = out[i - 1] + in[i];
}

void cpu_scan_block_sums(const uint64_t* in, uint64_t* out, uint64_t m)
{
    if (m == 0) return;
    out[0] = in[0];
    for (uint64_t i = 1; i < m; i++)
        out[i] = out[i - 1] + in[i];
}


// -----------------------------------------------------------------------------
// Checker
// -----------------------------------------------------------------------------
void check_result(const uint64_t* a, const uint64_t* b, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            cout << "Mismatch at index " << i
                 << "   CPU=" << a[i]
                 << "   GPU=" << b[i] << endl;
            assert(false);
        }
    }
    cout << "No differences found\n";
}


// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main()
{
    uint32_t* h_input  = new uint32_t[N];
    uint64_t* h_ref    = new uint64_t[N];
    uint64_t* h_output = new uint64_t[N];

    for (uint64_t i = 0; i < N; i++)
        h_input[i] = 1;

    // CPU timing
    HRTimer cpu_t1 = HR::now();
    cpu_inclusive_scan(h_input, h_ref);
    HRTimer cpu_t2 = HR::now();
    double cpu_ms = duration_cast<milliseconds>(cpu_t2 - cpu_t1).count();

    // GPU buffers
    uint32_t *d_input;
    uint64_t *d_output;
    size_t bytes_in  = N * sizeof(uint32_t);
    size_t bytes_out = N * sizeof(uint64_t);

    cudaCheckError(cudaMallocManaged(&d_input, bytes_in));
    cudaCheckError(cudaMallocManaged(&d_output, bytes_out));

    const int BLOCK = 1024;
    const uint64_t GRID = (N + BLOCK - 1) / BLOCK;

    uint64_t *d_blk, *d_blk_scanned;
    cudaCheckError(cudaMallocManaged(&d_blk, GRID * sizeof(uint64_t)));
    cudaCheckError(cudaMallocManaged(&d_blk_scanned, GRID * sizeof(uint64_t)));

    uint64_t* h_blk         = d_blk;
    uint64_t* h_blk_scanned = d_blk_scanned;

    cudaEvent_t k_start, k_end;
    cudaCheckError(cudaEventCreate(&k_start));
    cudaCheckError(cudaEventCreate(&k_end));

    int device = 0;

    HRTimer t1 = HR::now();

    // Copy host → UVM
    for (uint64_t i = 0; i < N; i++)
        d_input[i] = h_input[i];

    // Advise + prefetch
    cudaCheckError(cudaMemAdvise(d_input, bytes_in,
                                 cudaMemAdviseSetPreferredLocation, device));
    cudaCheckError(cudaMemAdvise(d_output, bytes_out,
                                 cudaMemAdviseSetPreferredLocation, device));
    cudaCheckError(cudaMemPrefetchAsync(d_input, bytes_in, device));
    cudaCheckError(cudaMemPrefetchAsync(d_output, bytes_out, device));
    cudaCheckError(cudaMemPrefetchAsync(d_blk,
                                        GRID * sizeof(uint64_t), device));
    cudaCheckError(cudaMemPrefetchAsync(d_blk_scanned,
                                        GRID * sizeof(uint64_t), device));
    cudaCheckError(cudaDeviceSynchronize());

    // Kernel 1
    block_scan_kernel<<<GRID, BLOCK, BLOCK * sizeof(uint64_t)>>>(
        d_input, d_output, d_blk, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // CPU scan of block sums
    cpu_scan_block_sums(h_blk, h_blk_scanned, GRID);

    cudaCheckError(cudaMemPrefetchAsync(d_blk_scanned,
                                        GRID * sizeof(uint64_t), device));
    cudaCheckError(cudaDeviceSynchronize());

    // Kernel 2
    add_offsets_kernel<<<GRID, BLOCK>>>(d_output, d_blk_scanned, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Bring output back to CPU
    cudaCheckError(cudaMemPrefetchAsync(d_output, bytes_out, cudaCpuDeviceId));
    cudaCheckError(cudaDeviceSynchronize());

    // Copy UVM → plain host
    for (uint64_t i = 0; i < N; i++)
        h_output[i] = d_output[i];

    // KERNEL ONLY TIME
    cudaCheckError(cudaEventRecord(k_start));

    block_scan_kernel<<<GRID, BLOCK, BLOCK * sizeof(uint64_t)>>>(
        d_input, d_output, d_blk, N);
    cudaCheckError(cudaGetLastError());

    add_offsets_kernel<<<GRID, BLOCK>>>(d_output, d_blk_scanned, N);
    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaEventRecord(k_end));
    cudaCheckError(cudaEventSynchronize(k_end));

    float kernel_only_ms = 0;
    cudaCheckError(cudaEventElapsedTime(&kernel_only_ms, k_start, k_end));

    HRTimer t2 = HR::now();
    double end_to_end_ms = duration_cast<milliseconds>(t2 - t1).count();

    // Final correctness
    check_result(h_ref, h_output, N);

    cout << "CPU time (ms):          " << cpu_ms << endl;
    cout << "Kernel-only time (ms):  " << kernel_only_ms << endl;
    cout << "End-to-end time (ms):   " << end_to_end_ms << endl;
    cout << "Pow used is 2^ " << POW << endl;

    delete[] h_input;
    delete[] h_ref;
    delete[] h_output;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blk);
    cudaFree(d_blk_scanned);

    cudaEventDestroy(k_start);
    cudaEventDestroy(k_end);

    return 0;
}
