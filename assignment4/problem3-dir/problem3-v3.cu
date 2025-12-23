#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define MAX_POINTS 20000
#define MAX_LOCAL 128
#define STRIP 16


__device__ __forceinline__ bool check_constraints(
    const double* __restrict__ x,
    const double* __restrict__ c,
    const double* __restrict__ d,
    const double* __restrict__ e)
{
    for (int i = 0; i < 10; i++) {
        const double* ci = &c[i * 10];
        double sum = 0.0;

        #pragma unroll
        for (int j = 0; j < 10; j++)
            sum += ci[j] * x[j];

        if (fabs(sum - d[i]) > e[i])
            return false;
    }
    return true;
}


// Launch bounds to force lower registers
__global__ __launch_bounds__(256, 2)
void grid_kernel_partitioned_uvm(
    const double* __restrict__ grid,
    const double* __restrict__ disp,
    const int*    __restrict__ steps,
    double kk,
    double* results,
    long*  keys,          // added: global key array
    int* count,
    int fixed_r1,
    long total_r2_to_r10)
{
    __shared__ double s_c[100];
    __shared__ double s_d[10];
    __shared__ double s_e[10];
    __shared__ double local_x[MAX_LOCAL][10];
    __shared__ long   local_k[MAX_LOCAL];  
    __shared__ int local_count;

    int tid = threadIdx.x;

    if (tid < 100)
        s_c[tid] = disp[(tid / 10) * 12 + (tid % 10)];
    else if (tid < 110)
        s_d[tid - 100] = disp[(tid - 100) * 12 + 10];
    else if (tid < 120)
        s_e[tid - 110] = kk * disp[(tid - 110) * 12 + 11];

    if (tid == 0)
        local_count = 0;

    __syncthreads();

    long gid = blockIdx.x * blockDim.x + tid;
    long stride = (long)blockDim.x * (long)gridDim.x;

    double x[10];     // reused across STRIP

    for (long idx = gid; idx < total_r2_to_r10; idx += stride * STRIP) {
        for (int s = 0; s < STRIP; s++) {

            long my_idx = idx + (long)s * stride;
            if (my_idx >= total_r2_to_r10) break;

            long tmp = my_idx;

            x[0] = grid[0] + fixed_r1 * grid[2];

            for (int i = 9; i >= 1; i--) {
                int ri = (int)(tmp % steps[i]);
                x[i] = grid[i * 3 + 0] + (double)ri * grid[i * 3 + 2];
                tmp /= steps[i];
            }

            if (check_constraints(x, s_c, s_d, s_e)) {
                int lpos = atomicAdd(&local_count, 1);

                if (lpos < MAX_LOCAL) {
                    #pragma unroll
                    for (int j = 0; j < 10; j++)
                        local_x[lpos][j] = x[j];

                    // CPU-order index for this point:
                    // full 10D index = r1 * (s2..s10) + my_idx
                    long cpu_idx = (long)fixed_r1 * total_r2_to_r10 + my_idx;
                    local_k[lpos] = cpu_idx;
                }
            }
        }
    }

    __syncthreads();

    if (tid == 0 && local_count > 0) {
        int gpos = atomicAdd(count, local_count);
        int to_write = min(local_count, MAX_POINTS - gpos);

        for (int i = 0; i < to_write; i++) {
            // write global key
            keys[gpos + i] = local_k[i];

            #pragma unroll
            for (int j = 0; j < 10; j++)
                results[(gpos + i) * 10 + j] = local_x[i][j];
        }
    }
}


int main()
{
    int device = 0;
    cudaSetDevice(device);

    double *h_grid, *h_disp, *h_results;
    int *h_count, *h_steps;
    long *h_keys;   // added: keys in UVM

    cudaMallocManaged(&h_grid,    sizeof(double) * 30);
    cudaMallocManaged(&h_disp,    sizeof(double) * 120);
    cudaMallocManaged(&h_results, sizeof(double) * 10 * MAX_POINTS);
    cudaMallocManaged(&h_count,   sizeof(int));
    cudaMallocManaged(&h_steps,   sizeof(int) * 10);
    cudaMallocManaged(&h_keys,    sizeof(long) * MAX_POINTS);

    FILE* fgrid = fopen("grid.txt", "r");
    for (int i = 0; i < 30; i++)
        fscanf(fgrid, "%lf", &h_grid[i]);
    fclose(fgrid);

    FILE* fdisp = fopen("disp.txt", "r");
    for (int i = 0; i < 120; i++)
        fscanf(fdisp, "%lf", &h_disp[i]);
    fclose(fdisp);

    long total_r2_to_r10 = 1;
    for (int i = 1; i < 10; i++) {
        h_steps[i] =
            (int)floor((h_grid[i * 3 + 1] - h_grid[i * 3 + 0]) /
                        h_grid[i * 3 + 2]);
        total_r2_to_r10 *= h_steps[i];
    }

    h_steps[0] =
        (int)floor((h_grid[1] - h_grid[0]) / h_grid[2]);

    int r1_steps = h_steps[0];
    *h_count = 0;

    int dev = device;

    cudaMemAdvise(h_grid,    sizeof(double)*30,               cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(h_disp,    sizeof(double)*120,              cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(h_steps,   sizeof(int)*10,                  cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(h_results, sizeof(double)*10*MAX_POINTS,    cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(h_count,   sizeof(int),                     cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(h_keys,    sizeof(long)*MAX_POINTS,         cudaMemAdviseSetPreferredLocation, dev);

    cudaMemPrefetchAsync(h_grid,    sizeof(double)*30,            dev);
    cudaMemPrefetchAsync(h_disp,    sizeof(double)*120,           dev);
    cudaMemPrefetchAsync(h_steps,   sizeof(int)*10,               dev);
    cudaMemPrefetchAsync(h_results, sizeof(double)*10*MAX_POINTS, dev);
    cudaMemPrefetchAsync(h_count,   sizeof(int),                  dev);
    cudaMemPrefetchAsync(h_keys,    sizeof(long)*MAX_POINTS,      dev);

    cudaDeviceSynchronize();

    cudaEvent_t start_all, stop_all, start_kernel, stop_kernel;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_all);

    int blockSize = 256;
    int numBlocks = 1024;
    double kk = 0.3;

    cudaEventRecord(start_kernel);
    for (int r1 = 0; r1 < r1_steps; r1++) {
        grid_kernel_partitioned_uvm<<<numBlocks, blockSize>>>(
            h_grid, h_disp, h_steps, kk,
            h_results, h_keys, h_count,
            r1, total_r2_to_r10);
    }
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // bring data back to CPU for sorting/printing
    cudaMemPrefetchAsync(h_results, sizeof(double)*10*MAX_POINTS, cudaCpuDeviceId);
    cudaMemPrefetchAsync(h_count,   sizeof(int),                  cudaCpuDeviceId);
    cudaMemPrefetchAsync(h_keys,    sizeof(long)*MAX_POINTS,      cudaCpuDeviceId);
    cudaDeviceSynchronize();

    int count = *h_count;
    if (count > MAX_POINTS) count = MAX_POINTS;

    printf("Total satisfying points: %d\n", count);

    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);

    float kernel_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&total_ms,  start_all,    stop_all);

    printf("Kernel-only time: %.3f ms\n", kernel_ms);
    printf("End-to-end time : %.3f ms\n", total_ms);

    struct Row {
        long key;
        double x[10];
    };

    std::vector<Row> rows(count);
    for (int i = 0; i < count; i++) {
        rows[i].key = h_keys[i];
        for (int j = 0; j < 10; j++)
            rows[i].x[j] = h_results[i * 10 + j];
    }

    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) {
                  return a.key < b.key;
              });


    FILE* fptr = fopen("results-v3.txt", "w");
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 10; j++) {
            if (j < 9)
                fprintf(fptr, "%lf\t", rows[i].x[j]);
            else
                fprintf(fptr, "%lf\n", rows[i].x[j]);
        }
    }
    fclose(fptr);

    cudaFree(h_grid);
    cudaFree(h_disp);
    cudaFree(h_results);
    cudaFree(h_count);
    cudaFree(h_steps);
    cudaFree(h_keys);

    return 0;
}

