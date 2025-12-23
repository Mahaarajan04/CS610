#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <algorithm>

#define MAX_POINTS 20000
#define MAX_LOCAL 128
#define STRIP 8


__device__ __forceinline__ bool check_constraints(
    const double* __restrict__ x,
    const double* __restrict__ c,
    const double* __restrict__ d,
    const double* __restrict__ e)
{
#pragma unroll
    for (int i = 0; i < 10; i++) {
        double sum =
              c[i*10 + 0] * x[0] + c[i*10 + 1] * x[1]
            + c[i*10 + 2] * x[2] + c[i*10 + 3] * x[3]
            + c[i*10 + 4] * x[4] + c[i*10 + 5] * x[5]
            + c[i*10 + 6] * x[6] + c[i*10 + 7] * x[7]
            + c[i*10 + 8] * x[8] + c[i*10 + 9] * x[9];

        if (fabs(sum - d[i]) > e[i])
            return false;
    }
    return true;
}



__global__ void grid_kernel_partitioned_stripmined(
    const double* __restrict__ grid,
    const double* __restrict__ disp,
    const int*    __restrict__ steps,
    double kk,
    double* results,
    long*   keys,       
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
    long stride = blockDim.x * gridDim.x;

    // strip-mined outer loop
    for (long idx = gid; idx < total_r2_to_r10; idx += stride * STRIP) {
#pragma unroll
        for (int k = 0; k < STRIP; k++) {

            long my_idx = idx + k * stride;
            if (my_idx >= total_r2_to_r10)
                break;

            long tmp = my_idx;
            double x[10];

            
            x[0] = grid[0] + fixed_r1 * grid[2];

            
#pragma unroll
            for (int dim = 9; dim >= 1; dim--) {
                int ri = tmp % steps[dim];
                x[dim] = grid[dim*3 + 0] + ri * grid[dim*3 + 2];
                tmp /= steps[dim];
            }

            if (check_constraints(x, s_c, s_d, s_e)) {

                int lpos = atomicAdd(&local_count, 1);

                if (lpos < MAX_LOCAL) {

                    // Store coordinates
#pragma unroll
                    for (int j = 0; j < 10; j++)
                        local_x[lpos][j] = x[j];

                    // Store pointer (CPU-order index)
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

            keys[gpos + i] = local_k[i];   

#pragma unroll
            for (int j = 0; j < 10; j++)
                results[(gpos + i) * 10 + j] = local_x[i][j];
        }
    }
}




int main()
{
    double h_grid[30], h_disp[120];

    FILE* fg = fopen("grid.txt", "r");
    for (int i = 0; i < 30; i++) fscanf(fg, "%lf", &h_grid[i]);
    fclose(fg);

    FILE* fd = fopen("disp.txt", "r");
    for (int i = 0; i < 120; i++) fscanf(fd, "%lf", &h_disp[i]);
    fclose(fd);

    // steps for r2..r10
    int s[10];
    long total_r2_to_r10 = 1;

    for (int i = 1; i < 10; i++) {
        s[i] = (int)floor((h_grid[i*3 + 1] - h_grid[i*3 + 0]) / h_grid[i*3 + 2]);
        total_r2_to_r10 *= s[i];
    }

    s[0] = (int)floor((h_grid[1] - h_grid[0]) / h_grid[2]);
    int r1_steps = s[0];


    // Allocate device data
    double *d_grid, *d_disp, *d_results;
    int *d_count, *d_steps;
    long *d_keys;

    cudaMalloc(&d_grid,    sizeof(double) * 30);
    cudaMalloc(&d_disp,    sizeof(double) * 120);
    cudaMalloc(&d_results, sizeof(double) * 10 * MAX_POINTS);
    cudaMalloc(&d_keys,    sizeof(long)   * MAX_POINTS);
    cudaMalloc(&d_count,   sizeof(int));
    cudaMalloc(&d_steps,   sizeof(int) * 10);

    cudaMemcpy(d_grid,  h_grid,  sizeof(double) * 30, cudaMemcpyHostToDevice);
    cudaMemcpy(d_disp,  h_disp,  sizeof(double) * 120, cudaMemcpyHostToDevice);
    cudaMemcpy(d_steps, s,       sizeof(int) * 10,      cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    double kk = 0.3;
    int blockSize = 256;
    int numBlocks = 1024;

    // Timing
    cudaEvent_t start_all, stop_all, start_kernel, stop_kernel;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_all);
    cudaEventRecord(start_kernel);

    // Launch one kernel per r1
    for (int r1 = 0; r1 < r1_steps; r1++) {
        grid_kernel_partitioned_stripmined<<<numBlocks, blockSize>>>(
            d_grid, d_disp, d_steps, kk, d_results, d_keys, d_count,
            r1, total_r2_to_r10);
    }

    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_count > MAX_POINTS) h_count = MAX_POINTS;

    printf("Total satisfying points: %d\n", h_count);

    double* h_results = (double*)malloc(sizeof(double) * 10 * h_count);
    long*   h_keys    = (long*)  malloc(sizeof(long)   * h_count);

    cudaMemcpy(h_results, d_results, sizeof(double) * 10 * h_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keys,    d_keys,    sizeof(long)   * h_count,      cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);

    float kernel_ms, total_ms;
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&total_ms,  start_all,    stop_all);

    printf("Kernel-only time: %.3f ms\n", kernel_ms);
    printf("End-to-end time : %.3f ms\n", total_ms);


    struct Row {
        long key;
        double x[10];
    };

    Row* rows = (Row*)malloc(sizeof(Row) * h_count);

    for (int i = 0; i < h_count; i++) {
        rows[i].key = h_keys[i];
        for (int j = 0; j < 10; j++)
            rows[i].x[j] = h_results[i*10 + j];
    }

    std::sort(rows, rows + h_count,
              [](const Row& a, const Row& b){
                  return a.key < b.key;
              });


    FILE* fptr = fopen("results-v2.txt", "w");

    for (int i = 0; i < h_count; i++) {
        for (int j = 0; j < 10; j++) {
            if (j < 9)
                fprintf(fptr, "%lf\t", rows[i].x[j]);
            else
                fprintf(fptr, "%lf\n", rows[i].x[j]);
        }
    }

    fclose(fptr);


    cudaFree(d_grid);
    cudaFree(d_disp);
    cudaFree(d_results);
    cudaFree(d_keys);
    cudaFree(d_steps);
    cudaFree(d_count);

    free(h_results);
    free(h_keys);
    free(rows);

    return 0;
}

