#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include<algorithm>

#define MAX_POINTS 20000

__device__ bool check_constraints(
    const double* __restrict__ x,
    const double* __restrict__ c,
    const double* __restrict__ d,
    const double* __restrict__ e)
{
    for (int i = 0; i < 10; i++) {
        double sum = 0.0;
        for (int j = 0; j < 10; j++)
            sum += c[i * 10 + j] * x[j];
        if (fabs(sum - d[i]) > e[i])
            return false;
    }
    return true;
}

__global__ void grid_kernel(
    const double* __restrict__ grid,
    const double* __restrict__ disp,
    double kk,
    double* results,
    long* keys,         
    int* count,
    long total)
{
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = gridDim.x * blockDim.x;

    int s[10];
    for (int i = 0; i < 10; i++) {
        s[i] = (int)floor((grid[i * 3 + 1] - grid[i * 3 + 0]) / grid[i * 3 + 2]);
    }

    double c[100], d[10], e[10];
    for (int i = 0; i < 10; i++) {
        int base = i * 12;
        for (int j = 0; j < 10; j++)
            c[i * 10 + j] = disp[base + j];
        d[i] = disp[base + 10];
        e[i] = kk * disp[base + 11];
    }

    for (long idx = tid; idx < total; idx += stride) {
        int r[10];
        long tmp = idx;
        for (int i = 9; i >= 0; i--) {
            r[i] = tmp % s[i];
            tmp /= s[i];
        }

        double x[10];
        for (int i = 0; i < 10; i++) {
            x[i] = grid[i * 3 + 0] + r[i] * grid[i * 3 + 2];
        }

        if (check_constraints(x, c, d, e)) {
            int pos = atomicAdd(count, 1);
            if (pos < MAX_POINTS) {

                keys[pos] = idx; //For ordering

                for (int j = 0; j < 10; j++)
                    results[pos * 10 + j] = x[j];
            }
        }
    }
}

int main()
{
    double h_grid[30], h_disp[120];
    FILE* fgrid = fopen("grid.txt", "r");
    for (int i = 0; i < 30; i++) fscanf(fgrid, "%lf", &h_grid[i]);
    fclose(fgrid);

    FILE* fdisp = fopen("disp.txt", "r");
    for (int i = 0; i < 120; i++) fscanf(fdisp, "%lf", &h_disp[i]);
    fclose(fdisp);

    int s[10];
    long total = 1;
    for (int i = 0; i < 10; i++) {
        s[i] = (int)floor((h_grid[i * 3 + 1] - h_grid[i * 3 + 0]) / h_grid[i * 3 + 2]);
        total *= s[i];
    }

    double *d_grid, *d_disp, *d_results;
    int* d_count;
    long *d_keys;               

    cudaMalloc(&d_grid, sizeof(double) * 30);
    cudaMalloc(&d_disp, sizeof(double) * 120);
    cudaMalloc(&d_results, sizeof(double) * 10 * MAX_POINTS);
    cudaMalloc(&d_keys, sizeof(long) * MAX_POINTS);      
    cudaMalloc(&d_count, sizeof(int));

    // Timers
    cudaEvent_t start_all, stop_all, start_kernel, stop_kernel;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_all);

    cudaMemcpy(d_grid, h_grid, sizeof(double) * 30, cudaMemcpyHostToDevice);
    cudaMemcpy(d_disp, h_disp, sizeof(double) * 120, cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    int blockSize = 256;
    int numBlocks = 1024;
    double kk = 0.3;

    cudaEventRecord(start_kernel);
    grid_kernel<<<numBlocks, blockSize>>>(d_grid, d_disp, kk, d_results, d_keys, d_count, total);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    double* h_results = (double*)malloc(sizeof(double) * 10 * h_count);
    long*   h_keys    = (long*)malloc(sizeof(long) * h_count);     

    cudaMemcpy(h_results, d_results, sizeof(double) * 10 * h_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keys,    d_keys,    sizeof(long)   * h_count,      cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);

    float kernel_ms = 0, total_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&total_ms, start_all, stop_all);

    printf("GPU result count: %d\n", h_count);
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
            rows[i].x[j] = h_results[i * 10 + j];
    }

    std::sort(rows, rows + h_count,
              [](const Row& a, const Row& b) {
                  return a.key < b.key;   
              });

    FILE* fptr = fopen("results-v1.txt", "w");
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
    cudaFree(d_count);
    free(h_results);
    free(h_keys);          
    free(rows);              

    return 0;
}
