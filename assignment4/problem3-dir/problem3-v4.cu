#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>

// Correct Thrust headers (your cluster supports these)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#define MAX_POINTS 20000
#define MAX_LOCAL  128
#define STRIP      16


struct Row {
    long key;
    double x[10];
};

// Comparator for thrust::sort
struct RowKeyLess {
    __host__ __device__
    bool operator()(const Row& a, const Row& b) const {
        return a.key < b.key;
    }
};

// functor for thrust::transform() to create validity mask
struct IsValidRow {
    __host__ __device__
    char operator()(const Row& r) const {
        return (r.key != -1);
    }
};


__device__ __forceinline__
bool check_constraints(
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


// Kernel (r1-sliced multi-launch)
__global__ __launch_bounds__(256, 2)
void grid_kernel_partitioned_uvm(
    const double* __restrict__ grid,
    const double* __restrict__ disp,
    const int*    __restrict__ steps,
    double kk,
    Row* rows,
    int* count,
    int fixed_r1,
    long total_r2_to_r10)
{
    __shared__ double s_c[100];
    __shared__ double s_d[10];
    __shared__ double s_e[10];

    __shared__ double local_x[MAX_LOCAL][10];
    __shared__ long   local_key[MAX_LOCAL];
    __shared__ int    local_count;

    int tid = threadIdx.x;

    // Load disp into shared memory
    if (tid < 100)
        s_c[tid] = disp[(tid/10)*12 + (tid%10)];
    else if (tid < 110)
        s_d[tid - 100] = disp[(tid-100)*12 + 10];
    else if (tid < 120)
        s_e[tid - 110] = kk * disp[(tid-110)*12 + 11];

    if (tid == 0)
        local_count = 0;

    __syncthreads();

    long gid = blockIdx.x * (long)blockDim.x + tid;
    long stride = (long)blockDim.x * (long)gridDim.x;

    double x[10];

    for (long idx = gid; idx < total_r2_to_r10; idx += stride * STRIP) {
        for (int s = 0; s < STRIP; s++) {

            long my_idx = idx + (long)s * stride;
            if (my_idx >= total_r2_to_r10) break;

            long tmp = my_idx;

            // r1 coordinate (fixed)
            x[0] = grid[0] + fixed_r1 * grid[2];


            for (int i = 9; i >= 1; i--) {
                int ri = (int)(tmp % steps[i]);
                x[i] = grid[i*3 + 0] + ri * grid[i*3 + 2];
                tmp /= steps[i];
            }

            if (check_constraints(x, s_c, s_d, s_e)) {
                int pos = atomicAdd(&local_count, 1);

                if (pos < MAX_LOCAL) {
                    #pragma unroll
                    for (int j = 0; j < 10; j++)
                        local_x[pos][j] = x[j];

                    long cpu_idx = (long)fixed_r1 * total_r2_to_r10 + my_idx;
                    local_key[pos] = cpu_idx;
                }
            }
        }
    }

    __syncthreads();


    if (tid == 0 && local_count > 0) {
        int gpos = atomicAdd(count, local_count);
        int to_write = min(local_count, MAX_POINTS - gpos);

        for (int i = 0; i < to_write; i++) {
            rows[gpos + i].key = local_key[i];
            #pragma unroll
            for (int j = 0; j < 10; j++)
                rows[gpos + i].x[j] = local_x[i][j];
        }
    }
}


int main()
{
    int device = 0;
    cudaSetDevice(device);

    double *h_grid, *h_disp;
    int    *h_steps, *h_count;
    Row    *h_rows, *h_compact;

    cudaMallocManaged(&h_grid,    sizeof(double)*30);
    cudaMallocManaged(&h_disp,    sizeof(double)*120);
    cudaMallocManaged(&h_steps,   sizeof(int)*10);
    cudaMallocManaged(&h_count,   sizeof(int));
    cudaMallocManaged(&h_rows,    sizeof(Row)*MAX_POINTS);
    cudaMallocManaged(&h_compact, sizeof(Row)*MAX_POINTS);


    thrust::device_ptr<Row> d_rows(h_rows);
    thrust::fill(d_rows, d_rows + MAX_POINTS, Row{ -1, {0} });

    // Read grid + disp
    FILE* fgrid = fopen("grid.txt", "r");
    for (int i = 0; i < 30; i++) fscanf(fgrid, "%lf", &h_grid[i]);
    fclose(fgrid);

    FILE* fdisp = fopen("disp.txt", "r");
    for (int i = 0; i < 120; i++) fscanf(fdisp, "%lf", &h_disp[i]);
    fclose(fdisp);

    long total_r2_to_r10 = 1;
    for (int i = 1; i < 10; i++) {
        h_steps[i] = (int)floor((h_grid[i*3+1] - h_grid[i*3+0]) / h_grid[i*3+2]);
        total_r2_to_r10 *= h_steps[i];
    }
    h_steps[0] = (int)floor((h_grid[1] - h_grid[0]) / h_grid[2]);
    int r1_steps = h_steps[0];

    *h_count = 0;


    cudaEvent_t t_start_all, t_stop_all;
    cudaEvent_t t_start_kernel, t_stop_kernel;
    cudaEvent_t t_start_transform, t_stop_transform;
    cudaEvent_t t_start_copyif, t_stop_copyif;
    cudaEvent_t t_start_sort, t_stop_sort;

    cudaEventCreate(&t_start_all);
    cudaEventCreate(&t_stop_all);
    cudaEventCreate(&t_start_kernel);
    cudaEventCreate(&t_stop_kernel);
    cudaEventCreate(&t_start_transform);
    cudaEventCreate(&t_stop_transform);
    cudaEventCreate(&t_start_copyif);
    cudaEventCreate(&t_stop_copyif);
    cudaEventCreate(&t_start_sort);
    cudaEventCreate(&t_stop_sort);

    cudaEventRecord(t_start_all);

    int blockSize = 256;
    int numBlocks = 1024;
    double kk = 0.3;


    cudaEventRecord(t_start_kernel);

    for (int r1 = 0; r1 < r1_steps; r1++) {
        grid_kernel_partitioned_uvm<<<numBlocks, blockSize>>>(
            h_grid, h_disp, h_steps, kk,
            h_rows, h_count,
            r1, total_r2_to_r10);
    }
    cudaEventRecord(t_stop_kernel);
    cudaEventSynchronize(t_stop_kernel);

    int count = *h_count;
    if (count > MAX_POINTS) count = MAX_POINTS;

    printf("Raw kernel output count = %d\n", count);


    thrust::device_vector<int> d_idx(count);
    thrust::sequence(d_idx.begin(), d_idx.end(), 0);


    thrust::device_vector<char> d_mask(count);

    cudaEventRecord(t_start_transform);
    thrust::transform(d_rows, d_rows + count, d_mask.begin(), IsValidRow());
    cudaEventRecord(t_stop_transform);
    cudaEventSynchronize(t_stop_transform);


    thrust::device_ptr<Row> d_compact(h_compact);

    cudaEventRecord(t_start_copyif);
    auto it_end = thrust::copy_if(
        d_rows, d_rows + count,
        d_mask.begin(),
        d_compact,
        thrust::identity<char>()
    );
    cudaEventRecord(t_stop_copyif);
    cudaEventSynchronize(t_stop_copyif);

    int compact_count = it_end - d_compact;

    printf("After copy_if compact count = %d\n", compact_count);


    cudaEventRecord(t_start_sort);
    thrust::sort(d_compact, d_compact + compact_count, RowKeyLess());
    cudaEventRecord(t_stop_sort);
    cudaEventSynchronize(t_stop_sort);

    cudaMemPrefetchAsync(h_compact, sizeof(Row)*compact_count, cudaCpuDeviceId);
    cudaDeviceSynchronize();


    FILE* fptr = fopen("results-v4.txt", "w");
    for (int i = 0; i < compact_count; i++) {
        for (int j = 0; j < 10; j++) {
            if (j < 9) fprintf(fptr, "%lf\t", h_compact[i].x[j]);
            else       fprintf(fptr, "%lf\n",  h_compact[i].x[j]);
        }
    }
    fclose(fptr);

    cudaEventRecord(t_stop_all);
    cudaEventSynchronize(t_stop_all);


    float ms_total, ms_kernel, ms_transform, ms_copyif, ms_sort;

    cudaEventElapsedTime(&ms_total,     t_start_all,      t_stop_all);
    cudaEventElapsedTime(&ms_kernel,    t_start_kernel,   t_stop_kernel);
    cudaEventElapsedTime(&ms_transform, t_start_transform,t_stop_transform);
    cudaEventElapsedTime(&ms_copyif,    t_start_copyif,   t_stop_copyif);
    cudaEventElapsedTime(&ms_sort,      t_start_sort,     t_stop_sort);

    printf("\n================ Timing Report ================\n");
    printf("Kernel compute time       : %.3f ms\n", ms_kernel);
    printf("Thrust transform time     : %.3f ms\n", ms_transform);
    printf("Thrust copy_if time       : %.3f ms\n", ms_copyif);
    printf("Thrust sort time          : %.3f ms\n", ms_sort);
    printf("End-to-end time           : %.3f ms\n", ms_total);
    printf("================================================\n\n");

    cudaFree(h_grid);
    cudaFree(h_disp);
    cudaFree(h_steps);
    cudaFree(h_count);
    cudaFree(h_rows);
    cudaFree(h_compact);

    return 0;
}

