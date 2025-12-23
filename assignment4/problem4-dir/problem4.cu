#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <cmath>
#include<assert.h>

#define THRESHOLD 1e-3

using std::cerr;
using std::cout;
using std::endl;


#define R_2D 1
#define R_3D 1    
#define FILTER_W_2D (2*R_2D+1)
#define FILTER_W_3D (2*R_3D+1)

// Tunable block sizes for optimized kernels
#define BX 4                  // blockDim.x
#define BY 4                  // blockDim.y

// Shared-memory tile dimensions for optimized 2D
#define TILE_X (BX + 2*R_2D)      // BX + 2 = 18
#define TILE_Y (BY + 2*R_2D)      // BY + 2 = 18

// We define TILE_3D sizes for 3D
#define BZ_3D 4
#define BY_3D 4
#define BX_3D 4

#define TILE_3D_Z (BZ_3D + 2*R_3D)
#define TILE_3D_Y (BY_3D + 2*R_3D)
#define TILE_3D_X (BX_3D + 2*R_3D)

// Constant memory filter for optimized 2D kernel
__constant__ float d_filter2D[FILTER_W_2D * FILTER_W_2D];
__constant__ float d_filter3D[FILTER_W_3D * FILTER_W_3D * FILTER_W_3D];



#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

const int N3 = (1 << 8);
const int N2 = (1 << 10);

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    gettimeofday(&Tp, &Tzp);
    return Tp.tv_sec + Tp.tv_usec * 1e-6;
}


void cpu_conv2D(const float* inp,
                float* out,
                const float* filter)
{
    int N =N2;
    int R=R_2D;
    int filter_w= FILTER_W_2D;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            float sum = 0.0f;

            // iterate over filter window
            for (int fy = -R; fy <= R; fy++) {
                for (int fx = -R; fx <= R; fx++) {

                    int ni = i + fy;
                    int nj = j + fx;

                    // zero padding
                    float pixel = 0.0f;
                    if (ni >= 0 && ni < N && nj >= 0 && nj < N)
                        pixel = inp[ni * N + nj];

                    float w = filter[(fy + R) * filter_w + (fx + R)];
                    sum += pixel * w;
                }
            }

            out[i * N + j] = sum;
        }
    }
}



void cpu_conv3D(const float* inp,
                float* out,
                const float* filter)
{
    int N = N3;
    int R = R_3D;
    int filter_w= FILTER_W_3D;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {

                float sum = 0.0f;

                for (int dz = -R; dz <= R; dz++) {
                    for (int dy = -R; dy <= R; dy++) {
                        for (int dx = -R; dx <= R; dx++) {

                            int ni = i + dz;
                            int nj = j + dy;
                            int nk = k + dx;

                            float voxel = 0.0f;
                            if (ni >= 0 && ni < N &&
                                nj >= 0 && nj < N &&
                                nk >= 0 && nk < N)
                            {
                                voxel = inp[ni*N*N + nj*N + nk];
                            }

                            int f_index =
                                (dz + R) * filter_w * filter_w +
                                (dy + R) * filter_w +
                                (dx + R);

                            sum += voxel * filter[f_index];
                        }
                    }
                }

                out[i*N*N + j*N + k] = sum;
            }
        }
    }
}


__global__ void kernel2D_basic(const float*  inp,
                               float*  out,
                               const float* filter2D,
                               int N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N)
        return;

    float sum = 0.0f;

    // Convolution loops (3x3)
    for (int fy = -R_2D; fy <= R_2D; fy++) {
        for (int fx = -R_2D; fx <= R_2D; fx++) {

            int ni = i + fy;
            int nj = j + fx;

            float pixel = 0.0f;
            if (ni >= 0 && ni < N && nj >= 0 && nj < N)
                pixel = inp[ni * N + nj];

            float w = filter2D[(fy + R_2D) * FILTER_W_2D + (fx + R_2D)];
            sum += pixel * w;
        }
    }

    out[i * N + j] = sum;
}



__global__ void kernel3D_basic(const float* inp,
                               float* out,
                               const float*filter3D,
                               int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N)
        return;

    float sum = 0.0f;

    for (int fz = -R_3D; fz <= R_3D; fz++) {
        for (int fy = -R_3D; fy <= R_3D; fy++) {
            for (int fx = -R_3D; fx <= R_3D; fx++) {

                int ni = i + fz;
                int nj = j + fy;
                int nk = k + fx;

                float voxel = 0.0f;

                if (ni >= 0 && ni < N &&
                    nj >= 0 && nj < N &&
                    nk >= 0 && nk < N)
                {
                    voxel = inp[ni*N*N + nj*N + nk];
                }

                int f_idx =
                    (fz + R_3D) * FILTER_W_3D * FILTER_W_3D +
                    (fy + R_3D) * FILTER_W_3D +
                    (fx + R_3D);

                sum += voxel * filter3D[f_idx];
            }
        }
    }

    out[i*N*N + j*N + k] = sum;
}



__global__ void kernel2D_opt(const float* __restrict__ inp,
                             float* __restrict__ out,
                             int N)
{
    __shared__ float tile[TILE_Y][TILE_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_i = blockIdx.y * BY;
    int base_j = blockIdx.x * BX;

    // Load shared tile with zero padding
    for (int ti = ty; ti < TILE_Y; ti += BY) {
        int gi = base_i + ti - R_2D;

        for (int tj = tx; tj < TILE_X; tj += BX) {
            int gj = base_j + tj - R_2D;

            if (gi < 0 || gi >= N || gj < 0 || gj >= N)
                tile[ti][tj] = 0.0f;
            else
                tile[ti][tj] = inp[gi * N + gj];
        }
    }

    __syncthreads();

    // compute output pixel
    int i = base_i + ty;
    int j = base_j + tx;

    if (i >= N || j >= N)
        return;

    float sum = 0.0f;

    #pragma unroll
    for (int fy = -R_2D; fy <= R_2D; fy++) {
        int ti = ty + fy + R_2D;

        #pragma unroll
        for (int fx = -R_2D; fx <= R_2D; fx++) {
            int tj = tx + fx + R_2D;
            float pixel = tile[ti][tj];
            float w = d_filter2D[(fy + R_2D)*FILTER_W_2D + (fx + R_2D)];

            sum += pixel * w;
        }
    }

    out[i*N + j] = sum;
}


__global__ void kernel3D_opt(const float* __restrict__ inp,
                             float* __restrict__ out,
                             int N)
{
    __shared__ float TILE_3D[TILE_3D_Z][TILE_3D_Y][TILE_3D_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int base_k = blockIdx.x * BX_3D;
    int base_j = blockIdx.y * BY_3D;
    int base_i = blockIdx.z * BZ_3D;


    for (int tz2 = tz; tz2 < TILE_3D_Z; tz2 += blockDim.z) {
        int gi = base_i + tz2 - R_3D;

        for (int ty2 = ty; ty2 < TILE_3D_Y; ty2 += blockDim.y) {
            int gj = base_j + ty2 - R_3D;

            for (int tx2 = tx; tx2 < TILE_3D_X; tx2 += blockDim.x) {
                int gk = base_k + tx2 - R_3D;

                if (gi < 0 || gi >= N ||
                    gj < 0 || gj >= N ||
                    gk < 0 || gk >= N)
                {
                    TILE_3D[tz2][ty2][tx2] = 0.0f;
                }
                else {
                    TILE_3D[tz2][ty2][tx2] = inp[(gi*N + gj)*N + gk];
                }
            }
        }
    }

    __syncthreads();


    int i = base_i + tz;
    int j = base_j + ty;
    int k = base_k + tx;

    if (i >= N || j >= N || k >= N)
        return;

    float sum = 0.0f;

#pragma unroll
    for (int fz = -R_3D; fz <= R_3D; fz++) {
        int tz2 = tz + fz + R_3D;

#pragma unroll
        for (int fy = -R_3D; fy <= R_3D; fy++) {
            int ty2 = ty + fy + R_3D;

#pragma unroll
            for (int fx = -R_3D; fx <= R_3D; fx++) {
                int tx2 = tx + fx + R_3D;

                float voxel = TILE_3D[tz2][ty2][tx2];

                int f_idx =
                    (fz+R_3D)*FILTER_W_3D*FILTER_W_3D +
                    (fy+R_3D)*FILTER_W_3D +
                    (fx+R_3D);

                float w = d_filter3D[f_idx];

                sum += voxel * w;
            }
        }
    }

    out[(i*N + j)*N + k] = sum;
}




// ------------------------------------------------------------
void check2D(const float* ref, const float* test) {
    int numdiffs = 0;
    double maxdiff = 0.0;
    int N=N2;
    for (int i = 0; i < N*N; i++) {
        double diff = fabs(ref[i] - test[i]);
        if (diff > THRESHOLD) {
            numdiffs++;
            if (diff > maxdiff) maxdiff = diff;
        }
    }

    if (numdiffs == 0)
        cout << "No differences found\n";
    else
        cout << numdiffs << " diffs. Max diff = " << maxdiff << endl;
}



void check3D(const float* ref, const float* test) {
    int numdiffs = 0;
    double maxdiff = 0.0;

    int N=N3;
    for (int i = 0; i < N*N*N; i++) {
        double diff = fabs(ref[i] - test[i]);
        if (diff > THRESHOLD) {
            numdiffs++;
            if (diff > maxdiff) maxdiff = diff;
        }
    }

    if (numdiffs == 0)
        cout << "No differences found\n";
    else
        cout << numdiffs << " diffs. Max diff = " << maxdiff << endl;
}

// ------------------------------------------------------------
int main() {
    assert(N2>FILTER_W_2D && N3>FILTER_W_3D);
    size_t bytes2D = N2 * N2 * sizeof(float);
    size_t bytes3D = N3 * N3 * N3 * sizeof(float);

    float *h_in2D = (float*)malloc(bytes2D);
    float *h_ref2D = (float*)malloc(bytes2D);
    float *h_out2D = (float*)malloc(bytes2D);

    float *h_in3D = (float*)malloc(bytes3D);
    float *h_ref3D = (float*)malloc(bytes3D);
    float *h_out3D = (float*)malloc(bytes3D);

    for (int i = 0; i < N2*N2; i++)
        h_in2D[i] = (float)(rand() % 100);

    for (int i = 0; i < N3*N3*N3; i++)
        h_in3D[i] = (float)(rand() % 100);


    float h_filter2D[FILTER_W_2D * FILTER_W_2D];
    float h_filter3D[FILTER_W_3D * FILTER_W_3D * FILTER_W_3D];

    srand(0);  // fixed seed for reproducibility (optional)

    for (int i = 0; i < FILTER_W_2D * FILTER_W_2D; i++) {
        // random value in [-1, 1]
        float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_filter2D[i] = r * 0.1f;   // scale for stability
    }

    for (int i = 0; i < FILTER_W_3D * FILTER_W_3D * FILTER_W_3D; i++) {
        float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_filter3D[i] = r * 0.1f;
    }
    cudaMemcpyToSymbol(d_filter2D, h_filter2D, FILTER_W_2D * FILTER_W_2D * sizeof(float));
    cudaMemcpyToSymbol(d_filter3D, h_filter3D, FILTER_W_3D * FILTER_W_3D * FILTER_W_3D * sizeof(float));
    // copy 2D filter to constant memory for optimized 2D kernel
    cudaCheckError(cudaMemcpyToSymbol(d_filter2D,
                                      h_filter2D,
                                      FILTER_W_2D * FILTER_W_2D * sizeof(float)));

    // copy filters to global memory for basic + 3D optimized kernels
    float *d_filter2D, *d_filter3D;
    cudaCheckError(cudaMalloc(&d_filter2D, FILTER_W_2D * FILTER_W_2D * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_filter3D, FILTER_W_3D * FILTER_W_3D * FILTER_W_3D * sizeof(float)));

    cudaCheckError(cudaMemcpy(d_filter2D, h_filter2D,
                              FILTER_W_2D * FILTER_W_2D * sizeof(float),
                              cudaMemcpyHostToDevice));

    cudaCheckError(cudaMemcpy(d_filter3D, h_filter3D,
                              FILTER_W_3D * FILTER_W_3D * FILTER_W_3D * sizeof(float),
                              cudaMemcpyHostToDevice));



    double t0 = rtclock();
    cpu_conv2D(h_in2D, h_ref2D, h_filter2D);
    double cpu2D_time = (rtclock() - t0) * 1000.0;


    float *d_in2D, *d_out2D;
    cudaCheckError(cudaMalloc(&d_in2D, bytes2D));
    cudaCheckError(cudaMalloc(&d_out2D, bytes2D));

    double t_2d_e2e = rtclock();

    cudaMemcpy(d_in2D, h_in2D, bytes2D, cudaMemcpyHostToDevice);

    dim3 block2D(BX, BY);
    dim3 grid2D((N2 + BX - 1)/BX, (N2 + BY - 1)/BY);

    // kernel-only
    double t_2d_kernel = rtclock();
    kernel2D_basic<<<grid2D, block2D>>>(d_in2D, d_out2D,
                                        d_filter2D,
                                        N2);
    cudaCheckError(cudaDeviceSynchronize());
    t_2d_kernel = (rtclock() - t_2d_kernel) * 1000.0;

    cudaMemcpy(h_out2D, d_out2D, bytes2D, cudaMemcpyDeviceToHost);

    t_2d_e2e = (rtclock() - t_2d_e2e) * 1000.0;

    cout << "CPU 2D time (ms): " << cpu2D_time << endl;
    cout << "GPU 2D kernel-only time (ms): " << t_2d_kernel << endl;
    cout << "GPU 2D end-to-end time (ms): " << t_2d_e2e << endl;

    check2D(h_ref2D, h_out2D);



    double t1 = rtclock();
    cpu_conv3D(h_in3D, h_ref3D, h_filter3D);
    double cpu3D_time = (rtclock() - t1) * 1000.0;


    float *d_in3D, *d_out3D;
    cudaCheckError(cudaMalloc(&d_in3D, bytes3D));
    cudaCheckError(cudaMalloc(&d_out3D, bytes3D));

    double t_3d_e2e = rtclock();

    cudaMemcpy(d_in3D, h_in3D, bytes3D, cudaMemcpyHostToDevice);

    dim3 block3D(BX_3D, BY_3D, BZ_3D);
    dim3 grid3D((N3 + BX_3D - 1)/BX_3D, (N3 + BY_3D - 1)/BY_3D, (N3 + BZ_3D - 1)/BZ_3D);

    double t_3d_kernel = rtclock();
    kernel3D_basic<<<grid3D, block3D>>>(d_in3D, d_out3D,
                                        d_filter3D,
                                        N3);
    cudaCheckError(cudaDeviceSynchronize());
    t_3d_kernel = (rtclock() - t_3d_kernel) * 1000.0;

    cudaMemcpy(h_out3D, d_out3D, bytes3D, cudaMemcpyDeviceToHost);

    t_3d_e2e = (rtclock() - t_3d_e2e) * 1000.0;

    cout << "CPU 3D time (ms): " << cpu3D_time << endl;
    cout << "GPU 3D kernel-only time (ms): " << t_3d_kernel << endl;
    cout << "GPU 3D end-to-end time (ms): " << t_3d_e2e << endl;

    check3D(h_ref3D, h_out3D);

    double t2d_opt_e2e = rtclock();

    cudaMemcpy(d_in2D, h_in2D, bytes2D, cudaMemcpyHostToDevice);

    double t2d_opt_kernel = rtclock();
    kernel2D_opt<<<grid2D, block2D>>>(d_in2D, d_out2D,
                                      N2);
    cudaCheckError(cudaDeviceSynchronize());
    t2d_opt_kernel = (rtclock() - t2d_opt_kernel) * 1000.0;

    cudaMemcpy(h_out2D, d_out2D, bytes2D, cudaMemcpyDeviceToHost);

    t2d_opt_e2e = (rtclock() - t2d_opt_e2e) * 1000.0;

    cout << "GPU 2D OPT kernel-only time (ms): " << t2d_opt_kernel << endl;
    cout << "GPU 2D OPT end-to-end time (ms): " << t2d_opt_e2e << endl;

    check2D(h_ref2D, h_out2D);

    double t3d_opt_e2e = rtclock();

    cudaMemcpy(d_in3D, h_in3D, bytes3D, cudaMemcpyHostToDevice);

    double t3d_opt_kernel = rtclock();
    kernel3D_opt<<<grid3D, block3D>>>(d_in3D, d_out3D,
                                      N3);
    cudaCheckError(cudaDeviceSynchronize());
    t3d_opt_kernel = (rtclock() - t3d_opt_kernel) * 1000.0;

    cudaMemcpy(h_out3D, d_out3D, bytes3D, cudaMemcpyDeviceToHost);

    t3d_opt_e2e = (rtclock() - t3d_opt_e2e) * 1000.0;

    cout << "GPU 3D OPT kernel-only time (ms): " << t3d_opt_kernel << endl;
    cout << "GPU 3D OPT end-to-end time (ms): " << t3d_opt_e2e << endl;

    check3D(h_ref3D, h_out3D);


    // ----------------------------------------------------------------------
    // CLEANUP
    // ----------------------------------------------------------------------
    cudaFree(d_in2D);
    cudaFree(d_out2D);
    cudaFree(d_in3D);
    cudaFree(d_out3D);
    cudaFree(d_filter2D);
    cudaFree(d_filter3D);

    free(h_in2D);
    free(h_ref2D);
    free(h_out2D);
    free(h_in3D);
    free(h_ref3D);
    free(h_out3D);

    return 0;
}
