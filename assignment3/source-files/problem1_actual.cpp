#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <cassert>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::milliseconds;

const uint64_t TIMESTEPS = 100;

const double W_OWN = (1.0 / 7.0);
const double W_NEIGHBORS = (1.0 / 7.0);

const uint64_t NX = 66; // 64 interior points + 2 boundary points
const uint64_t NY = 66;
const uint64_t NZ = 66;
const uint64_t TOTAL_SIZE = NX * NY * NZ;

// a slightly looser epsilon for FP rounding across different orders of ops
const static double EPSILON = 1e-9;

/*---------------------------------------------------------------
  Baseline scalar stencil kernel
----------------------------------------------------------------*/
void stencil_3d_7pt(const double* curr, double* next) {
    for (int i = 1; i < (int)NX - 1; ++i) {
        for (int j = 1; j < (int)NY - 1; ++j) {
            for (int k = 1; k < (int)NZ - 1; ++k) {
                double neighbors_sum = 0.0;
                neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
                neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
                neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
                neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
                neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
                neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

                next[i * NY * NZ + j * NZ + k] =
                    W_OWN * curr[i * NY * NZ + j * NZ + k] +
                    W_NEIGHBORS * neighbors_sum;
            }
        }
    }
}

/*---------------------------------------------------------------
  Unrolled stencil kernel
----------------------------------------------------------------*/
#define UNROLL_FACTOR 4

void stencil_3d_7pt_unrolled(const double* curr, double* next) {
    for (int i = 1; i < (int)NX - 1; ++i) {
        for (int j = 1; j < (int)NY - 1; ++j) {
            for (int k = 1; k < (int)NZ - 1; k += UNROLL_FACTOR) {
#pragma unroll(UNROLL_FACTOR)
                for (int u = 0; u < UNROLL_FACTOR; ++u) {
                    int kk = k + u;
                    double neighbors_sum =
                        curr[(i + 1) * NY * NZ + j * NZ + kk] +
                        curr[(i - 1) * NY * NZ + j * NZ + kk] +
                        curr[i * NY * NZ + (j + 1) * NZ + kk] +
                        curr[i * NY * NZ + (j - 1) * NZ + kk] +
                        curr[i * NY * NZ + j * NZ + (kk + 1)] +
                        curr[i * NY * NZ + j * NZ + (kk - 1)];

                    next[i * NY * NZ + j * NZ + kk] =
                        W_OWN * curr[i * NY * NZ + j * NZ + kk] +
                        W_NEIGHBORS * neighbors_sum;
                }
            }
        }
    }
}

/*---------------------------------------------------------------
  Unrolled and parallelised stencil kernel
----------------------------------------------------------------*/


void stencil_3d_7pt_unrolled_parallel(const double* curr, double* next) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < (int)NX - 1; ++i) {
        for (int j = 1; j < (int)NY - 1; ++j) {
            int k = 1;
            // ensure that the last kk = k + (UNROLL_FACTOR - 1) ≤ NZ - 2
            for (; k <= (int(NZ) - 2) - (UNROLL_FACTOR - 1); k += UNROLL_FACTOR) {
#pragma unroll(UNROLL_FACTOR)
                for (int u = 0; u < UNROLL_FACTOR; ++u) {
                    int kk = k + u;

                    double neighbors_sum =
                        curr[(i + 1) * NY * NZ + j * NZ + kk] +
                        curr[(i - 1) * NY * NZ + j * NZ + kk] +
                        curr[i * NY * NZ + (j + 1) * NZ + kk] +
                        curr[i * NY * NZ + (j - 1) * NZ + kk] +
                        curr[i * NY * NZ + j * NZ + (kk + 1)] +
                        curr[i * NY * NZ + j * NZ + (kk - 1)];

                    next[i * NY * NZ + j * NZ + kk] =
                        W_OWN * curr[i * NY * NZ + j * NZ + kk] +
                        W_NEIGHBORS * neighbors_sum;
                }
            }
        }
    }
}

/*---------------------------------------------------------------
  simd kernel
----------------------------------------------------------------*/
void stencil_3d_7pt_simd(const double* __restrict__ curr, double* __restrict__ next) {
#pragma omp parallel for schedule(static)
    for (int i = 1; i < (int)NX - 1; ++i) {
        for (int j = 1; j < (int)NY - 1; ++j) {
#pragma omp simd
            for (int k = 1; k < (int)NZ - 1; ++k) {
                double neighbors_sum =
                    curr[(i + 1) * NY * NZ + j * NZ + k] +
                    curr[(i - 1) * NY * NZ + j * NZ + k] +
                    curr[i * NY * NZ + (j + 1) * NZ + k] +
                    curr[i * NY * NZ + (j - 1) * NZ + k] +
                    curr[i * NY * NZ + j * NZ + (k + 1)] +
                    curr[i * NY * NZ + j * NZ + (k - 1)];

                next[i * NY * NZ + j * NZ + k] =
                    W_OWN * curr[i * NY * NZ + j * NZ + k] +
                    W_NEIGHBORS * neighbors_sum;
            }
        }
    }
}


/*---------------------------------------------------------------
  Utility: init hot-spot; checksum helpers
----------------------------------------------------------------*/
inline void init_hotspot(double* grid) {
    std::fill_n(grid, TOTAL_SIZE, 0.0);
    grid[(NX/2) * NY * NZ + (NY/2) * NZ + (NZ/2)] = 100.0;
}

double compute_sum(const double* grid) {
    double total = 0.0;
    for (size_t i = 0; i < TOTAL_SIZE; ++i) total += grid[i];
    return total;
}

/*---------------------------------------------------------------
  Main driver
----------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    cout << "Unroll factor = " << UNROLL_FACTOR << "\n";

    // Three buffers so we can keep runs independent
    auto* gridA = new double[TOTAL_SIZE];
    auto* gridB = new double[TOTAL_SIZE];
    auto* gridC = new double[TOTAL_SIZE];

    // ---------- Baseline run ----------
    init_hotspot(gridA);
    init_hotspot(gridB);
    //init_hotspot(gridC);

    double* curr = gridA;
    double* next = gridB;

    auto start = HR::now();
    for (int t = 0; t < (int)TIMESTEPS; ++t) {
        stencil_3d_7pt(curr, next);
        std::swap(curr, next);
    }
    auto end = HR::now();
    auto base_time = duration_cast<milliseconds>(end - start).count();

    double base_center = curr[(NX/2)*NY*NZ + (NY/2)*NZ + (NZ/2)];
    double base_sum    = compute_sum(curr);

    cout << "Baseline kernel time: " << base_time << " ms\n";
    cout << "Baseline center: " << base_center << ", total sum: " << base_sum << endl;

    // ---------- Unrolled run (RE-INITIALIZE!) ----------
    init_hotspot(gridB);
    init_hotspot(gridC);
    curr = gridB;
    next = gridC;

    start = HR::now();
    for (int t = 0; t < (int)TIMESTEPS; ++t) {
        stencil_3d_7pt_unrolled_parallel(curr, next);
        std::swap(curr, next);
    }
    end = HR::now();
    auto unroll_time = duration_cast<milliseconds>(end - start).count();

    double unroll_center = curr[(NX/2)*NY*NZ + (NY/2)*NZ + (NZ/2)];
    double unroll_sum    = compute_sum(curr);

    cout << "Unrolled kernel time: " << unroll_time << " ms\n";
    cout << "Unrolled center: " << unroll_center << ", total sum: " << unroll_sum << endl;

    // ---------- Correctness ----------
    assert(std::fabs(base_center - unroll_center) < EPSILON &&
           "Center value mismatch between baseline and unrolled versions!");
    assert(std::fabs(base_sum - unroll_sum) < EPSILON &&
           "Total sum mismatch between baseline and unrolled versions!");

    cout << "Results verified: outputs match within tolerance.\n";

    delete[] gridA;
    delete[] gridB;
    delete[] gridC;
    return EXIT_SUCCESS;
}
