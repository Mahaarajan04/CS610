#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <x86intrin.h>
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint32_t NX = 128;
const uint32_t NY = 128;
const uint32_t NZ = 128;
const uint64_t TOTAL_SIZE = (NX * NY * NZ);

const uint32_t N_ITERATIONS = 100;
const uint64_t INITIAL_VAL = 1000000;

void scalar_3d_gradient(const uint64_t* A, uint64_t* B) {
  const uint64_t stride_i = (NY * NZ);
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
        // A[i+1, j, k]
        int A_right = A[base_idx + stride_i];
        // A[i-1, j, k]
        int A_left = A[base_idx - stride_i];
        B[base_idx] = A_right - A_left;
      }
    }
  }
}

// -------------------- SSE4 version (128-bit) --------------------
void sse_3d_gradient(const uint64_t* A, uint64_t* B) {
    const uint64_t stride_i = (NY * NZ);
    for (int i = 1; i < NX - 1; ++i) {
        for (int j = 0; j < NY; ++j) {
            int k = 0;
            for (; k + 1 < NZ; k += 2) { // 2 x 64-bit = 128-bit
                __m128i va_r = _mm_loadu_si128((__m128i*)&A[(i + 1) * NY * NZ + j * NZ + k]);
                __m128i va_l = _mm_loadu_si128((__m128i*)&A[(i - 1) * NY * NZ + j * NZ + k]);
                __m128i vout = _mm_sub_epi64(va_r, va_l);
                _mm_storeu_si128((__m128i*)&B[i * NY * NZ + j * NZ + k], vout);
            }
            // remainder
            for (; k < NZ; ++k) {
                uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
                B[base_idx] = A[base_idx + stride_i] - A[base_idx - stride_i];
            }
        }
    }
}

// -------------------- AVX2 version (256-bit) --------------------
void avx2_3d_gradient(const uint64_t* A, uint64_t* B) {
    const uint64_t stride_i = (NY * NZ);
    for (int i = 1; i < NX - 1; ++i) {
        for (int j = 0; j < NY; ++j) {
            int k = 0;
            for (; k + 3 < NZ; k += 4) { // 4 x 64-bit = 256-bit
                __m256i va_r = _mm256_loadu_si256((__m256i*)&A[(i + 1) * NY * NZ + j * NZ + k]);
                __m256i va_l = _mm256_loadu_si256((__m256i*)&A[(i - 1) * NY * NZ + j * NZ + k]);
                __m256i vout = _mm256_sub_epi64(va_r, va_l);
                _mm256_storeu_si256((__m256i*)&B[i * NY * NZ + j * NZ + k], vout);
            }
            // remainder
            for (; k < NZ; ++k) {
                uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
                B[base_idx] = A[base_idx + stride_i] - A[base_idx - stride_i];
            }
        }
    }
}


long compute_checksum(const uint64_t* grid) {
  uint64_t sum = 0;
  for (int i = 1; i < (NX - 1); i++) {
    for (int j = 0; j < NY; j++) {
      for (int k = 0; k < NZ; k++) {
        sum += grid[i * NY * NZ + j * NZ + k];
      }
    }
  }
  return sum;
}

int main() {
  auto* i_grid = new uint64_t[TOTAL_SIZE];
   for (int i = 0; i < NX; i++) {
    for (int j = 0; j < NY; j++) {
      for (int k = 0; k < NZ; k++) {
        i_grid[i*NY*NZ+j*NZ+k] = (INITIAL_VAL + i +
                                  2 * j + 3 * k);
      }
    }
  }

  auto* o_grid1 = new uint64_t[TOTAL_SIZE];
  std::fill_n(o_grid1, TOTAL_SIZE, 0);

  auto start = HR::now();
  for (int iter = 0; iter < N_ITERATIONS; ++iter) {
    scalar_3d_gradient(i_grid, o_grid1);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Scalar kernel time (ms): " << duration << "\n";

  // Compare checksum with vector versions
  uint64_t scalar_checksum = compute_checksum(o_grid1);
  cout << "Checksum: " << scalar_checksum << "\n";

  // Assert the checksum for vectors variants
  
  delete[] i_grid;
  delete[] o_grid1;

  return EXIT_SUCCESS;
}
