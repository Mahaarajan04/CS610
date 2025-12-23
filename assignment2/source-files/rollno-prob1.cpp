#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <papi.h>

using std::cerr;
using std::cout;
using std::endl;
using std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t;

#define INP_H (1 << 7)
#define INP_W (1 << 7)
#define INP_D (1 << 7)
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)

/** Cross-correlation with blocking */
void cc_3d_blocked(const uint64_t* input,
                     const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                     const uint64_t outputHeight, const uint64_t outputWidth,
                     const uint64_t outputDepth, const uint64_t blockSize) {
  for (uint64_t i = 0; i < outputHeight; i += blockSize) {
    for (uint64_t j = 0; j < outputWidth; j += blockSize) {
      for (uint64_t k = 0; k < outputDepth; k += blockSize) {
        for (uint64_t bi = 0; bi < blockSize && (i + bi) < outputHeight; bi++) {
          for (uint64_t bj = 0; bj < blockSize && (j + bj) < outputWidth; bj++) {
            for (uint64_t bk = 0; bk < blockSize && (k + bk) < outputDepth; bk++) {
              uint64_t sum = 0;
              for (uint64_t ki = 0; ki < FIL_H; ki++) {
                for (uint64_t kj = 0; kj < FIL_W; kj++) {
                  for (uint64_t kk = 0; kk < FIL_D; kk++) {
                    sum += input[(i + bi + ki) * INP_W * INP_D +
                                 (j + bj + kj) * INP_D + (k + bk + kk)] *
                           kernel[ki][kj][kk];
                  }
                }
              }
              result[(i + bi) * outputWidth * outputDepth +
                     (j + bj) * outputDepth + (k + bk)] += sum;
            }
          }
        }
      }
    }
  }
}


/** Cross-correlation without padding */
void cc_3d_no_padding(const uint64_t* input,
                      const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                      const uint64_t outputHeight, const uint64_t outputWidth,
                      const uint64_t outputDepth) {
  for (uint64_t i = 0; i < outputHeight; i++) {
    for (uint64_t j = 0; j < outputWidth; j++) {
      for (uint64_t k = 0; k < outputDepth; k++) {
        uint64_t sum = 0;
        for (uint64_t ki = 0; ki < FIL_H; ki++) {
          for (uint64_t kj = 0; kj < FIL_W; kj++) {
            for (uint64_t kk = 0; kk < FIL_D; kk++) {
              sum += input[(i + ki) * INP_W * INP_D + (j + kj) * INP_D +
                           (k + kk)] *
                     kernel[ki][kj][kk];
            }
          }
        }
        result[i * outputWidth * outputDepth + j * outputDepth + k] += sum;
      }
    }
  }
}

int main() {
  uint64_t* input = new uint64_t[INP_H * INP_W * INP_D];
  std::fill_n(input, INP_H * INP_W * INP_D, 1);

  uint64_t filter[FIL_H][FIL_W][FIL_D] = {{{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}};

  uint64_t outputHeight = INP_H - FIL_H + 1;
  uint64_t outputWidth = INP_W - FIL_W + 1;
  uint64_t outputDepth = INP_D - FIL_D + 1;

  auto* result = new uint64_t[outputHeight * outputWidth * outputDepth]{0};

  // --- PAPI and timing start ---
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT) {
    std::cerr << "PAPI_library_init failed" << std::endl;
    return 1;
  }
  int event_set = PAPI_NULL;
  retval = PAPI_create_eventset(&event_set);
  if (retval != PAPI_OK) {
    std::cerr << "Failed to create event set: " << PAPI_strerror(retval) << std::endl;
    return 1;
  }
  retval = PAPI_add_event(event_set, PAPI_L2_TCM);
  if (retval != PAPI_OK) {
    std::cerr << "Failed to add PAPI_L2_TCM: " << PAPI_strerror(retval) << std::endl;
    return 1;
  }
  retval = PAPI_start(event_set);
  if (retval != PAPI_OK) {
    std::cerr << "PAPI_start failed: " << PAPI_strerror(retval) << std::endl;
    return 1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  // --- PAPI and timing end ---

  cc_3d_no_padding(input, filter, result, outputHeight, outputWidth, outputDepth);

  // --- PAPI and timing stop ---
  auto end = std::chrono::high_resolution_clock::now();
  long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  long long values[1];
  retval = PAPI_stop(event_set, values);
  if (retval != PAPI_OK) {
    std::cerr << "PAPI_stop failed: " << PAPI_strerror(retval) << std::endl;
    return 1;
  }
  PAPI_cleanup_eventset(event_set);
  PAPI_destroy_eventset(&event_set);
  PAPI_shutdown();
  // --- PAPI and timing stop ---

  // cout << "3D convolution without padding:\n";
  // for (uint64_t i = 0; i < outputHeight; i++) {
  //   for (uint64_t j = 0; j < outputWidth; j++) {
  //     for (uint64_t k = 0; k < outputDepth; k++) {
  //       cout << result[i * outputWidth * outputDepth + j * outputDepth + k]
  //            << " ";
  //     }
  //     cout << "\n";
  //   }
  //   cout << "\n";
  // }

  cout << "========== Run Stats ==========\n";
  cout << "Runtime (microseconds): " << elapsed_us << endl;
  cout << "L2 Cache Misses: " << values[0] << endl;

  delete[] result;

  return EXIT_SUCCESS;
}
