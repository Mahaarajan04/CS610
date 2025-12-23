#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <chrono>
#include <papi.h>

using namespace std;
using std::uint64_t;

#define INP_H (1 << 7)
#define INP_W (1 << 7)
#define INP_D (1 << 7)
//Large enough so that would never fit in cache
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)

// Blocked 3D convolution
void cc_3d_blocked(const uint64_t* input,
                   const uint64_t (*kernel)[FIL_W][FIL_D],
                   uint64_t* result,
                   uint64_t outputHeight, uint64_t outputWidth,
                   uint64_t outputDepth,
                   uint64_t bx, uint64_t by, uint64_t bz) {
  for (uint64_t i = 0; i < outputHeight; i += bx) {
    for (uint64_t j = 0; j < outputWidth; j += by) {
      for (uint64_t k = 0; k < outputDepth; k += bz) {
        for (uint64_t bi = 0; bi < bx && i + bi < outputHeight; ++bi) {
          for (uint64_t bj = 0; bj < by && j + bj < outputWidth; ++bj) {
            for (uint64_t bk = 0; bk < bz && k + bk < outputDepth; ++bk) {
              uint64_t sum = 0;
              for (uint64_t ki = 0; ki < FIL_H; ++ki) {
                for (uint64_t kj = 0; kj < FIL_W; ++kj) {
                  for (uint64_t kk = 0; kk < FIL_D; ++kk) {
                    sum += input[(i+bi+ki)*INP_W*INP_D + (j+bj+kj)*INP_D + (k+bk+kk)] *
                           kernel[ki][kj][kk];
                  }
                }
              }
              result[(i+bi)*outputWidth*outputDepth + (j+bj)*outputDepth + (k+bk)] += sum;
            }
          }
        }
      }
    }
  }
}

// Naive unblocked version
void cc_3d_no_padding(const uint64_t* input,
                      const uint64_t (*kernel)[FIL_W][FIL_D],
                      uint64_t* result,
                      uint64_t outputHeight, uint64_t outputWidth,
                      uint64_t outputDepth) {
  for (uint64_t i = 0; i < outputHeight; ++i) {
    for (uint64_t j = 0; j < outputWidth; ++j) {
      for (uint64_t k = 0; k < outputDepth; ++k) {
        uint64_t sum = 0;
        for (uint64_t ki = 0; ki < FIL_H; ++ki) {
          for (uint64_t kj = 0; kj < FIL_W; ++kj) {
            for (uint64_t kk = 0; kk < FIL_D; ++kk) {
              sum += input[(i+ki)*INP_W*INP_D + (j+kj)*INP_D + (k+kk)] *
                     kernel[ki][kj][kk];
            }
          }
        }
        result[i * outputWidth * outputDepth + j * outputDepth + k] += sum;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (!(argc==2 || argc==4)) {
    cerr << "Usage: " << argv[0] << " <blockX> <blockY> <blockZ> [--noblock]\n";
    return 1;
  }
  bool use_blocking = true;
  uint64_t bx = 0, by = 0, bz = 0;

  if (argc == 2) {
    if( strcmp(argv[1], "--noblock") == 0) use_blocking = false;
    else{
        cerr << "Usage: " << argv[0] << " <blockX> <blockY> <blockZ> [--noblock]\n";
        return 1;
    } 
  }

  if (use_blocking) {
    bx = atoi(argv[1]);
    by = atoi(argv[2]);
    bz = atoi(argv[3]);
  }

  const uint64_t outputHeight = INP_H - FIL_H + 1;
  const uint64_t outputWidth  = INP_W - FIL_W + 1;
  const uint64_t outputDepth  = INP_D - FIL_D + 1;

  uint64_t* input = new uint64_t[INP_H * INP_W * INP_D];
  std::fill_n(input, INP_H * INP_W * INP_D, 1);

  uint64_t kernel[FIL_H][FIL_W][FIL_D] = {{{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}};

  uint64_t* result = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
   int retval;
    // Initialize the PAPI library
    retval = PAPI_library_init(PAPI_VER_CURRENT);
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

    // Add L2 total cache misses to the event set
    retval = PAPI_add_event(event_set, PAPI_L2_TCM);
    if (retval != PAPI_OK) {
        std::cerr << "Failed to add PAPI_L2_TCM: " << PAPI_strerror(retval) << std::endl;
        return 1;
    }

    // Start counting
    retval = PAPI_start(event_set);
    if (retval != PAPI_OK) {
        std::cerr << "PAPI_start failed: " << PAPI_strerror(retval) << std::endl;
        return 1;
    }

  auto start = std::chrono::high_resolution_clock::now();

  if (use_blocking) {
    cc_3d_blocked(input, kernel, result, outputHeight, outputWidth, outputDepth, bx, by, bz);
  } else {
    cc_3d_no_padding(input, kernel, result, outputHeight, outputWidth, outputDepth);
  }

  auto end = std::chrono::high_resolution_clock::now();
  long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

     long long values[1];
    retval = PAPI_stop(event_set, values);
    if (retval != PAPI_OK) {
        std::cerr << "PAPI_stop failed: " << PAPI_strerror(retval) << endl;
        return 1;
    }

    // Cleanup
    PAPI_cleanup_eventset(event_set);
    PAPI_destroy_eventset(&event_set);
    PAPI_shutdown();

  // Output Results
  cout << "========== Run Stats ==========\n";
  cout << "Blocking: " << (use_blocking ? "Yes" : "No") << endl;
  if (use_blocking) {
    cout << "Block sizes: X=" << bx << ", Y=" << by << ", Z=" << bz << endl;
  }
  cout << "Runtime (microseconds): " << elapsed_us << endl;
  cout << "L2 Cache Misses: " << values[0] << endl;

  delete[] input;
  delete[] result;
  return 0;
}
