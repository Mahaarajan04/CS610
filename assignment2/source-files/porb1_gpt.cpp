#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <papi.h>

#define INP_H (1 << 6)
#define INP_W (1 << 6)
#define INP_D (1 << 6)
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)

using namespace std;
using namespace std::chrono;

void cc_3d_blocked(const uint64_t* input,
                   const uint64_t (*kernel)[FIL_W][FIL_D],
                   uint64_t* result,
                   uint64_t outputHeight, uint64_t outputWidth,
                   uint64_t outputDepth, 
                   uint64_t blockX, uint64_t blockY, uint64_t blockZ) {
    for (uint64_t i = 0; i < outputHeight; i += blockX) {
        for (uint64_t j = 0; j < outputWidth; j += blockY) {
            for (uint64_t k = 0; k < outputDepth; k += blockZ) {
                for (uint64_t bi = 0; bi < blockX && (i + bi) < outputHeight; bi++) {
                    for (uint64_t bj = 0; bj < blockY && (j + bj) < outputWidth; bj++) {
                        for (uint64_t bk = 0; bk < blockZ && (k + bk) < outputDepth; bk++) {
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

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: ./runner <blockX> <blockY> <blockZ> <output_csv_file>" << endl;
        return 1;
    }

    int blockX = stoi(argv[1]);
    int blockY = stoi(argv[2]);
    int blockZ = stoi(argv[3]);
    string out_csv = argv[4];

    ofstream fout(out_csv);
    uint64_t* input = new uint64_t[INP_H * INP_W * INP_D];
    fill_n(input, INP_H * INP_W * INP_D, 1);

    uint64_t filter[FIL_H][FIL_W][FIL_D];
    for (int i = 0; i < FIL_H; i++)
        for (int j = 0; j < FIL_W; j++)
            for (int k = 0; k < FIL_D; k++)
                filter[i][j][k] = 1;

    uint64_t outputH = INP_H - FIL_H + 1;
    uint64_t outputW = INP_W - FIL_W + 1;
    uint64_t outputD = INP_D - FIL_D + 1;
    uint64_t* result = new uint64_t[outputH * outputW * outputD];
    memset(result, 0, sizeof(uint64_t) * outputH * outputW * outputD);

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        std::cerr << "PAPI library init error!" << std::endl;
        return 1;
    }
    PAPI_reset(PAPI_NULL);
    long long papi_val = 0;
    PAPI_start_counters((int[]){PAPI_L2_TCM}, 1);

    auto start = high_resolution_clock::now();
    cc_3d_blocked(input, filter, result, outputH, outputW, outputD, blockX, blockY, blockZ);
    auto end = high_resolution_clock::now();

    PAPI_stop_counters(&papi_val, 1);
    double duration = duration_cast<milliseconds>(end - start).count();
    fout << iter + 1 << "," << duration << "," << papi_val << "\n";

    delete[] input;
    delete[] result;

    fout.close();
    return 0;
}
