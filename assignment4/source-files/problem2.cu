#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <iterator>

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

// Size this based on the kernel you are executing
const uint64_t N = (1 << 3);

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void cte_sum() {}

__global__ void uvm_sum() {}

__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt,
                           const uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if (w_ref[i] != w_opt[i]) {
      cout << "Differences found between the two arrays.\n";
      assert(false);
    }
  }
  cout << "No differences found between base and test versions\n";
}

__host__ void inclusive_prefix_sum(const uint32_t* input, uint32_t* output) {
  output[0] = input[0];
  for (uint64_t i = 1; i < N; i++) {
    output[i] = output[i - 1] + input[i];
  }
}

int main() {
  auto* h_input = new uint32_t[N];
  std::fill_n(h_input, N, 1);

  std::inclusive_scan(h_input, h_input + N,
                      std::ostream_iterator<int>(std::cout, " "));
  cout <<"\n";

  auto* h_output_cpu = new uint32_t[N];
  inclusive_prefix_sum(h_input, h_output_cpu);
  cout << h_output_cpu[N - 1] << "\n";

  // TODO: Use a CUDA kernel without UVM, time your code

  // TODO: Use a CUDA kernel with UVM, time your code

  delete[] h_input;
  delete[] h_output_cpu;

  return EXIT_SUCCESS;
}
