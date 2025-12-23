Ideal candidate for SIMD because each iteration is independent.

To do for 512

for 128
Running ./problem3.out for 5 runs...
-----------------------------------------
Run #1:
  Scalar = 279 ms, SSE = 261 ms, AVX2 = 272 ms
Run #2:
  Scalar = 271 ms, SSE = 257 ms, AVX2 = 277 ms
Run #3:
  Scalar = 283 ms, SSE = 282 ms, AVX2 = 296 ms
Run #4:
  Scalar = 274 ms, SSE = 264 ms, AVX2 = 297 ms
Run #5:
  Scalar = 278 ms, SSE = 255 ms, AVX2 = 291 ms

-----------------------------------------
Averages over 5 runs:
  Scalar avg time : 277.00 ms
  SSE avg time    : 263.80 ms
  AVX2 avg time   : 286.60 ms

Average speedups (relative to Scalar):
  SSE  Speedup = 1.05x
  AVX2 Speedup = .96x
-----------------------------------------

for 256

-----------------------------------------
Run #1:
  Scalar = 2332 ms, SSE = 2257 ms, AVX2 = 2229 ms
Run #2:
  Scalar = 2302 ms, SSE = 2214 ms, AVX2 = 2244 ms
Run #3:
  Scalar = 2305 ms, SSE = 2418 ms, AVX2 = 2441 ms
Run #4:
  Scalar = 2510 ms, SSE = 2437 ms, AVX2 = 2454 ms
Run #5:
  Scalar = 2526 ms, SSE = 2448 ms, AVX2 = 2436 ms

-----------------------------------------
Averages over 5 runs:
  Scalar avg time : 2395.00 ms
  SSE avg time    : 2354.80 ms
  AVX2 avg time   : 2360.80 ms

Average speedups (relative to Scalar):
  SSE  Speedup = 1.01x
  AVX2 Speedup = 1.01x
-----------------------------------------

for 512
Running ./problem3.out for 5 runs...
-----------------------------------------
Run #1:
  Scalar = 21871 ms, SSE = 21056 ms, AVX2 = 20925 ms
Run #2:
  Scalar = 21624 ms, SSE = 20972 ms, AVX2 = 20924 ms
Run #3:
  Scalar = 22245 ms, SSE = 22424 ms, AVX2 = 21626 ms
Run #4:
  Scalar = 21751 ms, SSE = 20842 ms, AVX2 = 20725 ms
Run #5:
  Scalar = 21736 ms, SSE = 20729 ms, AVX2 = 20541 ms

-----------------------------------------
Averages over 5 runs:
  Scalar avg time : 21845.40 ms
  SSE avg time    : 21204.60 ms
  AVX2 avg time   : 20948.20 ms

Average speedups (relative to Scalar):
  SSE  Speedup = 1.03x
  AVX2 Speedup = 1.04x
-----------------------------------------


not cache friendly....

g++ -std=c++17 -masm=att -msse4 -mavx2 -march=native -fopenmp      -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions      -fno-rtti -fcf-protection=none -O2 problem3_new.cpp -o problem3.out