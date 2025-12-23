og:15ms

unroll alone
2: 25ms
4: 48ms
8: 13ms
16 : 14ms
32 : 14ms
(not much improvement after 8)

unroll+parallel (collapse 2) no difference without collapse 2 as we have a large kernel anyhow
1: 3ms
2: 5ms
4: 8ms
8: 3ms
16: 3ms
32: 3ms
(unroll factor not much effect)

parallel+simd: 4ms

on csews8 server

g++ -O3 -fopenmp -march=native problem1_nodef.cpp -o prob1

prob1, actual is the main one...
