OG
result pnts: 11608
Total time = 362.939169 seconds

mahaarajan@csews8 ~/CS610/assignment3/source-files $ ./problem4-v3.out 
result pnts: 11608
Total time = 258.401334 seconds

buffering... no sig gain (v4)
result pnts: 11608
Total time = 266.812988 seconds

mahaarajan@csews8 ~/CS610/assignment3/source-files $ ./problem4-v4.out
result pnts: 11608
Total time = 266.812988 seconds

mahaarajan@csews8 ~/CS610/assignment3/source-files $ ./problem4-unroll (unroll=4)
result pnts: 11608
Total time = 252.736971 seconds

mahaarajan@csews8 ~/CS610/assignment3/source-files $ ./problem4-unroll (unroll=8)
result pnts: 11608
Total time = 264.745187 seconds

loop permutes and tiling inc time....a lot about 850s (v5)

loop permutes alone (v6)... not much gains
result pnts: 11608
Total time = 295.842870 seconds

best yet is v3...


Part2:
v1
1st critical writes not at the end: but correctness issue in order of file write
mahaarajan@csews8 ~/CS610/assignment3/source-files $ ./problem4-2-v1.out 
result pnts: 11608
Total time = 70.949647 seconds

v2
Correct everything
mahaarajan@csews8 ~/CS610/assignment3/source-files $ ./problem4-2-v2.out 
result pnts: 11608
Total time = 73.429063 seconds



gcc -std=c17 -O3 -masm=att -msse4 -mavx2 -march=native -fopenmp     -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions     -fcf-protection=none -Wall -Wextra problem4-2-v2.c -o problem4-2-v2.out -pthread

