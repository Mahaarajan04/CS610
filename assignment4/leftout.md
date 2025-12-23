- Nsight stuff for prob1, experimentation for prob2, sorting before writes
- 2a out of mem for 2^33 N... 2^32 works
- memadvise prefecth sync no use in part i
- make and get perf metrics with sir's flags
seq prob3 takes 432s

mahaarajan@gpu3 ~/CS610/assignment4/source-files/220600/problem2-dir $ ./p2a
No differences found
CPU time (ms):          157
Kernel-only time (ms):  1.84826
End-to-end time (ms):   139
Ran for 2^ 25
No differences found
CPU time (ms):          224
Kernel-only time (ms):  61.0263
End-to-end time (ms):   395
Pow used is 2^ 25
mahaarajan@gpu3 ~/CS610/assignment4/source-files/220600/problem2-dir $ ./p2a
No differences found
CPU time (ms):          509
Kernel-only time (ms):  7.30435
End-to-end time (ms):   526
Ran for 2^ 27
mahaarajan@gpu3 ~/CS610/assignment4/source-files/220600/problem2-dir $ ./p2b
No differences found
CPU time (ms):          829
Kernel-only time (ms):  918.837
End-to-end time (ms):   2199
Pow used is 2^ 27
mahaarajan@gpu3 ~/CS610/assignment4/source-files/220600/problem2-dir $ 
mahaarajan@gpu3 ~/CS610/assignment4/source-files/220600/problem2-dir $ ./p2b
No differences found
CPU time (ms):          1689
Kernel-only time (ms):  492.145
End-to-end time (ms):   3163
Pow used is 2^ 28
mahaarajan@gpu3 ~/CS610/assignment4/source-files/220600/problem2-dir $ ./p2b
No differences found
CPU time (ms):          8706
Kernel-only time (ms):  1656.26
End-to-end time (ms):   12701
Pow used is 2^ 30
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p2b
No differences found
CPU time (ms):          4101
Kernel-only time (ms):  56.2112
End-to-end time (ms):   4135
Ran for 2^ 30
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ nvcc -O2 -o p2b problem2b.cu 
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p2b
GPUassert: out of memory problem2b.cu 135
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ nvcc -O2 -o p2b problem2b.cu 
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p2b
No differences found
CPU time (ms):          18625
Kernel-only time (ms):  224.739
End-to-end time (ms):   18006
Ran for 2^ 32
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ 

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_1
CPU result count: 11608
Kernel-only time: 29024.781 ms
End-to-end time : 29025.824 ms.    (naive version)

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_2
✅ Result count (v2): 11608
🚀 Kernel-only time: 23657.271 ms
📦 End-to-end time : 23658.400 ms.  (multi launch)

Total satisfying points: 11608
Kernel-only time: 23609.631 ms
End-to-end time: 23610.764 ms (naive sharing)

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_2_3
Running Version 2 (Optimized) with Cooperative Shared Load + Step Passing
Partitioning across 13 r1 slices...
Launching kernel for r1 = 0
Launching kernel for r1 = 1
Launching kernel for r1 = 2
Launching kernel for r1 = 3
Launching kernel for r1 = 4
Launching kernel for r1 = 5
Launching kernel for r1 = 6
Launching kernel for r1 = 7
Launching kernel for r1 = 8
Launching kernel for r1 = 9
Launching kernel for r1 = 10
Launching kernel for r1 = 11
Launching kernel for r1 = 12
Total satisfying points: 11608
Kernel-only time: 23743.688 ms
End-to-end time : 23744.670 ms (Cooperative Shared Load + Step Passing)

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_2_3
Running Version 2 (Optimized) with Cooperative Shared Load + Step Passing+ Loop Unrolling
Partitioning across 13 r1 slices...
Launching kernel for r1 = 0
Launching kernel for r1 = 1
Launching kernel for r1 = 2
Launching kernel for r1 = 3
Launching kernel for r1 = 4
Launching kernel for r1 = 5
Launching kernel for r1 = 6
Launching kernel for r1 = 7
Launching kernel for r1 = 8
Launching kernel for r1 = 9
Launching kernel for r1 = 10
Launching kernel for r1 = 11
Launching kernel for r1 = 12
Total satisfying points: 11608
Kernel-only time: 23480.557 ms
End-to-end time : 23481.613 ms


mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./lol
Total satisfying points: 11608
Kernel-only time: 25228.600 ms
End-to-end time : 25229.592 ms
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p
p1a              p1d              p2b              p3_1             p3_2_3           p4               problem3-v0.out  
p1b              p1f              p2c              p3_2             p3_2_4           problem1.out     problem4.out     
p1c              p2a              p3_0             p3_2_2           p3_cpp           problem2.out     
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_
p3_0    p3_1    p3_2    p3_2_2  p3_2_3  p3_2_4  p3_cpp  
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_
p3_0    p3_1    p3_2    p3_2_2  p3_2_3  p3_2_4  p3_cpp  
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_1
CPU result count: 11608
Kernel-only time: 27315.594 ms
End-to-end time : 27316.660 ms

marginally better

UVM:
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_3
Total satisfying points: 11608
Kernel-only time: 24410.770 ms
End-to-end time : 24414.703 ms

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_3_2
Total satisfying points: 11608
Kernel-only time: 24538.041 ms
End-to-end time : 24541.602 ms 
(Nothing great... not much of speedup...marginal only)

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ mv problem3_v3_single_alligned.cu problem3_v2_single_alligned.cu 
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_2
Total satisfying points: 11608
Kernel-only time: 29740.359 ms
End-to-end time : 29741.453 ms
(simgle kernel no use)

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z27grid_kernel_partitioned_uvmPKdS0_PKidPdPlPiil' for 'sm_80'
ptxas info    : Function properties for _Z27grid_kernel_partitioned_uvmPKdS0_PKidPdPlPiil
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 92 registers, 12228 bytes smem, 424 bytes cmem[0]
problem3_v3_alligned.cu(148): warning #1650-D: result of call is not used

problem3_v3_alligned.cu(153): warning #1650-D: result of call is not used (from 255 reg to 92)

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_3
Total satisfying points: 11608
Kernel-only time: 26957.318 ms
End-to-end time : 26957.947 ms
mahaarajan@gpu3 ~/CS610/assignment4/source-files $  (single in 3)


problem3_v2_alligned_warp.cu(163): warning #1650-D: result of call is not used

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z34grid_kernel_partitioned_stripminedPKdS0_PKidPdPlPiil' for 'sm_80'
ptxas info    : Function properties for _Z34grid_kernel_partitioned_stripminedPKdS0_PKidPdPlPiil
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 114 registers, 960 bytes smem, 424 bytes cmem[0]
problem3_v2_alligned_warp.cu(159): warning #1650-D: result of call is not used

problem3_v2_alligned_warp.cu(163): warning #1650-D: result of call is not used

problem3_v2_alligned_warp.cu: In function ‘int main()’:
problem3_v2_alligned_warp.cu:159:38: warning: ignoring return value of ‘int fscanf(FILE*, const char*, ...)’ declared with attribute ‘warn_unused_result’ [-Wunused-result]
  159 |     for (int i = 0; i < 30; i++) fscanf(fg, "%lf", &h_grid[i]);
      |                                ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~
problem3_v2_alligned_warp.cu:163:39: warning: ignoring return value of ‘int fscanf(FILE*, const char*, ...)’ declared with attribute ‘warn_unused_result’ [-Wunused-result]
  163 |     for (int i = 0; i < 120; i++) fscanf(fd, "%lf", &h_disp[i]);
      |                                 ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_2
Total satisfying points: 11608
Kernel-only time: 28936.682 ms
End-to-end time : 28937.869 ms (warp...not great)

thrust v1
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_4
Total satisfying points (unsorted): 11608
Total satisfying points (sorted): 11608
Kernel-only time    : 24539.746 ms
Thrust sort time    : 0.850 ms
End-to-end time     : 24541.059 ms

thrust v2
mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_4
Raw kernel output count = 11608
After copy_if compact count = 11608

================ Timing Report ================
Kernel compute time       : 24535.971 ms
Thrust transform time     : 0.308 ms
Thrust copy_if time       : 0.638 ms
Thrust sort time          : 0.277 ms
End-to-end time           : 24734.830 ms
================================================.

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_3
Total satisfying points: 11608
Kernel-only time: 56991.145 ms
End-to-end time : 56993.621 ms (const., adding this inc time... but did not pass as arg)

mahaarajan@gpu3 ~/CS610/assignment4/source-files $ ./p3_3
Total satisfying points: 11608
Kernel-only time: 25226.385 ms
End-to-end time : 25226.885 ms (const., better with const)

q4:
mahaarajan@gpu3 ~/CS610/assignment4/source-files/prob4 $ ./p4_1
CPU 2D time (ms): 3749.43
GPU 2D kernel-only time (ms): 4.00496
GPU 2D end-to-end time (ms): 1067.97
No differences found
CPU 3D time (ms): 7956.04
GPU 3D kernel-only time (ms): 10.391
GPU 3D end-to-end time (ms): 703.195
No differences found
GPU 2D OPT kernel-only time (ms): 5.43308
GPU 2D OPT end-to-end time (ms): 2736.95
No differences found
GPU 3D OPT kernel-only time (ms): 5.93495
GPU 3D OPT end-to-end time (ms): 541.091
No differences found
(16,16) (4,4,4)

CPU 2D time (ms): 4235.93
GPU 2D kernel-only time (ms): 35.048
GPU 2D end-to-end time (ms): 2023.97
No differences found
CPU 3D time (ms): 6343.06
GPU 3D kernel-only time (ms): 10.2952
GPU 3D end-to-end time (ms): 643.269
No differences found
GPU 2D OPT kernel-only time (ms): 21.7721
GPU 2D OPT end-to-end time (ms): 1023.64
No differences found
GPU 3D OPT kernel-only time (ms): 5.93996
GPU 3D OPT end-to-end time (ms): 392.847
No differences found
(4,4) (4,4,4).... use this and report ki there is high variance

mahaarajan@gpu3 ~/CS610/assignment4/source-files/prob4 $ ./p4_1
CPU 2D time (ms): 4227.41
GPU 2D kernel-only time (ms): 43.3469
GPU 2D end-to-end time (ms): 1144.7
No differences found
CPU 3D time (ms): 6950.61
GPU 3D kernel-only time (ms): 39.1099
GPU 3D end-to-end time (ms): 715.393
No differences found
GPU 2D OPT kernel-only time (ms): 73.4241
GPU 2D OPT end-to-end time (ms): 2590.15
No differences found
GPU 3D OPT kernel-only time (ms): 16.855
GPU 3D OPT end-to-end time (ms): 535.088
No differences found
(same as above)

tried padding but more noisier outputs
mahaarajan@gpu3 ~/CS610/assignment4/source-files/prob4 $ ./p4_2
CPU 2D time (ms): 3943.91
GPU 2D kernel-only time (ms): 33.504
GPU 2D end-to-end time (ms): 1099.8
No differences found
CPU 3D time (ms): 7315.72
GPU 3D kernel-only time (ms): 20.1561
GPU 3D end-to-end time (ms): 553.073
No differences found
GPU 2D OPT kernel-only time (ms): 47.9202
GPU 2D OPT end-to-end time (ms): 1215.83
No differences found
GPU 3D OPT kernel-only time (ms): 8.89206
GPU 3D OPT end-to-end time (ms): 1072.23
No differences found

mahaarajan@gpu3 ~/CS610/assignment4/source-files/220600/problem2-dir $ ./p2a
No differences found
CPU time (ms):          4547
Kernel-only time (ms):  58.2351
End-to-end time (ms):   4202
Ran for 2^ 30