// rollno-prob4-v1.c  —  Inlined, compute-only baseline (no per-iteration I/O)
// Build:  gcc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra -o prob4_v1.out rollno-prob4-v1.c
// Run:    ./prob4_v1.out

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)

// a[120] layout: for t=0..9 (10 constraints), block is:
//   c[t][0..9], d[t], ey[t]              (10 + 2 = 12 doubles per constraint)  → 10*12 = 120
// b[30] layout: for v=0..9 (x1..x10), triple is:
//   start[v], end[v], step[v]            (3 doubles per variable)              → 10*3 = 30

static inline void die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}

int main(void) {
  // --- Read inputs (small, done once; kept separate from the hot loop) ---
  double a[120];  // coefficients/targets/tolerances
  double b[30];   // grid ranges: start,end,step for x1..x10

  {
    FILE* fp = fopen("./disp.txt", "r");
    if (!fp) die("Error: could not open disp.txt");
    for (int i = 0; i < 120; ++i) {
      if (fscanf(fp, "%lf", &a[i]) != 1) die("Error: fscanf failed (disp.txt)");
    }
    fclose(fp);
  }
  {
    FILE* fp = fopen("./grid.txt", "r");
    if (!fp) die("Error: could not open grid.txt");
    for (int i = 0; i < 30; ++i) {
      if (fscanf(fp, "%lf", &b[i]) != 1) die("Error: fscanf failed (grid.txt)");
    }
    fclose(fp);
  }

  // --- Parameters / constants that stay live in registers ---
  const double kk = 0.3;

  // Map a[] into C[10][10], D[10], EY[10]
  double C[10][10], D[10], EY[10], E[10];
  {
    int p = 0;
    for (int t = 0; t < 10; ++t) {
      for (int j = 0; j < 10; ++j) C[t][j] = a[p++];  // c_{t+1, j+1}
      D[t]  = a[p++];                                 // d_{t+1}
      EY[t] = a[p++];                                 // ey_{t+1}
    }
    for (int t = 0; t < 10; ++t) E[t] = kk * EY[t];   // tolerances e1..e10  (LICM)
  }

  // Unpack b[] into starts/ends/steps (x1..x10)
  double Xstart[10], Xend[10], Xstep[10];
  {
    int q = 0;
    for (int v = 0; v < 10; ++v) {
      Xstart[v] = b[q++];
      Xend[v]   = b[q++];
      Xstep[v]  = b[q++];
    }
  }

  // Precompute loop counts si = floor((end - start)/step)
  // NOTE: matches the original code’s semantics exactly.
  int S[10];
  for (int v = 0; v < 10; ++v) {
    const double span = Xend[v] - Xstart[v];
    S[v] = (int)floor(span / Xstep[v]);
    if (S[v] < 0) S[v] = 0; // guard against bad inputs
  }

  // --- Timing starts here (compute-only region) ---
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

  long pnts = 0;  // number of feasible points

  // We’ll keep hierarchical partial sums arrays ready for next step.
  // For this “baseline-inlined” version we still compute full q’s, but
  // moving data into arrays already cuts param traffic and enables LICM.
  double x[10];

  for (int r0 = 0; r0 < S[0]; ++r0) {
    x[0] = Xstart[0] + r0 * Xstep[0];

    for (int r1 = 0; r1 < S[1]; ++r1) {
      x[1] = Xstart[1] + r1 * Xstep[1];

      for (int r2 = 0; r2 < S[2]; ++r2) {
        x[2] = Xstart[2] + r2 * Xstep[2];

        for (int r3 = 0; r3 < S[3]; ++r3) {
          x[3] = Xstart[3] + r3 * Xstep[3];

          for (int r4 = 0; r4 < S[4]; ++r4) {
            x[4] = Xstart[4] + r4 * Xstep[4];

            for (int r5 = 0; r5 < S[5]; ++r5) {
              x[5] = Xstart[5] + r5 * Xstep[5];

              for (int r6 = 0; r6 < S[6]; ++r6) {
                x[6] = Xstart[6] + r6 * Xstep[6];

                for (int r7 = 0; r7 < S[7]; ++r7) {
                  x[7] = Xstart[7] + r7 * Xstep[7];

                  for (int r8 = 0; r8 < S[8]; ++r8) {
                    x[8] = Xstart[8] + r8 * Xstep[8];

                    for (int r9 = 0; r9 < S[9]; ++r9) {
                      x[9] = Xstart[9] + r9 * Xstep[9];

                      // ---- innermost: compute q1..q10 and early-exit ----
                      // Inline fabs via conditional to avoid function-call overhead.
                      int ok = 1;  // optimistic; invalidate on first failure

                      // Compute constraints one by one, fail-fast:
                      // (Order matters for pruning: we can reorder later by “most likely to fail”.)
                      for (int t = 0; t < 10; ++t) {
                        double acc = -D[t];
                        // dot(C[t][:], x[:])
                        // (Next version: replace this with incremental partial sums.)
                        acc += C[t][0]*x[0] + C[t][1]*x[1] + C[t][2]*x[2] + C[t][3]*x[3]
                             + C[t][4]*x[4] + C[t][5]*x[5] + C[t][6]*x[6] + C[t][7]*x[7]
                             + C[t][8]*x[8] + C[t][9]*x[9];

                        const double q = acc < 0.0 ? -acc : acc;  // fabs(acc)
                        if (q > E[t]) { ok = 0; break; }          // early exit
                      }

                      if (ok) {
                        // Count only; we’ll add output buffering at the end if needed.
                        ++pnts;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
  const double sec = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) / NSEC_SEC_MUL;

  printf("Feasible points: %ld\n", pnts);
  printf("Compute-only time: %.6f s\n", sec);

  return 0;
}
