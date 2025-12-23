// rollno-prob4-v3.c — Fully optimized sequential kernel for Problem 4 Part (i)
// Compile: gcc -std=c17 -O3 -march=native -ffast-math -fopenmp \
//          -funroll-loops -fno-asynchronous-unwind-tables -fno-exceptions \
//          -fcf-protection=none -Wall -Wextra rollno-prob4-v3.c -o rollno-prob4-v3.out -pthread

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)
#define ABS(x) ((x) < 0.0 ? -(x) : (x))

static inline void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

int main(void) {
    double a[120], b[30];
    FILE *fp = fopen("./disp.txt", "r");
    if (!fp) die("Error: could not open disp.txt");
    for (int i = 0; i < 120; ++i)
        if (fscanf(fp, "%lf", &a[i]) != 1) die("Error reading disp.txt");
    fclose(fp);

    fp = fopen("./grid.txt", "r");
    if (!fp) die("Error: could not open grid.txt");
    for (int i = 0; i < 30; ++i)
        if (fscanf(fp, "%lf", &b[i]) != 1) die("Error reading grid.txt");
    fclose(fp);

    const double kk = 0.3;

    double C[10][10], D[10], EY[10], E[10];
    int p = 0;
    for (int t = 0; t < 10; ++t) {
        for (int j = 0; j < 10; ++j)
            C[t][j] = a[p++];
        D[t] = a[p++];
        EY[t] = a[p++];
    }
    for (int t = 0; t < 10; ++t)
        E[t] = kk * EY[t];

    double Xstart[10], Xend[10], Xstep[10];
    for (int v = 0, q = 0; v < 10; ++v) {
        Xstart[v] = b[q++];
        Xend[v]   = b[q++];
        Xstep[v]  = b[q++];
    }
    int S[10];
    for (int v = 0; v < 10; ++v) {
        double span = Xend[v] - Xstart[v];
        S[v] = (int)floor(span / Xstep[v]);
        if (S[v] < 0) S[v] = 0;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    long pnts = 0;
    double x[10];
    double partial[10][10]; // partial[t][depth]

    // initialize partials
    for (int t = 0; t < 10; ++t)
        for (int d = 0; d < 10; ++d)
            partial[t][d] = 0.0;

    // reorder constraints by increasing tolerance for better early exit
    int order[10];
    for (int t = 0; t < 10; ++t) order[t] = t;
    for (int i = 0; i < 9; ++i)
        for (int j = i + 1; j < 10; ++j)
            if (E[order[j]] < E[order[i]]) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }

    // ===== Nested loops with partial-sum reuse and unrolled constraint check =====
    for (int r0 = 0; r0 < S[0]; ++r0) {
        x[0] = Xstart[0] + r0 * Xstep[0];
        for (int t = 0; t < 10; ++t)
            partial[t][0] = C[t][0] * x[0];

        for (int r1 = 0; r1 < S[1]; ++r1) {
            x[1] = Xstart[1] + r1 * Xstep[1];
            for (int t = 0; t < 10; ++t)
                partial[t][1] = partial[t][0] + C[t][1] * x[1];

            for (int r2 = 0; r2 < S[2]; ++r2) {
                x[2] = Xstart[2] + r2 * Xstep[2];
                for (int t = 0; t < 10; ++t)
                    partial[t][2] = partial[t][1] + C[t][2] * x[2];

                for (int r3 = 0; r3 < S[3]; ++r3) {
                    x[3] = Xstart[3] + r3 * Xstep[3];
                    for (int t = 0; t < 10; ++t)
                        partial[t][3] = partial[t][2] + C[t][3] * x[3];

                    for (int r4 = 0; r4 < S[4]; ++r4) {
                        x[4] = Xstart[4] + r4 * Xstep[4];
                        for (int t = 0; t < 10; ++t)
                            partial[t][4] = partial[t][3] + C[t][4] * x[4];

                        for (int r5 = 0; r5 < S[5]; ++r5) {
                            x[5] = Xstart[5] + r5 * Xstep[5];
                            for (int t = 0; t < 10; ++t)
                                partial[t][5] = partial[t][4] + C[t][5] * x[5];

                            for (int r6 = 0; r6 < S[6]; ++r6) {
                                x[6] = Xstart[6] + r6 * Xstep[6];
                                for (int t = 0; t < 10; ++t)
                                    partial[t][6] = partial[t][5] + C[t][6] * x[6];

                                for (int r7 = 0; r7 < S[7]; ++r7) {
                                    x[7] = Xstart[7] + r7 * Xstep[7];
                                    for (int t = 0; t < 10; ++t)
                                        partial[t][7] = partial[t][6] + C[t][7] * x[7];

                                    for (int r8 = 0; r8 < S[8]; ++r8) {
                                        x[8] = Xstart[8] + r8 * Xstep[8];
                                        for (int t = 0; t < 10; ++t)
                                            partial[t][8] = partial[t][7] + C[t][8] * x[8];

                                        for (int r9 = 0; r9 < S[9]; ++r9) {
                                            x[9] = Xstart[9] + r9 * Xstep[9];
                                            int ok = 1;

                                            // manually unrolled constraint loop for ILP
                                            double q0, q1, q2, q3, q4, q5, q6, q7, q8, q9;
                                            {
                                                const int t0 = order[0];
                                                double sum0 = partial[t0][8] + C[t0][9]*x[9];
                                                q0 = fabs(sum0 - D[t0]);
                                                if (q0 > E[t0]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t1 = order[1];
                                                double sum1 = partial[t1][8] + C[t1][9]*x[9];
                                                q1 = fabs(sum1 - D[t1]);
                                                if (q1 > E[t1]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t2 = order[2];
                                                double sum2 = partial[t2][8] + C[t2][9]*x[9];
                                                q2 = fabs(sum2 - D[t2]);
                                                if (q2 > E[t2]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t3 = order[3];
                                                double sum3 = partial[t3][8] + C[t3][9]*x[9];
                                                q3 = fabs(sum3 - D[t3]);
                                                if (q3 > E[t3]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t4 = order[4];
                                                double sum4 = partial[t4][8] + C[t4][9]*x[9];
                                                q4 = fabs(sum4 - D[t4]);
                                                if (q4 > E[t4]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t5 = order[5];
                                                double sum5 = partial[t5][8] + C[t5][9]*x[9];
                                                q5 = fabs(sum5 - D[t5]);
                                                if (q5 > E[t5]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t6 = order[6];
                                                double sum6 = partial[t6][8] + C[t6][9]*x[9];
                                                q6 = fabs(sum6 - D[t6]);
                                                if (q6 > E[t6]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t7 = order[7];
                                                double sum7 = partial[t7][8] + C[t7][9]*x[9];
                                                q7 = fabs(sum7 - D[t7]);
                                                if (q7 > E[t7]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t8 = order[8];
                                                double sum8 = partial[t8][8] + C[t8][9]*x[9];
                                                q8 = fabs(sum8 - D[t8]);
                                                if (q8 > E[t8]) ok = 0;
                                            }
                                            if (ok) {
                                                const int t9 = order[9];
                                                double sum9 = partial[t9][8] + C[t9][9]*x[9];
                                                q9 = fabs(sum9 - D[t9]);
                                                if (q9 > E[t9]) ok = 0;
                                            }

                                            if (ok) ++pnts;
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
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / NSEC_SEC_MUL;
    printf("Feasible points: %ld\n", pnts);
    printf("Optimized compute time: %.6f s\n", sec);

    return 0;
}
