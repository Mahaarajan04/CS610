// rollno-prob4-v2-omp.c — OpenMP version with partial-sum reuse and buffered writes (results-omp.txt)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

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

    FILE *fptr = fopen("./results-omp.txt", "w");
    if (fptr == NULL) {
        printf("Error in creating results-omp.txt!\n");
        exit(1);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    long pnts = 0;

#pragma omp parallel
    {
        double x[10];
        double partial[10][10];
        long local_pnts = 0;
        char local_buf[1 << 16];
        int buf_off = 0;

        for (int t = 0; t < 10; ++t)
            for (int d = 0; d < 10; ++d)
                partial[t][d] = 0.0;

#pragma omp for schedule(dynamic) reduction(+:pnts)
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

                                                for (int t = 0; t < 10; ++t) {
                                                    double sum = partial[t][8] + C[t][9]*x[9];
                                                    double q = ABS(sum - D[t]);
                                                    if (q > E[t]) { ok = 0; break; }
                                                }

                                                if (ok) {
                                                    local_pnts++;
                                                    buf_off += snprintf(local_buf + buf_off,
                                                                        sizeof(local_buf) - buf_off,
                                                                        "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                                                                        x[0],x[1],x[2],x[3],x[4],
                                                                        x[5],x[6],x[7],x[8],x[9]);
                                                    if (buf_off > (int)(sizeof(local_buf) - 512)) {
#pragma omp critical
                                                        fwrite(local_buf, 1, buf_off, fptr);
                                                        buf_off = 0;
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
        }

#pragma omp critical
        {
            if (buf_off > 0)
                fwrite(local_buf, 1, buf_off, fptr);
            pnts += local_pnts;
        }
    }

    fclose(fptr);

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) +
                 (t1.tv_nsec - t0.tv_nsec) / NSEC_SEC_MUL;

    printf("result pnts: %ld\n", pnts);
    printf("Total time = %f seconds\n", sec);

    return 0;
}
