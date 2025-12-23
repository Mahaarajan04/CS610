// rollno-prob4-v5.c — Sequential Part (i) with loop permutation + tiling + partial-sum reuse
// Toggle file writes (off by default for speed):

#define DO_WRITE 1   // set to 1 to write results-v5.txt (slower, but same format as v0)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)
#define ABS(x) ((x) < 0.0 ? -(x) : (x))

static inline void die(const char *msg) { fprintf(stderr, "%s\n", msg); exit(EXIT_FAILURE); }
static inline int imin(int a, int b) { return a < b ? a : b; }

typedef struct {
    // Problem data
    double C[10][10], D[10], E[10];
    // Grid ranges
    double Xstart[10], Xstep[10];
    int    S[10];      // trip counts per original dim
    int    perm[10];   // permutation: perm[depth] = original_dim_index
    // Output control
#if DO_WRITE
    FILE *fout;
#endif
    long *pnts;
} Ctx;

// Heuristic tile sizes for the 3 hottest (innermost) permuted depths
#define TBLK2 16
#define TBLK1 16
#define TBLK0 16

// Depth-first recursive kernel with partial-sum reuse and optional tiling at depths 7,8,9.
static void dfs(Ctx *cx, int depth, double x[10], double partial[10]) {
    if (depth == 10) {
        // Check constraints and possibly write solution
        int ok = 1;
        for (int t = 0; t < 10; ++t) {
            double q = ABS(partial[t] - cx->D[t]);
            if (q > cx->E[t]) { ok = 0; break; }
        }
        if (ok) {
            ++(*cx->pnts);
#if DO_WRITE
            fprintf(cx->fout,
                "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]);
#endif
        }
        return;
    }

    const int dim = cx->perm[depth];          // which original dimension at this depth
    const int S   = cx->S[dim];               // trip count for this dim
    const double x0 = cx->Xstart[dim];
    const double dx = cx->Xstep[dim];

    // Choose whether to tile this depth (we tile depths 7,8,9: the hottest three)
    const int do_tile =
        (depth == 7) ? TBLK2 :
        (depth == 8) ? TBLK1 :
        (depth == 9) ? TBLK0 : 0;

    if (!do_tile) {
        for (int r = 0; r < S; ++r) {
            const double xv = x0 + r * dx;
            x[dim] = xv;

            // update partial sums once for this depth
            double next_partial[10];
            for (int t = 0; t < 10; ++t) {
                next_partial[t] = partial[t] + cx->C[t][dim] * xv;
            }
            dfs(cx, depth + 1, x, next_partial);
        }
    } else {
        for (int rb = 0; rb < S; rb += do_tile) {
            const int rend = imin(rb + do_tile, S);
            for (int r = rb; r < rend; ++r) {
                const double xv = x0 + r * dx;
                x[dim] = xv;

                double next_partial[10];
                for (int t = 0; t < 10; ++t) {
                    next_partial[t] = partial[t] + cx->C[t][dim] * xv;
                }
                dfs(cx, depth + 1, x, next_partial);
            }
        }
    }
}

int main(void) {
    // Read inputs
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

    // Unpack a[] → C, D, EY → E
    double C[10][10], D[10], EY[10], E[10];
    int p = 0;
    for (int t = 0; t < 10; ++t) {
        for (int j = 0; j < 10; ++j) C[t][j] = a[p++];
        D[t]  = a[p++];
        EY[t] = a[p++];
    }
    for (int t = 0; t < 10; ++t) E[t] = kk * EY[t];

    // Unpack b[] → starts/ends/steps, compute trip counts
    double Xstart[10], Xend[10], Xstep[10];
    int S[10];
    for (int v = 0, q = 0; v < 10; ++v) {
        Xstart[v] = b[q++];
        Xend[v]   = b[q++];
        Xstep[v]  = b[q++];
        double span = Xend[v] - Xstart[v];
        int s = (int)floor(span / Xstep[v]);
        S[v] = s < 0 ? 0 : s;
    }

    // Build a permutation: place larger S[] deeper (innermost has max S)
    int idx[10]; for (int i = 0; i < 10; ++i) idx[i] = i;
    // sort idx by S ascending → smaller trip counts outer, larger inner
    for (int i = 0; i < 9; ++i)
        for (int j = i + 1; j < 10; ++j)
            if (S[idx[i]] > S[idx[j]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    // Context setup
    long pnts = 0;
#if DO_WRITE
    FILE *fout = fopen("results-v5.txt", "w");
    if (!fout) die("Error opening results-v5.txt");
#endif
    Ctx cx = {0};
    memcpy(cx.C, C, sizeof(C));
    memcpy(cx.D, D, sizeof(D));
    memcpy(cx.E, E, sizeof(E));
    memcpy(cx.Xstart, Xstart, sizeof(Xstart));
    memcpy(cx.Xstep,  Xstep,  sizeof(Xstep));
    memcpy(cx.S,      S,      sizeof(S));
    memcpy(cx.perm,   idx,    sizeof(idx));
#if DO_WRITE
    cx.fout = fout;
#endif
    cx.pnts = &pnts;

    // Time and launch
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    double x[10] = {0};
    double partial0[10] = {0};   // starts at 0, accumulates C[t][dim]*x_dim
    dfs(&cx, 0, x, partial0);

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / NSEC_SEC_MUL;

#if DO_WRITE
    fclose(fout);
#endif

    printf("result pnts: %ld\n", pnts);
    printf("Total time = %f seconds\n", sec);
    // Optional: print the chosen permutation for your report
    // fprintf(stderr, "perm (outer→inner): "); for(int d=0; d<10; ++d) fprintf(stderr, "%d ", cx.perm[d]+1); fprintf(stderr, "\n");

    return 0;
}
