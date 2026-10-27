/*
 * Dense matrix multiplication: the Assignment 1 baseline and three sequential
 * optimisations, all on contiguous row-major IEEE 754 binary64 buffers.
 *
 *   ijk         baseline, the validated Assignment 1 kernel (unchanged)
 *   ikj         loop reordering: the innermost loop walks rows of B and C
 *   transposed  copies B into B^T, then i-j-k with both operands read by rows
 *   blocked     cache blocking (tiling) of the i-k-j loop nest with tile size bs
 *
 * Every kernel allocates and zero-initialises C and accumulates each C[i][j]
 * in increasing k order starting from 0.0, so all four produce bitwise
 * identical results (no -ffast-math, no FMA contraction without -march).
 *
 * Usage:
 *   dense selftest
 *   dense bench --method M --n N --a FILE --b FILE [--input TAG] [--block BS]
 *               [--warmup W] [--reps R] [--min-batch-time T] [--dump FILE]
 * FILE may be a dense .bin or a .csr file (expanded to dense before timing).
 *
 * Bench output (stdout, CSV, one row per warm-up or measured repetition):
 *   method,input,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum,nnz_c
 */
#include "common.h"

#define DEFAULT_BLOCK 64

/* ---------- kernels: the timed region is exactly one of these calls ---------- */

/* Baseline: Assignment 1 kernel (multiply), body copied without changes. */
static double *mm_ijk(const double *A, const double *B, int n) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            }
            C[(size_t)i * n + j] = s;
        }
    }
    return C;
}

/* Loop reordering: i-k-j. A[i][k] is hoisted; B and C are accessed with unit stride. */
static double *mm_ikj(const double *A, const double *B, int n) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
    for (int i = 0; i < n; i++) {
        double *Ci = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            const double a = A[(size_t)i * n + k];
            const double *Bk = B + (size_t)k * n;
            for (int j = 0; j < n; j++) {
                Ci[j] += a * Bk[j];
            }
        }
    }
    return C;
}

/* Data-layout change: transpose B (tiled copy, included in the timed region),
 * then the i-j-k dot products read rows of A and rows of B^T. */
static double *mm_transposed(const double *A, const double *B, int n) {
    const int T = 32;
    double *Bt = (double *)malloc((size_t)n * n * sizeof(double));
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (Bt == NULL || C == NULL) {
        free(Bt);
        free(C);
        return NULL;
    }
    for (int kk = 0; kk < n; kk += T)
        for (int jj = 0; jj < n; jj += T)
            for (int k = kk; k < kk + T && k < n; k++)
                for (int j = jj; j < jj + T && j < n; j++)
                    Bt[(size_t)j * n + k] = B[(size_t)k * n + j];
    for (int i = 0; i < n; i++) {
        const double *Ai = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *Btj = Bt + (size_t)j * n;
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += Ai[k] * Btj[k];
            }
            C[(size_t)i * n + j] = s;
        }
    }
    free(Bt);
    return C;
}

/* Cache blocking: i-k-j inside bs x bs tiles. For a fixed C[i][j], the kk tiles
 * and the k values inside them are visited in increasing order. */
static double *mm_blocked(const double *A, const double *B, int n, int bs) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
    for (int ii = 0; ii < n; ii += bs) {
        const int iend = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            const int kend = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                const int jend = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < iend; i++) {
                    double *Ci = C + (size_t)i * n;
                    for (int k = kk; k < kend; k++) {
                        const double a = A[(size_t)i * n + k];
                        const double *Bk = B + (size_t)k * n;
                        for (int j = jj; j < jend; j++) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
    return C;
}

enum { M_IJK, M_IKJ, M_TRANSPOSED, M_BLOCKED, M_COUNT };
static const char *METHOD_NAME[M_COUNT] = {"ijk", "ikj", "transposed", "blocked"};

static double *multiply(int method, const double *A, const double *B, int n, int bs) {
    switch (method) {
        case M_IJK: return mm_ijk(A, B, n);
        case M_IKJ: return mm_ikj(A, B, n);
        case M_TRANSPOSED: return mm_transposed(A, B, n);
        case M_BLOCKED: return mm_blocked(A, B, n, bs);
    }
    return NULL;
}

static int method_from_name(const char *name) {
    for (int m = 0; m < M_COUNT; m++)
        if (strcmp(name, METHOD_NAME[m]) == 0) return m;
    return -1;
}

/* ---------- self-test with exact known answers ---------- */
static uint64_t lcg_state = 88172645463325252ULL;
static double lcg_uniform(void) { /* uniform in [-1, 1), deterministic */
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

static int selftest(void) {
    int ok = 1;
    char name[96];
    int block_sizes[3] = {DEFAULT_BLOCK, 2, 16};

    for (int m = 0; m < M_COUNT; m++) {
        /* Same known-answer cases as Assignment 1, for every kernel. */
        double A3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        double B3[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
        double E3[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
        double *C3 = multiply(m, A3, B3, 3, 2);
        snprintf(name, sizeof name, "%s: known 3x3 product", METHOD_NAME[m]);
        ok &= check(name, equal_exact(C3, E3, 3));
        free(C3);

        double A4[16], I4[16], Z4[16];
        for (int i = 0; i < 16; i++) {
            A4[i] = (double)(i * 3 - 7);
            I4[i] = (i % 5 == 0) ? 1.0 : 0.0;
            Z4[i] = 0.0;
        }
        double *AI = multiply(m, A4, I4, 4, 3);
        double *IA = multiply(m, I4, A4, 4, 3);
        double *AZ = multiply(m, A4, Z4, 4, 3);
        snprintf(name, sizeof name, "%s: A * I == A", METHOD_NAME[m]);
        ok &= check(name, equal_exact(AI, A4, 4));
        snprintf(name, sizeof name, "%s: I * A == A", METHOD_NAME[m]);
        ok &= check(name, equal_exact(IA, A4, 4));
        snprintf(name, sizeof name, "%s: A * 0 == 0", METHOD_NAME[m]);
        ok &= check(name, equal_exact(AZ, Z4, 4));
        free(AI);
        free(IA);
        free(AZ);

        double A2[4] = {1, 2, 0, 1}, B2[4] = {1, 0, 3, 1}, E2[4] = {7, 2, 3, 1};
        double *C2 = multiply(m, A2, B2, 2, 1);
        snprintf(name, sizeof name, "%s: non-commuting 2x2", METHOD_NAME[m]);
        ok &= check(name, equal_exact(C2, E2, 2));
        free(C2);

        double a1 = 2.5, b1 = -4.0, e1 = -10.0;
        double *C1 = multiply(m, &a1, &b1, 1, DEFAULT_BLOCK);
        snprintf(name, sizeof name, "%s: 1x1", METHOD_NAME[m]);
        ok &= check(name, equal_exact(C1, &e1, 1));
        free(C1);
    }

    /* Random inputs, sizes that are not multiples of the tile sizes:
     * every optimised kernel must be bitwise identical to the baseline. */
    int sizes[5] = {7, 33, 100, 130, 257};
    for (int t = 0; t < 5; t++) {
        int n = sizes[t];
        double *A = (double *)malloc((size_t)n * n * sizeof(double));
        double *B = (double *)malloc((size_t)n * n * sizeof(double));
        for (int idx = 0; idx < n * n; idx++) A[idx] = lcg_uniform();
        for (int idx = 0; idx < n * n; idx++) B[idx] = lcg_uniform();
        double *R = mm_ijk(A, B, n);
        for (int m = 1; m < M_COUNT; m++) {
            for (int b = 0; b < (m == M_BLOCKED ? 3 : 1); b++) {
                double *C = multiply(m, A, B, n, block_sizes[b]);
                if (m == M_BLOCKED)
                    snprintf(name, sizeof name, "%s(bs=%d) == ijk bitwise, n=%d", METHOD_NAME[m],
                             block_sizes[b], n);
                else
                    snprintf(name, sizeof name, "%s == ijk bitwise, n=%d", METHOD_NAME[m], n);
                ok &= check(name, memcmp(C, R, (size_t)n * n * sizeof(double)) == 0);
                free(C);
            }
        }
        free(R);
        free(A);
        free(B);
    }

    printf("selftest %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

/* ---------- benchmark ---------- */
static int bench(int method, int n, const char *path_a, const char *path_b, const char *input,
                 int bs, int warmup, int reps, double min_batch_time, const char *dump_path) {
    double *A = load_as_dense(path_a, n);
    double *B = load_as_dense(path_b, n);
    if (A == NULL || B == NULL) return 2;
    const char *mname = METHOD_NAME[method];

    printf("method,input,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum,nnz_c\n");
    fflush(stdout);

    double last_single = 0.0;
    for (int w = 0; w < warmup; w++) {
        double t0 = now_seconds();
        double *C = multiply(method, A, B, n, bs);
        double t1 = now_seconds();
        if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
        last_single = t1 - t0;
        printf("%s,%s,%d,warmup,%d,1,%.9e,%.9e,%.17g,%lld\n", mname, input, n, w, last_single,
               last_single, checksum_dense(C, n), (long long)nnz_dense(C, n));
        fflush(stdout);
        free(C);
    }

    /* Batch very short multiplications so each sample lasts >= min_batch_time. */
    int inner = 1;
    if (last_single > 0.0 && last_single < min_batch_time)
        inner = (int)ceil(min_batch_time / last_single);

    double *C = NULL;
    for (int r = 0; r < reps; r++) {
        double t0 = now_seconds();
        for (int b = 0; b < inner; b++) {
            if (C != NULL) free(C);
            C = multiply(method, A, B, n, bs);
            if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
            sink = C[(size_t)b % ((size_t)n * n)]; /* every result is consumed */
        }
        double t1 = now_seconds();
        double batch = t1 - t0;
        printf("%s,%s,%d,measure,%d,%d,%.9e,%.9e,%.17g,%lld\n", mname, input, n, r, inner, batch,
               batch / inner, checksum_dense(C, n), (long long)nnz_dense(C, n));
        fflush(stdout);
    }

    if (dump_path != NULL && C != NULL && !write_dense(dump_path, C, n)) {
        fprintf(stderr, "cannot write %s\n", dump_path);
        return 4;
    }
    free(C);
    free(A);
    free(B);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return selftest();
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        int n = 0, warmup = 1, reps = 5, bs = DEFAULT_BLOCK, method = -1;
        double min_batch_time = 0.1;
        const char *path_a = NULL, *path_b = NULL, *input = "dense", *dump = NULL;
        for (int i = 2; i + 1 < argc; i += 2) {
            if (strcmp(argv[i], "--method") == 0) method = method_from_name(argv[i + 1]);
            else if (strcmp(argv[i], "--n") == 0) n = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--a") == 0) path_a = argv[i + 1];
            else if (strcmp(argv[i], "--b") == 0) path_b = argv[i + 1];
            else if (strcmp(argv[i], "--input") == 0) input = argv[i + 1];
            else if (strcmp(argv[i], "--block") == 0) bs = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--warmup") == 0) warmup = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--reps") == 0) reps = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--min-batch-time") == 0) min_batch_time = atof(argv[i + 1]);
            else if (strcmp(argv[i], "--dump") == 0) dump = argv[i + 1];
        }
        if (n <= 0 || method < 0 || path_a == NULL || path_b == NULL || bs <= 0) {
            fprintf(stderr, "--method, --n, --a and --b are required\n");
            return 1;
        }
        return bench(method, n, path_a, path_b, input, bs, warmup, reps, min_batch_time, dump);
    }
    fprintf(stderr, "usage: dense selftest | bench --method ijk|ikj|transposed|blocked --n N "
                    "--a FILE --b FILE [--input TAG] [--block BS] [--warmup W] [--reps R] "
                    "[--min-batch-time T] [--dump FILE]\n");
    return 1;
}
