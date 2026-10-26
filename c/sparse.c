/*
 * Sparse x sparse matrix multiplication in CSR format (row-wise Gustavson
 * algorithm), written for this study; no sparse library is used.
 *
 * For each row i of A, every stored A[i][k] scales the stored row k of B and
 * the products are accumulated in a dense work array indexed by column, with a
 * marker array recording which columns of row i have been touched:
 *   pass 1 (symbolic): count the distinct columns of every row of C -> row_ptr
 *   pass 2 (numeric):  accumulate the values and copy them into C
 * Work is proportional to the number of products sum_k nnz(A[:,k]) nnz(B[k,:]),
 * not to n^3. The columns of a row of C are stored in first-touch order
 * (not sorted). Each C[i][j] is accumulated in increasing k order, like the
 * dense kernels, so the values equal the dense results.
 *
 * Usage:
 *   sparse selftest
 *   sparse bench --n N --a FILE.csr --b FILE.csr [--input TAG] [--warmup W] [--reps R]
 *                [--min-batch-time T] [--dump FILE.csr] [--conversions 1]
 *
 * Bench output (stdout, CSV): same columns as dense.c,
 *   method,input,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum,nnz_c
 * phase = warmup | measure (CSR x CSR -> CSR), and with --conversions 1 also
 *   to_csr   (dense A and B -> CSR, both matrices) and to_dense (C -> dense),
 * which are timed separately and never added to the multiplication time.
 * A comment line "# key,value,..." reports nnz counts and CSR storage sizes.
 */
#include "common.h"

/* ---------- kernel: the timed region is exactly this function ---------- */
static csr_t *spgemm(const csr_t *A, const csr_t *B) {
    const int n = A->n;
    int32_t *mark = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    double *acc = (double *)malloc((size_t)n * sizeof(double));
    int64_t *row_ptr = (int64_t *)malloc(((size_t)n + 1) * sizeof(int64_t));
    if (mark == NULL || acc == NULL || row_ptr == NULL) {
        free(mark);
        free(acc);
        free(row_ptr);
        return NULL;
    }

    /* Pass 1: symbolic, number of distinct columns in each row of C. */
    for (int j = 0; j < n; j++) mark[j] = -1;
    row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        int64_t count = 0;
        for (int64_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            const int k = A->col[p];
            for (int64_t q = B->row_ptr[k]; q < B->row_ptr[k + 1]; q++) {
                const int j = B->col[q];
                if (mark[j] != i) {
                    mark[j] = i;
                    count++;
                }
            }
        }
        row_ptr[i + 1] = row_ptr[i] + count;
    }

    const int64_t nnz = row_ptr[n];
    csr_t *C = (csr_t *)calloc(1, sizeof(csr_t));
    if (C == NULL) {
        free(mark);
        free(acc);
        free(row_ptr);
        return NULL;
    }
    C->n = n;
    C->nnz = nnz;
    C->row_ptr = row_ptr;
    C->col = (int32_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int32_t));
    C->val = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    if (C->col == NULL || C->val == NULL) {
        csr_free(C);
        free(mark);
        free(acc);
        return NULL;
    }

    /* Pass 2: numeric. The first product of C[i][j] initialises acc[j]. */
    for (int j = 0; j < n; j++) mark[j] = -1;
    for (int i = 0; i < n; i++) {
        int32_t *ci = C->col + row_ptr[i];
        int64_t len = 0;
        for (int64_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            const int k = A->col[p];
            const double a = A->val[p];
            for (int64_t q = B->row_ptr[k]; q < B->row_ptr[k + 1]; q++) {
                const int j = B->col[q];
                if (mark[j] != i) {
                    mark[j] = i;
                    ci[len++] = j;
                    acc[j] = a * B->val[q];
                } else {
                    acc[j] += a * B->val[q];
                }
            }
        }
        double *vi = C->val + row_ptr[i];
        for (int64_t t = 0; t < len; t++) vi[t] = acc[ci[t]];
    }
    free(mark);
    free(acc);
    return C;
}

/* ---------- self-test ---------- */
/* Multiplies two dense test matrices through CSR and returns the dense result. */
static double *via_csr(const double *A, const double *B, int n) {
    csr_t *As = dense_to_csr(A, n), *Bs = dense_to_csr(B, n);
    csr_t *Cs = spgemm(As, Bs);
    double *C = csr_to_dense(Cs);
    csr_free(As);
    csr_free(Bs);
    csr_free(Cs);
    return C;
}

static double *dense_ijk(const double *A, const double *B, int n) { /* Assignment 1 kernel */
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
    return C;
}

static uint64_t lcg_state = 88172645463325252ULL;
static double lcg_unit(void) { /* uniform in [0, 1) */
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 11) * (1.0 / 9007199254740992.0);
}

static int selftest(void) {
    int ok = 1;
    char name[96];

    double A3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double B3[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    double E3[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
    double *C3 = via_csr(A3, B3, 3);
    ok &= check("csr: known 3x3 product", equal_exact(C3, E3, 3));
    free(C3);

    double A4[16], I4[16], Z4[16];
    for (int i = 0; i < 16; i++) {
        A4[i] = (double)(i * 3 - 7);
        I4[i] = (i % 5 == 0) ? 1.0 : 0.0;
        Z4[i] = 0.0;
    }
    double *AI = via_csr(A4, I4, 4), *IA = via_csr(I4, A4, 4), *AZ = via_csr(A4, Z4, 4);
    ok &= check("csr: A * I == A", equal_exact(AI, A4, 4));
    ok &= check("csr: I * A == A", equal_exact(IA, A4, 4));
    ok &= check("csr: A * 0 == 0 (empty result)", equal_exact(AZ, Z4, 4));
    free(AI);
    free(IA);
    free(AZ);

    double A2[4] = {1, 2, 0, 1}, B2[4] = {1, 0, 3, 1}, E2[4] = {7, 2, 3, 1};
    double *C2 = via_csr(A2, B2, 2);
    ok &= check("csr: non-commuting 2x2", equal_exact(C2, E2, 2));
    free(C2);

    double a1 = 2.5, b1 = -4.0, e1 = -10.0;
    double *C1 = via_csr(&a1, &b1, 1);
    ok &= check("csr: 1x1", equal_exact(C1, &e1, 1));
    free(C1);

    /* Conversions round-trip and the nnz of a known structure. */
    csr_t *S = dense_to_csr(A2, 2);
    double *back = csr_to_dense(S);
    ok &= check("dense -> csr -> dense round trip", equal_exact(back, A2, 2) && S->nnz == 3);
    free(back);
    csr_free(S);

    /* Random sparse inputs (with empty rows): values equal to the dense baseline. */
    int sizes[3] = {17, 64, 150};
    double dens[3] = {0.02, 0.2, 0.9};
    for (int t = 0; t < 3; t++) {
        for (int d = 0; d < 3; d++) {
            int n = sizes[t];
            double *A = (double *)calloc((size_t)n * n, sizeof(double));
            double *B = (double *)calloc((size_t)n * n, sizeof(double));
            for (int idx = 0; idx < n * n; idx++) {
                if (lcg_unit() < dens[d] && idx / n != 3) A[idx] = 2.0 * lcg_unit() - 1.0;
                if (lcg_unit() < dens[d]) B[idx] = 2.0 * lcg_unit() - 1.0;
            }
            double *R = dense_ijk(A, B, n), *C = via_csr(A, B, n);
            snprintf(name, sizeof name, "csr == dense ijk values, n=%d density=%.2f", n, dens[d]);
            ok &= check(name, equal_exact(C, R, n));
            free(A);
            free(B);
            free(R);
            free(C);
        }
    }

    printf("selftest %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

/* ---------- benchmark ---------- */
/* Time `inner` repetitions of a conversion; returns the batch time. */
static double time_to_csr(const double *Ad, const double *Bd, int n, int inner, double *chk) {
    double t0 = now_seconds();
    for (int b = 0; b < inner; b++) {
        csr_t *As = dense_to_csr(Ad, n), *Bs = dense_to_csr(Bd, n);
        sink = As->val[0] + Bs->val[0];
        *chk = (double)(As->nnz + Bs->nnz);
        csr_free(As);
        csr_free(Bs);
    }
    return now_seconds() - t0;
}

static double time_to_dense(const csr_t *C, int inner, double *chk) {
    int n = C->n;
    double t0 = now_seconds();
    for (int b = 0; b < inner; b++) {
        double *D = csr_to_dense(C);
        sink = D[(size_t)b % ((size_t)n * n)];
        *chk = (double)n;
        free(D);
    }
    return now_seconds() - t0;
}

static int bench(int n, const char *path_a, const char *path_b, const char *input, int warmup,
                 int reps, double min_batch_time, const char *dump_path, int conversions) {
    csr_t *A = read_csr(path_a);
    csr_t *B = read_csr(path_b);
    if (A == NULL || B == NULL) return 2;
    if (A->n != n || B->n != n) {
        fprintf(stderr, "input size mismatch\n");
        return 2;
    }

    printf("method,input,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum,nnz_c\n");
    fflush(stdout);

    double last_single = 0.0;
    for (int w = 0; w < warmup; w++) {
        double t0 = now_seconds();
        csr_t *C = spgemm(A, B);
        double t1 = now_seconds();
        if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
        last_single = t1 - t0;
        printf("csr,%s,%d,warmup,%d,1,%.9e,%.9e,%.17g,%lld\n", input, n, w, last_single,
               last_single, checksum_csr(C), (long long)C->nnz);
        fflush(stdout);
        csr_free(C);
    }

    int inner = 1;
    if (last_single > 0.0 && last_single < min_batch_time)
        inner = (int)ceil(min_batch_time / last_single);

    csr_t *C = NULL;
    for (int r = 0; r < reps; r++) {
        double t0 = now_seconds();
        for (int b = 0; b < inner; b++) {
            if (C != NULL) csr_free(C);
            C = spgemm(A, B);
            if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
            sink = C->nnz > 0 ? C->val[(size_t)b % (size_t)C->nnz] : 0.0;
        }
        double t1 = now_seconds();
        double batch = t1 - t0;
        printf("csr,%s,%d,measure,%d,%d,%.9e,%.9e,%.17g,%lld\n", input, n, r, inner, batch,
               batch / inner, checksum_csr(C), (long long)C->nnz);
        fflush(stdout);
    }
    printf("# nnz_a,%lld,nnz_b,%lld,nnz_c,%lld,csr_bytes_a,%.0f,csr_bytes_b,%.0f,csr_bytes_c,%.0f\n",
           (long long)A->nnz, (long long)B->nnz, (long long)C->nnz, csr_bytes(A), csr_bytes(B),
           csr_bytes(C));
    fflush(stdout);

    if (dump_path != NULL && !write_csr(dump_path, C)) {
        fprintf(stderr, "cannot write %s\n", dump_path);
        return 4;
    }

    /* Conversion costs, measured separately (not part of the multiplication time). */
    if (conversions) {
        double *Ad = csr_to_dense(A), *Bd = csr_to_dense(B);
        if (Ad == NULL || Bd == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
        double chk = 0.0;
        double t = time_to_csr(Ad, Bd, n, 1, &chk); /* warm-up and batch calibration */
        int in1 = t < min_batch_time ? (int)ceil(min_batch_time / (t > 0 ? t : 1e-9)) : 1;
        for (int r = 0; r < reps; r++) {
            double batch = time_to_csr(Ad, Bd, n, in1, &chk);
            printf("csr,%s,%d,to_csr,%d,%d,%.9e,%.9e,%.17g,%lld\n", input, n, r, in1, batch,
                   batch / in1, chk, (long long)C->nnz);
        }
        free(Ad);
        free(Bd);
        t = time_to_dense(C, 1, &chk);
        int in2 = t < min_batch_time ? (int)ceil(min_batch_time / (t > 0 ? t : 1e-9)) : 1;
        for (int r = 0; r < reps; r++) {
            double batch = time_to_dense(C, in2, &chk);
            printf("csr,%s,%d,to_dense,%d,%d,%.9e,%.9e,%.17g,%lld\n", input, n, r, in2, batch,
                   batch / in2, chk, (long long)C->nnz);
        }
        fflush(stdout);
    }
    csr_free(C);
    csr_free(A);
    csr_free(B);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return selftest();
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        int n = 0, warmup = 1, reps = 5, conversions = 0;
        double min_batch_time = 0.1;
        const char *path_a = NULL, *path_b = NULL, *input = "sparse", *dump = NULL;
        for (int i = 2; i + 1 < argc; i += 2) {
            if (strcmp(argv[i], "--n") == 0) n = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--a") == 0) path_a = argv[i + 1];
            else if (strcmp(argv[i], "--b") == 0) path_b = argv[i + 1];
            else if (strcmp(argv[i], "--input") == 0) input = argv[i + 1];
            else if (strcmp(argv[i], "--warmup") == 0) warmup = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--reps") == 0) reps = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--min-batch-time") == 0) min_batch_time = atof(argv[i + 1]);
            else if (strcmp(argv[i], "--dump") == 0) dump = argv[i + 1];
            else if (strcmp(argv[i], "--conversions") == 0) conversions = atoi(argv[i + 1]);
        }
        if (n <= 0 || path_a == NULL || path_b == NULL) {
            fprintf(stderr, "--n, --a and --b are required\n");
            return 1;
        }
        return bench(n, path_a, path_b, input, warmup, reps, min_batch_time, dump, conversions);
    }
    fprintf(stderr, "usage: sparse selftest | bench --n N --a FILE.csr --b FILE.csr [--input TAG] "
                    "[--warmup W] [--reps R] [--min-batch-time T] [--dump FILE] "
                    "[--conversions 1]\n");
    return 1;
}
