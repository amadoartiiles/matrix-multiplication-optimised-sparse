/*
 * Shared helpers for dense.c and sparse.c: monotonic timer, matrix file I/O
 * (dense .bin and CSR .csr), checksums and the self-test reporter.
 *
 * File formats
 *   .bin  n*n little-endian IEEE 754 binary64 values, row-major (Assignment 1 format).
 *   .csr  header int64[3] = {magic 0x31525343 ("CSR1"), n, nnz}, then
 *         row_ptr int64[n+1], col_idx int32[nnz], values float64[nnz];
 *         column indices are sorted within each row in the input files.
 */
#ifndef MM_COMMON_H
#define MM_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define CSR_MAGIC 0x31525343LL

/* Results are written here so the compiler cannot discard batched products. */
static volatile double sink;

typedef struct {
    int n;
    int64_t nnz;
    int64_t *row_ptr; /* n + 1 entries */
    int32_t *col;     /* nnz entries */
    double *val;      /* nnz entries */
} csr_t;

/* ---------- monotonic timer ---------- */
static inline double now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static inline int ends_with(const char *s, const char *suffix) {
    size_t ls = strlen(s), lx = strlen(suffix);
    return ls >= lx && strcmp(s + ls - lx, suffix) == 0;
}

/* ---------- CSR helpers ---------- */
static inline void csr_free(csr_t *M) {
    if (M == NULL) return;
    free(M->row_ptr);
    free(M->col);
    free(M->val);
    free(M);
}

static inline csr_t *csr_alloc(int n, int64_t nnz) {
    csr_t *M = (csr_t *)calloc(1, sizeof(csr_t));
    if (M == NULL) return NULL;
    M->n = n;
    M->nnz = nnz;
    M->row_ptr = (int64_t *)malloc(((size_t)n + 1) * sizeof(int64_t));
    M->col = (int32_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int32_t));
    M->val = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    if (M->row_ptr == NULL || M->col == NULL || M->val == NULL) {
        csr_free(M);
        return NULL;
    }
    return M;
}

/* Bytes of the three CSR arrays (the "storage model" of a CSR matrix). */
static inline double csr_bytes(const csr_t *M) {
    return 8.0 * ((double)M->n + 1) + 12.0 * (double)M->nnz;
}

/* Dense -> CSR: two passes over the n*n entries (count, then fill). */
static inline csr_t *dense_to_csr(const double *D, int n) {
    int64_t nnz = 0;
    for (size_t idx = 0; idx < (size_t)n * n; idx++)
        if (D[idx] != 0.0) nnz++;
    csr_t *M = csr_alloc(n, nnz);
    if (M == NULL) return NULL;
    int64_t p = 0;
    M->row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        const double *row = D + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            if (row[j] != 0.0) {
                M->col[p] = j;
                M->val[p] = row[j];
                p++;
            }
        }
        M->row_ptr[i + 1] = p;
    }
    return M;
}

/* CSR -> dense (zero-filled n*n buffer). */
static inline double *csr_to_dense(const csr_t *M) {
    int n = M->n;
    double *D = (double *)calloc((size_t)n * n, sizeof(double));
    if (D == NULL) return NULL;
    for (int i = 0; i < n; i++)
        for (int64_t p = M->row_ptr[i]; p < M->row_ptr[i + 1]; p++)
            D[(size_t)i * n + M->col[p]] = M->val[p];
    return D;
}

/* ---------- file I/O ---------- */
static inline csr_t *read_csr(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    int64_t hdr[3];
    if (fread(hdr, sizeof(int64_t), 3, f) != 3 || hdr[0] != CSR_MAGIC) {
        fprintf(stderr, "%s is not a CSR1 file\n", path);
        fclose(f);
        return NULL;
    }
    csr_t *M = csr_alloc((int)hdr[1], hdr[2]);
    if (M == NULL) {
        fclose(f);
        return NULL;
    }
    size_t n1 = (size_t)M->n + 1, nnz = (size_t)M->nnz;
    if (fread(M->row_ptr, sizeof(int64_t), n1, f) != n1 ||
        fread(M->col, sizeof(int32_t), nnz, f) != nnz ||
        fread(M->val, sizeof(double), nnz, f) != nnz) {
        fprintf(stderr, "truncated CSR file %s\n", path);
        csr_free(M);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return M;
}

static inline int write_csr(const char *path, const csr_t *M) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return 0;
    int64_t hdr[3] = {CSR_MAGIC, M->n, M->nnz};
    size_t n1 = (size_t)M->n + 1, nnz = (size_t)M->nnz;
    int ok = fwrite(hdr, sizeof(int64_t), 3, f) == 3 &&
             fwrite(M->row_ptr, sizeof(int64_t), n1, f) == n1 &&
             fwrite(M->col, sizeof(int32_t), nnz, f) == nnz &&
             fwrite(M->val, sizeof(double), nnz, f) == nnz;
    fclose(f);
    return ok;
}

static inline double *read_dense(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    size_t count = (size_t)n * n;
    double *M = (double *)malloc(count * sizeof(double));
    if (M == NULL || fread(M, sizeof(double), count, f) != count) {
        fprintf(stderr, "cannot read %zu doubles from %s\n", count, path);
        fclose(f);
        free(M);
        return NULL;
    }
    fclose(f);
    return M;
}

static inline int write_dense(const char *path, const double *M, int n) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return 0;
    size_t count = (size_t)n * n;
    size_t written = fwrite(M, sizeof(double), count, f);
    fclose(f);
    return written == count;
}

/* Reads a matrix as dense, whatever its file format (.csr is expanded). */
static inline double *load_as_dense(const char *path, int n) {
    if (!ends_with(path, ".csr")) return read_dense(path, n);
    csr_t *M = read_csr(path);
    if (M == NULL) return NULL;
    if (M->n != n) {
        fprintf(stderr, "%s has n=%d, expected %d\n", path, M->n, n);
        csr_free(M);
        return NULL;
    }
    double *D = csr_to_dense(M);
    csr_free(M);
    return D;
}

/* ---------- checksums ---------- */
/* Row-major sequential sum of a dense matrix, used to consume the result. */
static inline double checksum_dense(const double *C, int n) {
    double s = 0.0;
    for (size_t idx = 0; idx < (size_t)n * n; idx++) s += C[idx];
    return s;
}

static inline int64_t nnz_dense(const double *C, int n) {
    int64_t c = 0;
    for (size_t idx = 0; idx < (size_t)n * n; idx++)
        if (C[idx] != 0.0) c++;
    return c;
}

/* Sum of the stored values of a CSR matrix (its order differs from the dense sum). */
static inline double checksum_csr(const csr_t *C) {
    double s = 0.0;
    for (int64_t p = 0; p < C->nnz; p++) s += C->val[p];
    return s;
}

/* ---------- self-test reporting ---------- */
static inline int equal_exact(const double *X, const double *Y, int n) {
    for (size_t idx = 0; idx < (size_t)n * n; idx++)
        if (X[idx] != Y[idx]) return 0;
    return 1;
}

static inline int check(const char *name, int ok) {
    printf("%-44s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

#endif
