"""Check every dumped result against a trusted reference (NumPy / SciPy), as in Assignment 1.

Tolerance: componentwise forward-error bound of a length-n dot product in binary64,
|C_hat - C| <= gamma_n (|A| |B|), gamma_n = n u / (1 - n u), u = 2^-53 [Higham 2002, 3.5].
Implementation and reference both satisfy it, so the check is

    |C_impl - C_ref| <= 2 gamma_n (|A| |B|)      (element-wise)

References: NumPy A @ B on the dense matrices (n <= 4096), or SciPy's CSR product
for larger sparse inputs. The libraries are used only for checking.

Additional checks:
  * dense experiment: every optimised kernel is bitwise identical to the ijk baseline;
  * sparse experiments: the CSR result, expanded to dense, has the same values as the
    dense kernels on the same input (value equality, so +0.0 == -0.0), and its nnz
    equals the reference nnz;
  * all measured repetitions of a configuration printed the same checksum.
Configurations without a dump (n above dump_max_n) are compared by nnz and checksum
with the SciPy product when it fits in memory.

    python scripts/validate_outputs.py --data DIR --outputs DIR [--results results]
"""
import argparse
import csv
import os

import numpy as np
import scipy.sparse as sp

U = 2.0 ** -53
CSR_MAGIC = 0x31525343


def gamma(n):
    return n * U / (1 - n * U)


def read_csr(path):
    with open(path, "rb") as f:
        magic, n, nnz = np.fromfile(f, dtype="<i8", count=3)
        assert magic == CSR_MAGIC, path
        row_ptr = np.fromfile(f, dtype="<i8", count=n + 1)
        col = np.fromfile(f, dtype="<i4", count=nnz)
        val = np.fromfile(f, dtype="<f8", count=nnz)
    return sp.csr_matrix((val, col, row_ptr), shape=(n, n))


def same_values(x, y):
    x = x.toarray() if sp.issparse(x) else x
    y = y.toarray() if sp.issparse(y) else y
    return np.array_equal(x, y)


def read_dense(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    meas = list(csv.DictReader(open(os.path.join(args.results, "raw", "measurements.csv"))))
    runs = [r for r in csv.DictReader(open(os.path.join(args.results, "raw", "runs.csv")))
            if r["status"] == "ok"]
    checks = {}
    for r in meas:
        if r["phase"] == "measure":
            checks.setdefault((r["experiment"], r["method"], r["input"], r["n"]), set()).add(
                r["checksum"])

    rows = []
    groups = {}
    for r in runs:
        groups.setdefault((r["experiment"], r["input"], int(r["n"])), []).append(r)

    for (exp, inp, n), group in sorted(groups.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        if exp == "dense":
            A = read_dense(os.path.join(args.data, f"A_{n}.bin"), n)
            B = read_dense(os.path.join(args.data, f"B_{n}.bin"), n)
            As = Bs = None
        else:
            structure = group[0]["structure"]
            d = float(group[0]["density_pct"])
            stem = os.path.join(args.data, "sparse", f"{structure}_n{n}_d{d:g}")
            As, Bs = read_csr(stem + "_A.csr"), read_csr(stem + "_B.csr")
            A = B = None
            if n <= 4096:
                A, B = As.toarray(), Bs.toarray()

        if A is not None:
            ref = A @ B
            bound = 2 * gamma(n) * (np.abs(A) @ np.abs(B))
            ref_nnz = int(np.count_nonzero(ref))
            ref_sum = float(ref.sum())
        else:
            try:
                ref_s = (As @ Bs).tocsr()
                ref_nnz, ref_sum = int(ref_s.nnz), float(ref_s.sum())
            except MemoryError:
                ref_s = None
                ref_nnz = ref_sum = None

        results = {}
        for r in group:
            m = r["method"]
            tag = f"{m}_n{n}" if exp == "dense" else f"{m}_{inp}_n{n}"
            ext = "csr" if m == "csr" else "bin"
            path = os.path.join(args.outputs, exp, f"{tag}.{ext}")
            row = {"experiment": exp, "method": m, "input": inp, "n": n,
                   "reference": "numpy dense A@B" if A is not None else "scipy csr A@B",
                   "max_abs_error": "", "max_error_over_bound": "", "within_tolerance": "",
                   "nnz_c": r["nnz_c"], "nnz_reference": ref_nnz if ref_nnz is not None else "",
                   "checksums_consistent_across_reps":
                       len(checks.get((exp, m, inp, str(n)), {0})) == 1,
                   "equal_to": "", "note": ""}
            if os.path.exists(path):
                if A is not None:
                    C = read_csr(path).toarray() if m == "csr" else read_dense(path, n)
                    err = np.abs(C - ref)
                    row["max_abs_error"] = f"{err.max():.3e}"
                    row["max_error_over_bound"] = f"{(err / np.where(bound > 0, bound, 1)).max():.3e}"
                    row["within_tolerance"] = bool((err <= bound).all())
                    results[m] = C
                else:
                    bound_s = 2 * gamma(n) * (abs(As) @ abs(Bs))
                    if m == "csr":
                        C = read_csr(path)
                        diff = abs(C - ref_s)
                        over = diff - bound_s
                        row["max_abs_error"] = f"{diff.max():.3e}" if diff.nnz else "0.000e+00"
                        row["within_tolerance"] = bool(over.nnz == 0 or over.max() <= 0)
                        row["note"] = "elementwise sparse comparison"
                    else:
                        C = read_dense(path, n)
                        err = np.abs(C - ref_s.toarray())
                        bound = bound_s.toarray()
                        row["max_abs_error"] = f"{err.max():.3e}"
                        row["within_tolerance"] = bool((err <= bound).all())
                        row["nnz_c"] = int(np.count_nonzero(C))
                        row["note"] = "dense result vs scipy csr product"
                        del err, bound
                    results[m] = C
                    del bound_s
            elif ref_nnz is not None:
                chk = float(next(iter(checks[(exp, m, inp, str(n))])))
                ok_nnz = int(r["nnz_c"]) == ref_nnz
                ok_sum = abs(chk - ref_sum) <= 1e-9 * max(1.0, abs(ref_sum))
                row["within_tolerance"] = ok_nnz and ok_sum
                row["note"] = f"no dump: nnz {'equal' if ok_nnz else 'DIFFERENT'}, checksum " \
                              f"rel. diff {abs(chk - ref_sum) / max(1.0, abs(ref_sum)):.1e}"
            else:
                row["note"] = "not validated: reference product does not fit in memory"
            if m == "csr" and str(ref_nnz) != str(r["nnz_c"]) and ref_nnz is not None:
                row["note"] += " nnz differs from reference"
            rows.append(row)

        # Agreement between methods on the same input.
        for row in rows:
            if row["experiment"] == exp and row["input"] == inp and row["n"] == n:
                same = [o for o in results if o != row["method"] and row["method"] in results
                        and same_values(results[o], results[row["method"]])]
                row["equal_to"] = " ".join(sorted(same))
        status = ", ".join(f"{r['method']}={r['within_tolerance']}" for r in rows
                           if r["experiment"] == exp and r["input"] == inp and r["n"] == n)
        print(f"{exp:12s} {inp:14s} n={n:6d}: {status}", flush=True)

    path = os.path.join(args.results, "validation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    failed = [r for r in rows if r["within_tolerance"] is False]
    if failed:
        raise SystemExit(f"{len(failed)} results exceed the tolerance")


if __name__ == "__main__":
    main()
