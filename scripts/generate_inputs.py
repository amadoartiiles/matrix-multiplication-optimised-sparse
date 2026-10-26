"""Generate every input matrix of the study and a manifest with SHA-256 hashes.

Dense inputs (Part A) use exactly the Assignment 1 generator and seed, so A_n.bin
and B_n.bin are byte-identical to the Assignment 1 inputs for the shared sizes
(checked with --check-a1 against the Assignment 1 manifest).

Sparse inputs (Part B) are written in the CSR1 binary format described in
c/common.h and config/benchmark_config.json:

  uniform  exactly m = round(d * n^2) nonzeros at distinct, uniformly random positions
  banded   exactly m nonzeros in the narrowest band |i - j| <= w that can hold them;
           diagonals |i - j| < w are full and the rest are random positions on the
           two diagonals |i - j| = w

    python scripts/generate_inputs.py [--data data] [--only dense|sparse]
                                      [--check-a1 "../Assignment 1/data/manifest.json"]
"""
import argparse
import hashlib
import json
import os

import numpy as np

CSR_MAGIC = 0x31525343
STRUCTURE_ID = {"uniform": 0, "banded": 1}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- dense (Assignment 1)
def generate_dense(n, seed, low, high):
    rng = np.random.default_rng([seed, n])
    A = rng.uniform(low, high, size=(n, n)).astype("<f8")
    B = rng.uniform(low, high, size=(n, n)).astype("<f8")
    return A, B


# ---------------------------------------------------------------- sparse
def sparse_tag(structure, density_pct):
    return f"{structure}_d{density_pct:g}"


def sparse_path(data, structure, n, density_pct, name):
    return os.path.join(data, "sparse", f"{structure}_n{n}_d{density_pct:g}_{name}.csr")


def nonzero_values(rng, m):
    v = rng.uniform(-1.0, 1.0, size=m)
    while (v == 0.0).any():                      # probability ~2^-53 per value
        z = v == 0.0
        v[z] = rng.uniform(-1.0, 1.0, size=int(z.sum()))
    return v


def uniform_positions(rng, n, m):
    """m distinct linear indices in [0, n^2), uniformly at random, sorted."""
    total = n * n
    if m == total:
        return np.arange(total, dtype=np.int64)
    if m > total // 4:
        return np.sort(rng.permutation(total)[:m]).astype(np.int64)
    idx = np.unique(rng.integers(0, total, size=int(m * 1.05) + 16, dtype=np.int64))
    while idx.size < m:
        extra = rng.integers(0, total, size=2 * (m - idx.size) + 16, dtype=np.int64)
        idx = np.union1d(idx, extra)
    if idx.size > m:
        idx = np.sort(idx[rng.choice(idx.size, size=m, replace=False)])
    return idx


def band_count(n, w):
    """Number of positions with |i - j| <= w in an n x n matrix."""
    w = min(w, n - 1)
    return n * (2 * w + 1) - w * (w + 1)


def banded_positions(rng, n, m):
    w = 0
    while band_count(n, w) < m:
        w += 1
    inner = band_count(n, w - 1) if w > 0 else 0
    i = np.arange(n, dtype=np.int64)
    parts = []
    for off in range(-(w - 1), w) if w > 0 else []:
        rows = i[(i + off >= 0) & (i + off < n)]
        parts.append(rows * n + rows + off)
    outer = []
    for off in (-w, w) if w > 0 else (0,):
        rows = i[(i + off >= 0) & (i + off < n)]
        outer.append(rows * n + rows + off)
    outer = np.unique(np.concatenate(outer))
    chosen = rng.choice(outer.size, size=m - inner, replace=False)
    parts.append(outer[chosen])
    return np.sort(np.concatenate(parts)), w


def write_csr(path, n, idx, vals):
    rows = idx // n
    cols = (idx % n).astype("<i4")
    row_ptr = np.zeros(n + 1, dtype="<i8")
    np.cumsum(np.bincount(rows, minlength=n), out=row_ptr[1:])
    with open(path, "wb") as f:
        np.array([CSR_MAGIC, n, idx.size], dtype="<i8").tofile(f)
        row_ptr.tofile(f)
        cols.tofile(f)
        vals.astype("<f8").tofile(f)


def generate_sparse(data, n, density_pct, structure, seed):
    m = int(round(density_pct / 100.0 * n * n))
    info = {}
    for mid, name in enumerate(("A", "B")):
        rng = np.random.default_rng([seed, n, int(round(density_pct * 1000)),
                                     STRUCTURE_ID[structure], mid])
        if structure == "uniform":
            idx, w = uniform_positions(rng, n, m), None
        else:
            idx, w = banded_positions(rng, n, m)
        vals = nonzero_values(rng, idx.size)
        path = sparse_path(data, structure, n, density_pct, name)
        write_csr(path, n, idx, vals)
        counts = np.bincount(idx // n, minlength=n)
        info[name] = {"file": os.path.relpath(path, data).replace("\\", "/"), "nnz": int(idx.size),
                      "density_pct_actual": 100.0 * idx.size / (n * n),
                      "nnz_per_row_min": int(counts.min()), "nnz_per_row_max": int(counts.max()),
                      "empty_rows": int((counts == 0).sum()), "band_half_width": w,
                      "bytes": os.path.getsize(path), "sha256": sha256(path)}
    return info


def sparse_jobs(cfg):
    """Every (n, density, structure) needed by the sparse experiments."""
    jobs = set()
    for name, exp in cfg["experiments"].items():
        if name == "dense":
            continue
        for n in exp["sizes"]:
            for d in exp["densities_pct"]:
                jobs.add((n, d, exp["structure"]))
    return sorted(jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--only", choices=["dense", "sparse"])
    parser.add_argument("--check-a1", help="Assignment 1 data/manifest.json to compare hashes with")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(os.path.join(args.data, "sparse"), exist_ok=True)
    man_path = os.path.join(args.data, "manifest.json")
    manifest = {"numpy_version": np.__version__, "dense_input": cfg["dense_input"],
                "sparse_input": cfg["sparse_input"], "dense": {}, "sparse": {}}
    if os.path.exists(man_path):
        with open(man_path) as f:
            old = json.load(f)
        manifest["dense"], manifest["sparse"] = old.get("dense", {}), old.get("sparse", {})

    if args.only in (None, "dense"):
        inp = cfg["dense_input"]
        for n in cfg["experiments"]["dense"]["sizes"]:
            A, B = generate_dense(n, inp["seed"], inp["low"], inp["high"])
            for name, M in (("A", A), ("B", B)):
                path = os.path.join(args.data, f"{name}_{n}.bin")
                M.tofile(path)
                manifest["dense"][f"{name}_{n}.bin"] = {"n": n, "bytes": os.path.getsize(path),
                                                        "sha256": sha256(path)}
            print(f"dense n={n}: wrote A_{n}.bin and B_{n}.bin", flush=True)
        if args.check_a1:
            with open(args.check_a1) as f:
                a1 = json.load(f)["files"]
            shared = [k for k in a1 if k in manifest["dense"]]
            same = [k for k in shared if a1[k]["sha256"] == manifest["dense"][k]["sha256"]]
            manifest["dense_check_against_assignment1"] = {
                "files_compared": len(shared), "identical": len(same)}
            print(f"Assignment 1 inputs: {len(same)}/{len(shared)} shared files identical")

    if args.only in (None, "sparse"):
        seed = cfg["sparse_input"]["seed"]
        for n, d, structure in sparse_jobs(cfg):
            info = generate_sparse(args.data, n, d, structure, seed)
            manifest["sparse"][f"{structure}_n{n}_d{d:g}"] = {
                "n": n, "density_pct": d, "structure": structure, **info}
            print(f"sparse {structure} n={n} d={d}%: nnz={info['A']['nnz']}", flush=True)

    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
