"""Supplementary experiment: how much of the optimised kernels' gain is SIMD vectorisation?

GCC 16 at -O2 vectorises the unit-stride inner loops of ikj and blocked with 16-byte SSE2
vectors (results/vectorization_report.txt); the ijk and transposed inner loops are
reductions and stay scalar. This script runs the same source built with and without
-fno-tree-vectorize (build/dense vs build/dense_novec), alternating the two builds for
every (method, n), and writes results/supplementary/vectorization.csv.

    python scripts/supplementary_vectorization.py --data DIR [--sizes 256 512 1024 2048]
"""
import argparse
import json
import os
import sys

import numpy as np

from tune_block import bench_rows

EXE = ".exe" if sys.platform == "win32" else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--sizes", type=int, nargs="*", default=[256, 512, 1024, 2048])
    parser.add_argument("--methods", nargs="*", default=["ikj", "blocked", "transposed"])
    parser.add_argument("--out", default="results/supplementary")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    bs = str(cfg["experiments"]["dense"]["block_size"])
    builds = {"vectorized": os.path.join("build", f"dense{EXE}"),
              "novec": os.path.join("build", f"dense_novec{EXE}")}

    lines = ["build,method,input,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum,nnz_c"]
    for n in args.sizes:
        for m in args.methods:
            for build, exe in builds.items():
                rows = bench_rows(exe, m, n, args.data, cfg["warmup_runs"],
                                  cfg["measured_repetitions"], cfg["min_batch_time_s"],
                                  ["--block", bs])
                lines += [f"{build}," + ",".join(r) for r in rows]
                t = np.median([float(r[7]) for r in rows if r[3] == "measure"])
                print(f"n={n:5d} {m:10s} {build:10s}: median {t:.4g} s", flush=True)
    with open(os.path.join(args.out, "vectorization.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
