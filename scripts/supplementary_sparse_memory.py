"""Supplementary experiment: peak memory of the CSR multiplication alone.

In the main sparse experiment the CSR process also times the conversions, for which it
expands A and B to dense arrays (16 n^2 bytes) after the multiplications; its peak working
set therefore includes those buffers. This script reruns the CSR configurations of the
"sparse" experiment without conversions and without dumps (1 warm-up + 1 measured
repetition, same driver and memory sampling) and writes results/supplementary/sparse_memory.csv.

    python scripts/supplementary_sparse_memory.py --data DIR
"""
import argparse
import csv
import json
import os

from run_benchmarks import build_jobs, run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/supplementary")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    jobs = [j for j in build_jobs(cfg, ["sparse"], args.data, "unused") if j["method"] == "csr"]
    fields = ["n", "density_pct", "status", "peak_working_set_bytes", "nnz_a", "nnz_b", "nnz_c",
              "csr_bytes_a", "csr_bytes_b", "csr_bytes_c", "time_per_mult_s"]
    rows = []
    for job in jobs:
        job.update(dump=None, conversions=False)
        run, meas = run_config(job, cfg, warmup=1, reps=1)
        t = [r[-3] for r in meas if r[7] == "measure"]
        rows.append({**{k: run[k] for k in fields if k in run}, "n": job["n"],
                     "density_pct": job["density_pct"], "time_per_mult_s": t[0] if t else ""})
        print(f"n={job['n']:5d} d={job['density_pct']:>5}%: {run['status']}, "
              f"peak WS {run['peak_working_set_bytes'] / 1e6:.1f} MB", flush=True)
    with open(os.path.join(args.out, "sparse_memory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
