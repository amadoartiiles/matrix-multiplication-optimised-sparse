"""Supplementary experiment: peak memory of the CSR multiplication alone.

In the main sparse experiment the CSR process also times the conversions, for which it
expands A and B to dense arrays (16 n^2 bytes) after the multiplications; its peak working
set therefore includes those buffers. This script reruns the CSR configurations of the
"sparse" experiment without conversions and without dumps. The peak working set is a
per-process quantity, so each configuration is run in --reps separate processes
(1 warm-up + 1 measured multiplication each, same driver and memory sampling); every
process is one row of results/supplementary/sparse_memory.csv.

    python scripts/supplementary_sparse_memory.py --data DIR [--reps 5]
"""
import argparse
import csv
import json
import os
import time

from run_benchmarks import build_jobs, run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", default="results/supplementary")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    jobs = [j for j in build_jobs(cfg, ["sparse"], args.data, "unused") if j["method"] == "csr"]
    fields = ["n", "density_pct", "representation", "dtype", "threads", "process_rep", "status",
              "peak_working_set_bytes", "nnz_a", "nnz_b", "nnz_c", "csr_bytes_a", "csr_bytes_b",
              "csr_bytes_c", "time_per_mult_s"]
    rows = []
    for job in jobs:
        job.update(dump=None, conversions=False)
        peaks = []
        for rep in range(args.reps):
            run, meas = run_config(job, cfg, warmup=1, reps=1)
            t = [r[-3] for r in meas if r[7] == "measure"]
            rows.append({**{k: run[k] for k in fields if k in run}, "n": job["n"],
                         "density_pct": job["density_pct"], "representation": "csr",
                         "dtype": "float64", "threads": 1, "process_rep": rep,
                         "time_per_mult_s": t[0] if t else ""})
            peaks.append(run["peak_working_set_bytes"] / 1e6)
            time.sleep(cfg["pause_between_runs_s"])
        print(f"n={job['n']:5d} d={job['density_pct']:>5}%: peak WS "
              + " ".join(f"{p:.1f}" for p in peaks) + " MB", flush=True)
    with open(os.path.join(args.out, "sparse_memory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
