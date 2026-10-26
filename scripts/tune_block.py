"""Pre-study (run before the main benchmark): choose the tile size of the blocked kernel.

Runs build/dense bench --method blocked for every tile size and size below, with the
same warm-up/repetition protocol as the main study, and writes
results/tuning/block_sweep.csv. The tile size with the lowest median time summed
over the sizes is then fixed in config/benchmark_config.json (experiments.dense.block_size).

    python scripts/tune_block.py --data DIR [--sizes 512 1024 2048] [--blocks 16 32 64 128 256]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

EXE = ".exe" if sys.platform == "win32" else ""


def bench_rows(exe, method, n, data, warmup, reps, min_batch, extra=()):
    cmd = [exe, "bench", "--method", method, "--n", str(n), "--a", os.path.join(data, f"A_{n}.bin"),
           "--b", os.path.join(data, f"B_{n}.bin"), "--warmup", str(warmup), "--reps", str(reps),
           "--min-batch-time", str(min_batch), *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line.split(",") for line in out.strip().splitlines()[1:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--sizes", type=int, nargs="*", default=[512, 1024, 2048])
    parser.add_argument("--blocks", type=int, nargs="*", default=[16, 32, 64, 128, 256])
    parser.add_argument("--out", default="results/tuning")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    exe = os.path.join("build", f"dense{EXE}")

    lines = ["block,method,input,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum,nnz_c"]
    med = {}
    for n in args.sizes:
        for bs in args.blocks:
            rows = bench_rows(exe, "blocked", n, args.data, cfg["warmup_runs"],
                              cfg["measured_repetitions"], cfg["min_batch_time_s"],
                              ["--block", str(bs)])
            lines += [f"{bs}," + ",".join(r) for r in rows]
            med[(bs, n)] = float(np.median([float(r[7]) for r in rows if r[3] == "measure"]))
            print(f"n={n:5d} block={bs:4d}: median {med[(bs, n)]:.4g} s", flush=True)
    with open(os.path.join(args.out, "block_sweep.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    total = {bs: sum(med[(bs, n)] for n in args.sizes) for bs in args.blocks}
    best = min(total, key=total.get)
    print(f"best tile size (lowest summed median): {best}")


if __name__ == "__main__":
    main()
