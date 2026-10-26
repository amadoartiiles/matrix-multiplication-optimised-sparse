"""Benchmark driver: runs every configuration of the selected experiments as a separate process.

Same protocol as the Assignment 1 driver. For each configuration it
  * launches build/dense or build/sparse in "bench" mode on the shared input files,
  * reads its CSV rows (warm-up, measured and conversion repetitions) as they are produced,
  * samples the process' peak memory with psutil while it runs,
  * stops it if one multiplication exceeds the time limit or the process exceeds
    the memory budget, and then skips larger sizes of that (experiment, method, input).

Experiments (config/benchmark_config.json): dense, sparse, structure, sparse_large.

Outputs (in --out, default results/):
  raw/measurements.csv  every row printed by the programs, labelled with the configuration
  raw/runs.csv          one row per configuration: status, memory, nnz, wall time, ...
  selftest.txt          output of the known-answer self tests
Result matrices of the last repetition go to --outputs (for validate_outputs.py).

    python scripts/run_benchmarks.py --data DIR --outputs DIR [--experiments dense sparse]
                                     [--sizes 1024] [--methods csr blocked] [--reps 3]
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil

EXE = ".exe" if sys.platform == "win32" else ""
CONFIG_FIELDS = ["experiment", "method", "input", "n", "density_pct", "structure", "block"]
RUN_FIELDS = ["timestamp", *CONFIG_FIELDS, "status", "exit_code", "warmup_runs", "measured_rows",
              "inner", "wall_time_s", "peak_working_set_bytes", "peak_private_bytes",
              "nnz_a", "nnz_b", "nnz_c", "csr_bytes_a", "csr_bytes_b", "csr_bytes_c", "note"]
ROW_FIELDS = ["phase", "rep", "inner", "batch_time_s", "time_per_mult_s", "checksum", "nnz_c"]
MEAS_FIELDS = [*CONFIG_FIELDS, *ROW_FIELDS]


def sparse_file(data, structure, n, d, name):
    return os.path.join(data, "sparse", f"{structure}_n{n}_d{d:g}_{name}.csr")


def build_jobs(cfg, experiments, data, outputs):
    """List of configurations, in execution order (n ascending, then input, then method)."""
    jobs = []
    for exp_name in experiments:
        exp = cfg["experiments"][exp_name]
        out_dir = os.path.join(outputs, exp_name)
        if exp_name == "dense":
            for n in exp["sizes"]:
                for m in exp["methods"]:
                    jobs.append({
                        "experiment": exp_name, "method": m, "input": "dense", "n": n,
                        "density_pct": 100, "structure": "dense",
                        "block": exp["block_size"] if m == "blocked" else "",
                        "a": os.path.join(data, f"A_{n}.bin"), "b": os.path.join(data, f"B_{n}.bin"),
                        "dump": os.path.join(out_dir, f"{m}_n{n}.bin"), "conversions": False})
            continue
        structure = exp["structure"]
        for n in exp["sizes"]:
            for d in exp["densities_pct"]:
                tag = f"{structure}_d{d:g}"
                for m, method_sizes in exp["methods"].items():
                    if n not in method_sizes:
                        continue
                    if m != "csr" and d not in exp.get("dense_densities_pct", [d]):
                        continue
                    dump = None
                    if n <= exp.get("dump_max_n", n):
                        ext = "csr" if m == "csr" else "bin"
                        dump = os.path.join(out_dir, f"{m}_{tag}_n{n}.{ext}")
                    jobs.append({
                        "experiment": exp_name, "method": m, "input": tag, "n": n,
                        "density_pct": d, "structure": structure,
                        "block": cfg["experiments"]["dense"]["block_size"] if m == "blocked" else "",
                        "a": sparse_file(data, structure, n, d, "A"),
                        "b": sparse_file(data, structure, n, d, "B"), "dump": dump,
                        "conversions": m == "csr" and exp.get("conversions", False)
                        and n <= exp.get("conversions_max_n", n)})
    return jobs


def command(job, cfg, warmup, reps):
    common = ["--n", str(job["n"]), "--a", job["a"], "--b", job["b"], "--input", job["input"],
              "--warmup", str(warmup), "--reps", str(reps),
              "--min-batch-time", str(cfg["min_batch_time_s"])]
    if job["dump"]:
        common += ["--dump", job["dump"]]
    if job["method"] == "csr":
        return [os.path.join("build", f"sparse{EXE}"), "bench", *common,
                "--conversions", "1" if job["conversions"] else "0"]
    cmd = [os.path.join("build", f"dense{EXE}"), "bench", "--method", job["method"], *common]
    if job["block"]:
        cmd += ["--block", str(job["block"])]
    return cmd


def tree_peak_memory(proc):
    """Peak working set and peak private bytes of the process and any children (bytes)."""
    peak_ws = peak_priv = 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0, 0
    for p in procs:
        try:
            mi = p.memory_info()
        except psutil.Error:
            continue
        peak_ws += getattr(mi, "peak_wset", mi.rss)
        peak_priv += getattr(mi, "peak_pagefile", getattr(mi, "vms", 0))
    return peak_ws, peak_priv


def run_config(job, cfg, warmup, reps):
    cmd = command(job, cfg, warmup, reps)
    limit = cfg["time_limit_per_multiplication_s"]
    budget = cfg["memory_budget_bytes"]
    interval = cfg["memory_sampling_interval_s"]
    if job["dump"]:
        os.makedirs(os.path.dirname(job["dump"]), exist_ok=True)

    start = time.perf_counter()
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc = psutil.Process(popen.pid)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in popen.stdout], daemon=True).start()

    rows, comments = [], []

    def drain():
        try:
            while True:
                line = lines.get_nowait().strip()
                if line.startswith("#"):
                    comments.append(line[1:].strip())
                elif line and not line.startswith("method,"):
                    rows.append(line.split(","))
        except queue.Empty:
            pass

    peak_ws = peak_priv = 0
    last_progress = time.perf_counter()
    status, note = "ok", ""
    while True:
        ws, priv = tree_peak_memory(proc)
        peak_ws, peak_priv = max(peak_ws, ws), max(peak_priv, priv)
        before = len(rows) + len(comments)
        drain()
        if len(rows) + len(comments) > before:
            last_progress = time.perf_counter()
        if popen.poll() is not None:
            break
        if peak_ws > budget:
            status, note = "memory_limit", f"peak working set exceeded {budget} bytes"
            popen.kill()
            break
        if time.perf_counter() - last_progress > limit:
            status, note = "timeout", f"no multiplication finished within {limit} s"
            popen.kill()
            break
        time.sleep(interval)
    popen.wait()
    wall = time.perf_counter() - start
    time.sleep(0.2)
    drain()
    stderr = popen.stderr.read()
    if status == "ok" and popen.returncode != 0:
        status, note = "error", stderr.strip().replace("\n", " ")[:300]

    extra = {}
    for c in comments:
        parts = c.split(",")
        extra.update(dict(zip(parts[0::2], parts[1::2])))
    measured = [r for r in rows if r[3] == "measure"]
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: job[k] for k in CONFIG_FIELDS},
        "status": status, "exit_code": popen.returncode, "warmup_runs": warmup,
        "measured_rows": len(measured), "inner": measured[0][5] if measured else "",
        "wall_time_s": f"{wall:.3f}", "peak_working_set_bytes": peak_ws,
        "peak_private_bytes": peak_priv,
        "nnz_a": extra.get("nnz_a", ""), "nnz_b": extra.get("nnz_b", ""),
        "nnz_c": extra.get("nnz_c", measured[-1][9] if measured else ""),
        "csr_bytes_a": extra.get("csr_bytes_a", ""), "csr_bytes_b": extra.get("csr_bytes_b", ""),
        "csr_bytes_c": extra.get("csr_bytes_c", ""), "note": note,
    }
    # Program rows: method,input,n,phase,rep,inner,batch,time,checksum,nnz_c -> labelled rows.
    labelled = [[job[k] for k in CONFIG_FIELDS] + r[3:] for r in rows]
    return run, labelled


def append_csv(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerows(rows)


def selftests(out_dir):
    with open(os.path.join(out_dir, "selftest.txt"), "w") as log:
        for prog in ("dense", "sparse"):
            res = subprocess.run([os.path.join("build", f"{prog}{EXE}"), "selftest"],
                                 capture_output=True, text=True)
            log.write(f"== {prog} ==\n{res.stdout}\n")
            print(f"[selftest] {prog}: {'PASSED' if res.returncode == 0 else 'FAILED'}")
            if res.returncode != 0:
                sys.exit(f"self test failed for {prog}:\n{res.stdout}{res.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--experiments", nargs="*",
                        default=["dense", "sparse", "structure", "sparse_large"])
    parser.add_argument("--sizes", type=int, nargs="*")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--resume", action="store_true",
                        help="skip configurations that already have status ok in runs.csv")
    parser.add_argument("--allow-battery", action="store_true",
                        help="run even if the laptop is not on AC power")
    args = parser.parse_args()
    battery = psutil.sensors_battery()
    if battery and not battery.power_plugged and not args.allow_battery:
        sys.exit("not on AC power: connect the charger (or pass --allow-battery)")

    with open(args.config) as f:
        cfg = json.load(f)
    reps = args.reps or cfg["measured_repetitions"]
    warmup = cfg["warmup_runs"]
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    selftests(args.out)

    jobs = build_jobs(cfg, args.experiments, args.data, args.outputs)
    if args.sizes:
        jobs = [j for j in jobs if j["n"] in args.sizes]
    if args.methods:
        jobs = [j for j in jobs if j["method"] in args.methods]

    stopped = {}
    meas_path = os.path.join(args.out, "raw", "measurements.csv")
    runs_path = os.path.join(args.out, "raw", "runs.csv")
    done = set()
    if args.resume and os.path.exists(runs_path):
        done = {(r["experiment"], r["method"], r["input"], int(r["n"]))
                for r in csv.DictReader(open(runs_path)) if r["status"] == "ok"}
    for job in jobs:
        key = (job["experiment"], job["method"], job["input"])
        label = f"{job['experiment']}/{job['method']} {job['input']} n={job['n']}"
        if (*key, job["n"]) in done:
            print(f"[done] {label}")
            continue
        # Dense A, B and C alone (24 n^2 bytes) must fit in the memory budget.
        if job["method"] != "csr" and 24 * job["n"] ** 2 > cfg["memory_budget_bytes"]:
            stopped[key] = "memory_limit"
            append_csv(runs_path, RUN_FIELDS, [[
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                *[job[k] for k in CONFIG_FIELDS], "memory_limit", "", "", 0, "", "", "", "",
                "", "", "", "", "", "",
                f"not attempted: storage model 24n^2 = {24 * job['n'] ** 2 / 1e9:.2f} GB "
                f"exceeds the {cfg['memory_budget_bytes'] / 1e9:.0f} GB budget"]])
            print(f"[skip] {label}: dense storage exceeds the memory budget")
            continue
        if key in stopped:
            append_csv(runs_path, RUN_FIELDS, [[
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                *[job[k] for k in CONFIG_FIELDS], "skipped", "", "", 0, "", "", "", "",
                "", "", "", "", "", "", f"not attempted: {stopped[key]} at a smaller size"]])
            print(f"[skip] {label}")
            continue
        print(f"[run ] {label} ...", flush=True)
        run, rows = run_config(job, cfg, warmup, reps)
        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged:
            run["note"] = (run["note"] + " on battery power").strip()
            print("       WARNING: on battery power", flush=True)
        append_csv(meas_path, MEAS_FIELDS, rows)
        append_csv(runs_path, RUN_FIELDS, [[run[k] for k in RUN_FIELDS]])
        med = sorted(float(r[len(CONFIG_FIELDS) + 4]) for r in rows
                     if r[len(CONFIG_FIELDS)] == "measure")
        med_s = f"median {med[len(med) // 2]:.4g} s" if med else ""
        print(f"       {run['status']} {med_s} peakWS {run['peak_working_set_bytes'] / 1e6:.1f} MB "
              f"nnz_c {run['nnz_c']}", flush=True)
        if run["status"] != "ok":
            stopped[key] = run["status"]
        time.sleep(cfg["pause_between_runs_s"])


if __name__ == "__main__":
    main()
