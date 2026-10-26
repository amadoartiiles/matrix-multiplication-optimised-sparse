"""Summarise the raw measurements and draw the figures used in the report.

Inputs : results/raw/measurements.csv, results/raw/runs.csv, data/manifest.json,
         results/tuning/block_sweep.csv, results/supplementary/vectorization.csv (optional)
Outputs: results/summary/summary.csv     median, IQR, ... per configuration and phase
         results/summary/sparse.csv      Part B table: nnz, work, times, memory per input
         results/summary/limits.csv      status of every attempted configuration
         results/summary/tables.tex      LaTeX tables included by the report
         results/figures/*.png           figures

The "work" of the CSR product is the number of scalar products
sum_k nnz(A[:, k]) * nnz(B[k, :]), computed here from the input files.

    python scripts/analyze.py --data DIR [--results results]
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

METHODS = ["ijk", "ikj", "transposed", "blocked"]
NAME = {"ijk": "ijk (baseline)", "ikj": "ikj (loop reorder)", "transposed": "transposed B",
        "blocked": "blocked (tiled ikj)", "csr": "CSR (Gustavson)"}
SHORT = {"ijk": "ijk", "ikj": "ikj", "transposed": "transposed", "blocked": "blocked", "csr": "CSR"}
# Validated categorical palette (same first slots as Assignment 1).
COLOR = {"ijk": "#2a78d6", "ikj": "#eb6834", "transposed": "#1baf7a", "blocked": "#4a3aa7",
         "csr": "#d4357a"}
MARKER = {"ijk": "o", "ikj": "s", "transposed": "^", "blocked": "D", "csr": "v"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CSR_MAGIC = 0x31525343


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


def fmt_t(x):
    if x >= 1:
        return f"{x:.2f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.2f} ms"
    return f"{x * 1e6:.1f} $\\mu$s"


def fmt_x(x):
    if x >= 1000:
        return f"{x:,.0f}".replace(",", "\\,") + "$\\times$"
    return f"{x:.3g}$\\times$"


def pct_axis(ax, axis="x"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)


def fmt_int(x):
    return f"{int(x):,}".replace(",", "\\,")


def fmt_pct(x):
    return f"{x:.3g}"


def read_csr_counts(path):
    """Row and column nonzero counts of a CSR1 file."""
    with open(path, "rb") as f:
        _, n, nnz = np.fromfile(f, dtype="<i8", count=3)
        row_ptr = np.fromfile(f, dtype="<i8", count=n + 1)
        col = np.fromfile(f, dtype="<i4", count=nnz)
    return np.diff(row_ptr), np.bincount(col, minlength=n)


def products(data, structure, n, d):
    """sum_k nnz(A[:,k]) nnz(B[k,:]) for the stored input pair."""
    stem = os.path.join(data, "sparse", f"{structure}_n{n}_d{d:g}")
    _, a_cols = read_csr_counts(stem + "_A.csr")
    b_rows, _ = read_csr_counts(stem + "_B.csr")
    return int(np.dot(a_cols.astype(np.int64), b_rows.astype(np.int64)))


def stats(values):
    t = np.array(values, dtype=float)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return {"reps": len(t), "median_s": med, "q1_s": q1, "q3_s": q3,
            "rel_iqr_pct": 100 * (q3 - q1) / med if med > 0 else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    res = args.results
    sum_dir, fig_dir = os.path.join(res, "summary"), os.path.join(res, "figures")
    os.makedirs(sum_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

    meas = list(csv.DictReader(open(os.path.join(res, "raw", "measurements.csv"))))
    runs = list(csv.DictReader(open(os.path.join(res, "raw", "runs.csv"))))
    run_by = {(r["experiment"], r["method"], r["input"], int(r["n"])): r for r in runs}

    # ---------- per-configuration, per-phase summary ----------
    groups = {}
    for r in meas:
        if r["phase"] == "warmup":
            continue
        key = (r["experiment"], r["method"], r["input"], int(r["n"]), r["phase"])
        groups.setdefault(key, []).append(float(r["time_per_mult_s"]))
    summary = {}
    for key, vals in groups.items():
        exp, m, inp, n, phase = key
        run = run_by[(exp, m, inp, n)]
        if run["status"] != "ok":
            continue
        s = stats(vals)
        s.update({"experiment": exp, "method": m, "input": inp, "n": n, "phase": phase,
                  "density_pct": float(run["density_pct"]), "structure": run["structure"],
                  "inner": next(int(r["inner"]) for r in meas if r["experiment"] == exp
                                and r["method"] == m and r["input"] == inp and int(r["n"]) == n
                                and r["phase"] == phase),
                  "peak_working_set_mb": int(run["peak_working_set_bytes"]) / 1e6,
                  "nnz_c": int(run["nnz_c"]) if run["nnz_c"] else ""})
        summary[key] = s
    fields = ["experiment", "method", "input", "n", "density_pct", "structure", "phase", "reps",
              "inner", "median_s", "q1_s", "q3_s", "rel_iqr_pct", "peak_working_set_mb", "nnz_c"]
    with open(os.path.join(sum_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(summary.values(), key=lambda s: (s["experiment"], s["n"], s["input"],
                                                          s["method"], s["phase"])):
            w.writerow({k: (f"{s[k]:.6g}" if isinstance(s[k], float) else s[k]) for k in fields})

    def med(exp, m, inp, n, phase="measure"):
        s = summary.get((exp, m, inp, n, phase))
        return s["median_s"] if s else None

    # ---------- limits ----------
    with open(os.path.join(sum_dir, "limits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "method", "input", "largest_ok_n", "largest_attempted_n",
                    "failures"])
        for key in sorted({(r["experiment"], r["method"], r["input"]) for r in runs}):
            mine = [r for r in runs if (r["experiment"], r["method"], r["input"]) == key]
            ok = [int(r["n"]) for r in mine if r["status"] == "ok"]
            att = [int(r["n"]) for r in mine if r["status"] != "skipped"]
            fail = "; ".join(f"n={r['n']}: {r['status']} ({r['note']})" for r in mine
                             if r["status"] != "ok")
            w.writerow([*key, max(ok) if ok else "", max(att) if att else "", fail or "none"])

    tex = ["% Generated by scripts/analyze.py -- do not edit by hand."]

    # =====================================================================
    # Part A: dense
    # =====================================================================
    dense_sizes = sorted({s["n"] for s in summary.values() if s["experiment"] == "dense"})

    def dense_series(m, key="median_s"):
        rows = [summary[k] for k in summary if k[0] == "dense" and k[1] == m]
        rows.sort(key=lambda s: s["n"])
        return np.array([s["n"] for s in rows]), np.array([s[key] for s in rows], dtype=float)

    # Figure: time vs n
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for m in METHODS:
        n, t = dense_series(m)
        _, q1 = dense_series(m, "q1_s")
        _, q3 = dense_series(m, "q3_s")
        ax.errorbar(n, t, yerr=[t - q1, q3 - t], color=COLOR[m], marker=MARKER[m], markersize=4.5,
                    linewidth=2, capsize=3, label=NAME[m])
    ref = np.array([dense_sizes[0], dense_sizes[-1]], dtype=float)
    n_b, t_b = dense_series("blocked")
    n_anchor = 512 if 512 in n_b else n_b[0]
    anchor = t_b[list(n_b).index(n_anchor)]
    ax.plot(ref, anchor * (ref / n_anchor) ** 3, linestyle="--", color=INK2, linewidth=1,
            label=r"$\propto n^3$ guide")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "matrix size n (n × n)", "time per multiplication (s, median)",
          "Part A: runtime vs matrix size (error bars: IQR)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "fig_dense_time.png"))
    plt.close(fig)

    # Figure: ns per inner iteration (t / n^3)
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for m in METHODS:
        n, t = dense_series(m)
        ax.plot(n, t / n.astype(float) ** 3 * 1e9, color=COLOR[m], marker=MARKER[m],
                markersize=4.5, linewidth=2, label=NAME[m])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "matrix size n (n × n)", "median time / n³ (ns)",
          "Cost per multiply-add (flat line = pure cubic growth)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "fig_dense_per_n3.png"))
    plt.close(fig)

    # Figure: speed-up vs baseline
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    n0, t0 = dense_series("ijk")
    base = dict(zip(n0, t0))
    for m in METHODS[1:]:
        n, t = dense_series(m)
        common = [i for i, x in enumerate(n) if x in base]
        ax.plot(n[common], [base[n[i]] / t[i] for i in common], color=COLOR[m], marker=MARKER[m],
                markersize=4.5, linewidth=2, label=NAME[m])
    ax.axhline(1, color=INK2, linewidth=1, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "matrix size n (n × n)", "speed-up over ijk (×, medians)",
          "Speed-up of each optimisation over the baseline")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "fig_dense_speedup.png"))
    plt.close(fig)

    # Table: dense times
    tex.append("\\newcommand{\\DenseTimeTable}{")
    tex.append("\\begin{tabular}{r " + "r r " * len(METHODS) + "}\\toprule")
    tex.append("& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[m]}}}" for m in METHODS)
               + " \\\\" + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                                   for i in range(len(METHODS))))
    tex.append("$n$ & " + " & ".join(["Median & IQR\\%"] * len(METHODS)) + " \\\\\\midrule")
    for n in dense_sizes:
        cells = []
        for m in METHODS:
            s = summary.get(("dense", m, "dense", n, "measure"))
            if s:
                cells += [fmt_t(s["median_s"]), f"{s['rel_iqr_pct']:.1f}"]
            else:
                st = run_by.get(("dense", m, "dense", n), {}).get("status", "--")
                cells += [f"\\multicolumn{{2}}{{c}}{{\\textit{{{st}}}}}"]
        tex.append(f"{n} & " + " & ".join(cells) + " \\\\")
    tex.append("\\bottomrule\\end{tabular}}")

    # Table: speed-up, GFLOP/s, memory
    tex.append("\\newcommand{\\DenseSpeedupTable}{")
    tex.append("\\begin{tabular}{r " + "r " * 3 + "r " * 4 + "r r}\\toprule")
    tex.append("& \\multicolumn{3}{c}{Speed-up over ijk} & \\multicolumn{4}{c}{GFLOP/s ($2n^3$/median)}"
               " & \\multicolumn{2}{c}{Peak WS (MB)} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}"
               "\\cmidrule(lr){9-10}")
    tex.append("$n$ & ikj & transp. & blocked & ijk & ikj & transp. & blocked & ijk & transp. "
               "\\\\\\midrule")
    for n in dense_sizes:
        b = med("dense", "ijk", "dense", n)
        cells = []
        for m in METHODS[1:]:
            t = med("dense", m, "dense", n)
            cells.append(f"{b / t:.1f}" if b and t else "--")
        for m in METHODS:
            t = med("dense", m, "dense", n)
            cells.append(f"{2 * n ** 3 / t / 1e9:.2f}" if t else "--")
        for m in ("ijk", "transposed"):
            r = run_by.get(("dense", m, "dense", n))
            cells.append(f"{int(r['peak_working_set_bytes']) / 1e6:.0f}" if r and r["status"] == "ok"
                         else "--")
        tex.append(f"{n} & " + " & ".join(cells) + " \\\\")
    tex.append("\\bottomrule\\end{tabular}}")

    # Block-size sweep
    sweep_path = os.path.join(res, "tuning", "block_sweep.csv")
    if os.path.exists(sweep_path):
        sw = [r for r in csv.DictReader(open(sweep_path)) if r["phase"] == "measure"]
        blocks = sorted({int(r["block"]) for r in sw})
        sizes = sorted({int(r["n"]) for r in sw})
        smed = {(b, n): float(np.median([float(r["time_per_mult_s"]) for r in sw
                                         if int(r["block"]) == b and int(r["n"]) == n]))
                for b in blocks for n in sizes}
        tex.append("\\newcommand{\\BlockSweepTable}{")
        tex.append("\\begin{tabular}{r " + "r " * len(blocks) + "}\\toprule $n$ & "
                   + " & ".join(f"$b={b}$" for b in blocks) + " \\\\\\midrule")
        for n in sizes:
            best = min(blocks, key=lambda b: smed[(b, n)])
            tex.append(f"{n} & " + " & ".join(
                (f"\\textbf{{{fmt_t(smed[(b, n)])}}}" if b == best else fmt_t(smed[(b, n)]))
                for b in blocks) + " \\\\")
        tex.append("\\bottomrule\\end{tabular}}")

    # Vectorisation supplementary
    vec_path = os.path.join(res, "supplementary", "vectorization.csv")
    if os.path.exists(vec_path):
        vr = [r for r in csv.DictReader(open(vec_path)) if r["phase"] == "measure"]
        vmeth = list(dict.fromkeys(r["method"] for r in vr))
        vsizes = sorted({int(r["n"]) for r in vr})
        vmed = {}
        for r in vr:
            vmed.setdefault((r["build"], r["method"], int(r["n"])), []).append(
                float(r["time_per_mult_s"]))
        vmed = {k: float(np.median(v)) for k, v in vmed.items()}
        same = all(len({r["checksum"] for r in vr if r["n"] == str(n)}) == 1 for n in vsizes)
        tex.append("\\newcommand{\\VecTable}{")
        tex.append("\\begin{tabular}{l r r r r}\\toprule Method & $n$ & -O2 (SSE2 vectorised) & "
                   "-O2 -fno-tree-vectorize & Ratio \\\\\\midrule")
        for m in vmeth:
            for n in vsizes:
                a, b = vmed[("vectorized", m, n)], vmed[("novec", m, n)]
                tex.append(f"{SHORT[m]} & {n} & {fmt_t(a)} & {fmt_t(b)} & {b / a:.2f}$\\times$ \\\\")
            tex.append("\\addlinespace[2pt]")
        tex.append("\\bottomrule\\end{tabular}}")
        tex.append(f"\\newcommand{{\\VecChecksumsEqual}}{{{'yes' if same else 'no'}}}")

    # =====================================================================
    # Part B: sparse
    # =====================================================================
    csr_only = {}
    mem_path = os.path.join(res, "supplementary", "sparse_memory.csv")
    if os.path.exists(mem_path):
        for r in csv.DictReader(open(mem_path)):
            if r["status"] == "ok":
                csr_only[(int(r["n"]), float(r["density_pct"]))] = int(r["peak_working_set_bytes"]) / 1e6
    sparse_rows = []
    for exp in ("sparse", "structure", "sparse_large"):
        inputs = sorted({(int(r["n"]), float(r["density_pct"]), r["structure"], r["input"])
                         for r in runs if r["experiment"] == exp})
        for n, d, structure, inp in inputs:
            rc = run_by.get((exp, "csr", inp, n))
            row = {"experiment": exp, "n": n, "density_pct": d, "structure": structure,
                   "input": inp, "csr_status": rc["status"] if rc else "",
                   "nnz_a": "", "nnz_b": "", "nnz_c": "", "out_density_pct": "",
                   "products": products(args.data, structure, n, d)}
            if rc and rc["status"] == "ok":
                row.update({"nnz_a": int(rc["nnz_a"]), "nnz_b": int(rc["nnz_b"]),
                            "nnz_c": int(rc["nnz_c"]),
                            "out_density_pct": 100 * int(rc["nnz_c"]) / n ** 2,
                            "csr_bytes_in": float(rc["csr_bytes_a"]) + float(rc["csr_bytes_b"]),
                            "csr_bytes_c": float(rc["csr_bytes_c"]),
                            "csr_peak_ws_mb": int(rc["peak_working_set_bytes"]) / 1e6})
                if exp == "sparse":
                    row["csr_only_peak_ws_mb"] = csr_only.get((n, d), "")
            for phase in ("measure", "to_csr", "to_dense"):
                s = summary.get((exp, "csr", inp, n, phase))
                row[f"csr_{phase}_s"] = s["median_s"] if s else ""
                if phase == "measure":
                    row["csr_rel_iqr_pct"] = s["rel_iqr_pct"] if s else ""
            for m in ("blocked", "ijk"):
                s = summary.get((exp, m, inp, n, "measure"))
                r = run_by.get((exp, m, inp, n))
                row[f"{m}_s"] = s["median_s"] if s else ""
                row[f"{m}_status"] = r["status"] if r else ""
                row[f"{m}_peak_ws_mb"] = (int(r["peak_working_set_bytes"]) / 1e6
                                          if r and r["status"] == "ok" else "")
            sparse_rows.append(row)
    if sparse_rows:
        keys = list(dict.fromkeys(k for r in sparse_rows for k in r))
        with open(os.path.join(sum_dir, "sparse.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in sparse_rows:
                w.writerow({k: (f"{r.get(k, ''):.6g}" if isinstance(r.get(k), float) else r.get(k, ""))
                            for k in keys})

    def sp(exp, n=None, structure=None):
        return [r for r in sparse_rows if r["experiment"] == exp and (n is None or r["n"] == n)
                and (structure is None or r["structure"] == structure)]

    # Table: matched comparison
    tex.append("\\newcommand{\\SparseTable}{")
    tex.append("\\begin{tabular}{r r r r r r r r r r}\\toprule")
    tex.append("$n$ & Dens.\\,\\% & nnz(A) & nnz(C) & Out dens.\\,\\% & CSR mult. & "
               "+ conversions & Blocked & ijk & Speed-up$^\\dagger$ \\\\\\midrule")
    for n in sorted({r["n"] for r in sp("sparse")}):
        for r in sp("sparse", n):
            conv = (r["csr_measure_s"] + r["csr_to_csr_s"] + r["csr_to_dense_s"]
                    if r["csr_to_csr_s"] != "" else None)
            spd = r["blocked_s"] / r["csr_measure_s"] if r["blocked_s"] and r["csr_measure_s"] else None
            tex.append(f"{n} & {fmt_pct(r['density_pct'])} & {fmt_int(r['nnz_a'])} & "
                       f"{fmt_int(r['nnz_c'])} & {r['out_density_pct']:.3g} & "
                       f"{fmt_t(r['csr_measure_s'])} & {fmt_t(conv) if conv else '--'} & "
                       f"{fmt_t(r['blocked_s']) if r['blocked_s'] else '--'} & "
                       f"{fmt_t(r['ijk_s']) if r['ijk_s'] else '--'} & "
                       f"{fmt_x(spd) if spd else '--'} \\\\")
        tex.append("\\addlinespace[3pt]")
    tex.append("\\bottomrule\\end{tabular}}")

    # Table: conversion breakdown + memory
    tex.append("\\newcommand{\\SparseCostTable}{")
    tex.append("\\begin{tabular}{r r r r r r r r r r}\\toprule")
    tex.append("& & \\multicolumn{3}{c}{CSR time (median)} & & \\multicolumn{3}{c}{Peak WS (MB)} & "
               "Model (MB) \\\\\\cmidrule(lr){3-5}\\cmidrule(lr){7-9}\\cmidrule(lr){10-10}")
    tex.append("$n$ & Dens.\\,\\% & A,B$\\to$CSR & Multiply & C$\\to$dense & ns/product & "
               "CSR$^\\ast$ & CSR only & Blocked & CSR $A,B,C$ \\\\\\midrule")
    for n in sorted({r["n"] for r in sp("sparse")}):
        for r in sp("sparse", n):
            nsp = r["csr_measure_s"] / r["products"] * 1e9 if r["products"] else 0
            model = (r["csr_bytes_in"] + r["csr_bytes_c"]) / 1e6
            only = f"{r['csr_only_peak_ws_mb']:.1f}" if r.get("csr_only_peak_ws_mb") else "--"
            tex.append(f"{n} & {fmt_pct(r['density_pct'])} & {fmt_t(r['csr_to_csr_s'])} & "
                       f"{fmt_t(r['csr_measure_s'])} & {fmt_t(r['csr_to_dense_s'])} & {nsp:.2f} & "
                       f"{r['csr_peak_ws_mb']:.0f} & {only} & "
                       f"{r['blocked_peak_ws_mb']:.0f} & {model:.1f} \\\\"
                       if r["blocked_peak_ws_mb"] != "" else
                       f"{n} & {fmt_pct(r['density_pct'])} & -- & {fmt_t(r['csr_measure_s'])} & -- & "
                       f"{nsp:.2f} & {r['csr_peak_ws_mb']:.0f} & {only} & -- & {model:.1f} \\\\")
        tex.append("\\addlinespace[3pt]")
    tex.append("\\bottomrule\\end{tabular}}")

    # Table: structure
    st_rows = sp("structure")
    if st_rows:
        tex.append("\\newcommand{\\StructureTable}{")
        tex.append("\\begin{tabular}{r l r r r r r}\\toprule")
        tex.append("Dens.\\,\\% & Structure & nnz(A) & Products & nnz(C) & CSR mult. & "
                   "ns/product \\\\\\midrule")
        n = st_rows[0]["n"]
        for r in st_rows:
            u = next((x for x in sp("sparse", n) if x["density_pct"] == r["density_pct"]), None)
            for x in ([u] if u else []) + [r]:
                tex.append(f"{fmt_pct(x['density_pct'])} & {x['structure']} & {fmt_int(x['nnz_a'])} & "
                           f"{fmt_int(x['products'])} & {fmt_int(x['nnz_c'])} & "
                           f"{fmt_t(x['csr_measure_s'])} & "
                           f"{x['csr_measure_s'] / x['products'] * 1e9:.2f} \\\\")
            tex.append("\\addlinespace[2pt]")
        tex.append("\\bottomrule\\end{tabular}}")

    # Table: large sizes
    lg = sp("sparse_large")
    if lg:
        tex.append("\\newcommand{\\LargeTable}{")
        tex.append("\\begin{tabular}{r r r r r r r r}\\toprule")
        tex.append("$n$ & Dens.\\,\\% & nnz(A) & nnz(C) & Out dens.\\,\\% & CSR mult. & "
                   "CSR peak WS & Blocked dense \\\\\\midrule")
        for r in sorted(lg, key=lambda r: (r["density_pct"], r["n"])):
            if r["csr_status"] == "ok":
                cells = (f"{fmt_int(r['nnz_a'])} & {fmt_int(r['nnz_c'])} & "
                         f"{r['out_density_pct']:.3g} & {fmt_t(r['csr_measure_s'])} & "
                         f"{r['csr_peak_ws_mb']:.0f} MB")
            else:
                rc = run_by[("sparse_large", "csr", r["input"], r["n"])]
                cells = (f"-- & -- & -- & \\textit{{{r['csr_status'].replace('_', ' ')}}} & "
                         f"{int(rc['peak_working_set_bytes']) / 1e6 if rc['peak_working_set_bytes'] else 0:.0f}"
                         " MB" if rc["peak_working_set_bytes"] else
                         f"-- & -- & -- & \\textit{{{r['csr_status'].replace('_', ' ')}}} & --")
            if r["blocked_s"]:
                dense = fmt_t(r["blocked_s"])
            elif r["blocked_status"]:
                dense = f"\\textit{{{r['blocked_status'].replace('_', ' ')}}}"
            else:
                dense = "--"
            tex.append(f"{r['n']} & {fmt_pct(r['density_pct'])} & {cells} & {dense} \\\\")
        tex.append("\\bottomrule\\end{tabular}}")

    # Figure: time vs density (two panels)
    sizes_sp = sorted({r["n"] for r in sp("sparse")})
    if sizes_sp:
        fig, axes = plt.subplots(1, len(sizes_sp), figsize=(10.2, 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, sizes_sp):
            rows = sp("sparse", n)
            d = np.array([r["density_pct"] for r in rows])
            ax.plot(d, [r["csr_measure_s"] for r in rows], color=COLOR["csr"], marker="v",
                    linewidth=2, markersize=5, label="CSR multiply")
            conv = [r["csr_measure_s"] + r["csr_to_csr_s"] + r["csr_to_dense_s"]
                    if r["csr_to_csr_s"] != "" else np.nan for r in rows]
            ax.plot(d, conv, color=COLOR["csr"], linestyle=":", marker="v", markerfacecolor="white",
                    linewidth=1.6, markersize=5, label="CSR incl. conversions")
            for m in ("blocked", "ijk"):
                y = [r[f"{m}_s"] if r[f"{m}_s"] else np.nan for r in rows]
                if not np.all(np.isnan(y)):
                    ax.plot(d, y, color=COLOR[m], marker=MARKER[m], linewidth=2, markersize=4.5,
                            label=f"dense {NAME[m]}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            pct_axis(ax)
            style(ax, "input density (% nonzero, A and B)", "time per multiplication (s)" if n ==
                  sizes_sp[0] else "", f"n = {n}")
        axes[0].legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(fig_dir, "fig_sparse_time.png"))
        plt.close(fig)

        # Figure: output density
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        dd = np.logspace(-1, 2, 200)
        for n, c in zip(sizes_sp, ("#2a78d6", "#eb6834")):
            rows = sp("sparse", n)
            ax.plot([r["density_pct"] for r in rows], [r["out_density_pct"] for r in rows],
                    color=c, marker="o", linestyle="none", markersize=5, label=f"measured, n={n}")
            p = dd / 100
            ax.plot(dd, 100 * (1 - (1 - p * p) ** n), color=c, linewidth=1.2,
                    label=f"model $1-(1-d^2)^n$, n={n}")
        ax.plot(dd, dd, color=INK2, linestyle="--", linewidth=1, label="output = input density")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pct_axis(ax)
        pct_axis(ax, "y")
        style(ax, "input density d (%)", "output density of C (%)", "Fill-in: density of the product")
        ax.legend(frameon=False, fontsize=7.8)
        fig.savefig(os.path.join(fig_dir, "fig_sparse_fill.png"))
        plt.close(fig)

        # Figure: memory vs density
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        for n, ls in zip(sizes_sp, ("-", "--")):
            rows = sp("sparse", n)
            d = [r["density_pct"] for r in rows]
            y = [r.get("csr_only_peak_ws_mb") or np.nan for r in rows]
            ax.plot(d, y, color=COLOR["csr"], marker="v",
                    linestyle=ls, linewidth=2, markersize=5, label=f"CSR multiply only, n={n}")
            ax.plot(d, [(r["csr_bytes_in"] + r["csr_bytes_c"]) / 1e6 for r in rows],
                    color=COLOR["csr"], linestyle=ls, linewidth=0.9, alpha=0.55,
                    label=f"CSR storage model, n={n}")
            ax.plot(d, [r["blocked_peak_ws_mb"] or np.nan for r in rows], color=COLOR["blocked"],
                    marker="D", linestyle=ls, linewidth=2, markersize=4.5, label=f"dense blocked, n={n}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pct_axis(ax)
        style(ax, "input density (%)", "peak working set (MB)", "Peak process memory vs density")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(fig_dir, "fig_sparse_memory.png"))
        plt.close(fig)

        # Figure: time per product
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for n, ls in zip(sizes_sp, ("-", "--")):
            rows = sp("sparse", n)
            ax.plot([r["products"] for r in rows],
                    [r["csr_measure_s"] / r["products"] * 1e9 for r in rows], color=COLOR["csr"],
                    marker="v", linestyle=ls, linewidth=2, markersize=5, label=f"CSR uniform, n={n}")
        if st_rows:
            ax.plot([r["products"] for r in st_rows],
                    [r["csr_measure_s"] / r["products"] * 1e9 for r in st_rows], color="#1baf7a",
                    marker="s", linestyle="none", markersize=5, label=f"CSR banded, n={st_rows[0]['n']}")
        for n, ls in zip(sizes_sp, ("-", "--")):
            b = [r["blocked_s"] for r in sp("sparse", n) if r["blocked_s"]]
            if b:
                ax.axhline(np.median(b) / n ** 3 * 1e9, color=COLOR["blocked"], linestyle=ls,
                           linewidth=1.2, label=f"dense blocked per $n^3$, n={n}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "scalar products sum_k nnz(A[:,k]) nnz(B[k,:])", "time per product (ns)",
              "Cost per scalar product")
        ax.legend(frameon=False, fontsize=7.8)
        fig.savefig(os.path.join(fig_dir, "fig_sparse_per_product.png"))
        plt.close(fig)

    # Figure: large sizes
    if lg:
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        for d, c in zip(sorted({r["density_pct"] for r in lg}), ("#d4357a", "#2a78d6")):
            rows = sorted([r for r in lg if r["density_pct"] == d and r["csr_status"] == "ok"],
                          key=lambda r: r["n"])
            ax.plot([r["n"] for r in rows], [r["csr_measure_s"] for r in rows], color=c,
                    marker="v", linewidth=2, markersize=5, label=f"CSR, density {d:g}%")
        rows = sorted([r for r in lg if r["blocked_s"]], key=lambda r: r["n"])
        rows_small = sorted([r for r in sp("sparse") if r["density_pct"] == 0.1 and r["blocked_s"]],
                            key=lambda r: r["n"])
        allr = rows_small + rows
        if allr:
            ax.plot([r["n"] for r in allr], [r["blocked_s"] for r in allr], color=COLOR["blocked"],
                    marker="D", linewidth=2, markersize=4.5, label="dense blocked, density 0.1%")
        ax.axhline(300, color=INK2, linestyle="--", linewidth=1, label="300 s budget")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style(ax, "matrix size n (n × n)", "time per multiplication (s)",
              "Beyond the dense budget (uniform structure)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(fig_dir, "fig_sparse_large.png"))
        plt.close(fig)

    with open(os.path.join(sum_dir, "tables.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"summary rows: {len(summary)}, sparse rows: {len(sparse_rows)}")


if __name__ == "__main__":
    main()
