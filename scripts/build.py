"""Compile the dense and sparse programs into build/ using the configured flags.

    python scripts/build.py [--config config/benchmark_config.json]

Also builds build/dense_novec(.exe) with -fno-tree-vectorize added, used only by the
supplementary vectorisation experiment, and writes the GCC vectorisation report
(-fopt-info-vec-optimized) of the main build to results/vectorization_report.txt.
The C compiler is taken from the CC environment variable, or the configured one.
"""
import argparse
import json
import os
import subprocess
import sys

EXE = ".exe" if sys.platform == "win32" else ""


def run(cmd, **kw):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs("build", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cc = os.environ.get("CC", cfg["c"]["compiler"])
    flags = cfg["c"]["flags"]
    run([cc, *flags, "-o", f"build/dense{EXE}", "c/dense.c", "-lm"])
    run([cc, *flags, "-o", f"build/sparse{EXE}", "c/sparse.c", "-lm"])
    run([cc, *flags, "-fno-tree-vectorize", "-o", f"build/dense_novec{EXE}", "c/dense.c", "-lm"])

    report = []
    for src in ("c/dense.c", "c/sparse.c"):
        res = subprocess.run([cc, *flags, "-fopt-info-vec-optimized", "-c", src, "-o", os.devnull],
                             capture_output=True, text=True, check=True)
        report.append(f"== {src} ({' '.join(flags)}) ==\n{res.stderr}")
    with open("results/vectorization_report.txt", "w") as f:
        f.write("\n".join(report))
    print("wrote results/vectorization_report.txt")


if __name__ == "__main__":
    main()
