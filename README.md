# Assignment 2 — Optimised Dense Matrix Multiplication and Sparse Matrices

BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027

Research questions:

* **Part A:** how do loop reordering, a transposed-B layout and cache blocking change the
  runtime and memory of sequential dense matrix multiplication compared with the validated
  Assignment 1 triple loop, as the matrix size n grows?
* **Part B:** for which tested densities and nonzero structures does a CSR sparse
  multiplication beat a dense method on the same logical inputs, and what limits the advantage?

The report is in `report/` (`report.tex` → `report.pdf`).

## Reused material from Assignment 1

* Baseline kernel: the C i-j-k kernel of `c/matmul.c` from
  <https://github.com/amadoartiiles/matrix-multiplication-studies> (commit `441ae4d`), copied
  into `c/dense.c` with only the function renamed (`multiply` -> `mm_ijk`).
* Dense inputs: same generator, seed (20261012) and file format; `generate_inputs.py --check-a1`
  verifies that the files are byte-identical to the Assignment 1 inputs (16/16 shared files).
* Driver, memory measurement (psutil peak working set), validation bound and budgets follow
  the Assignment 1 scripts.

## Repository layout

```
c/common.h                  timer, dense/CSR file I/O, conversions, checksums
c/dense.c                   dense kernels ijk, ikj, transposed, blocked (+ self test + bench)
c/sparse.c                  CSR x CSR -> CSR (Gustavson) (+ self test + bench + conversion timing)
config/benchmark_config.json  sizes, densities, repetitions, budgets, seeds, flags, tile size
scripts/generate_inputs.py  dense (Assignment 1) and sparse CSR inputs + data/manifest.json
scripts/build.py            compiles build/dense, build/sparse, build/dense_novec; vectorisation report
scripts/tune_block.py       pre-study: tile size of the blocked kernel
scripts/run_benchmarks.py   driver: self tests, one process per configuration, memory, limits
scripts/validate_outputs.py correctness vs NumPy/SciPy with an error bound
scripts/supplementary_vectorization.py  supplementary: same kernels with -fno-tree-vectorize
scripts/supplementary_sparse_memory.py  supplementary: peak memory of the CSR multiply alone
scripts/collect_env.py      hardware / software description -> results/environment.json
scripts/analyze.py          summary tables and figures
results/raw/                raw measurements (measurements.csv, runs.csv)
results/tuning/             block_sweep.csv
results/supplementary/      vectorization.csv, sparse_memory.csv
results/summary/            summary.csv, sparse.csv, limits.csv, tables.tex
results/figures/            PNG figures used in the report
results/validation.csv      correctness results; results/selftest.txt known-answer tests
results/vectorization_report.txt  GCC -fopt-info-vec output
data/manifest.json          SHA-256 of every input file
report/                     LaTeX source and PDF of the report
```

## Requirements (versions used for the reported results)

| Component | Version |
|---|---|
| OS | Windows 11 Home 10.0.26300 |
| C compiler | GCC 16.2.0 (MinGW-w64 x86_64-ucrt-posix-seh, WinLibs build) |
| Python (scripts only) | CPython 3.11.9 |
| Python packages | numpy 2.2.5, scipy 1.17.1, matplotlib 3.10.1, psutil 7.2.2 |

The benchmarked programs are plain C11 without libraries. NumPy and SciPy are used only to
generate inputs and as correctness references.

## Reproduce everything

Run from the repository root (Git Bash shown). The input files (~1.2 GB) and result dumps are
large, so they can be placed outside the repository (the reported run used a local folder
outside OneDrive to avoid sync activity during the measurements):

```bash
pip install -r requirements.txt
DATA=data; OUT=outputs                                   # or any local folder

python scripts/generate_inputs.py --data $DATA           # 1. inputs (+ --check-a1 <A1 manifest>)
python scripts/build.py                                   # 2. gcc -O2 -std=c11 ...
python scripts/tune_block.py --data $DATA                 # 3. pre-study: tile size (-> config)
python scripts/run_benchmarks.py --data $DATA --outputs $OUT   # 4. full benchmark (~2 h, on AC power)
python scripts/supplementary_vectorization.py --data $DATA     # 5. supplementary
python scripts/supplementary_sparse_memory.py --data $DATA     #    (CSR memory without conversions)
python scripts/validate_outputs.py --data $DATA --outputs $OUT # 6. correctness
python scripts/collect_env.py                             # 7. environment
python scripts/analyze.py --data $DATA                    # 8. tables + figures
cd report && pdflatex report.tex && pdflatex report.tex   # 9. report
```

Individual programs:

```bash
build/dense.exe selftest
build/dense.exe bench --method blocked --block 64 --n 1024 --a data/A_1024.bin --b data/B_1024.bin
build/sparse.exe selftest
build/sparse.exe bench --n 2048 --a data/sparse/uniform_n2048_d1_A.csr \
    --b data/sparse/uniform_n2048_d1_B.csr --conversions 1
```

A quick pilot: `python scripts/run_benchmarks.py --data $DATA --outputs $OUT --out pilot --sizes 256 1024 --reps 3`.

## Fixed experimental settings

* Numeric type IEEE 754 binary64 everywhere; all kernels single-threaded; GCC flags
  `-O2 -std=c11 -Wall -Wextra` (no `-march`, no fast-math, so no FMA contraction).
* Dense kernels: allocation + zero initialisation of C + the loops are timed (transposed also
  times the copy into B^T). Inputs are loaded (and CSR inputs expanded) before timing.
* CSR kernel: both passes (symbolic + numeric) and the allocation of C are timed. Conversions
  dense→CSR (A and B) and CSR→dense (C) are timed **separately** and never added silently.
* 1 warm-up + 5 measured repetitions per configuration, one process per configuration; samples
  shorter than 0.1 s are batched (time per multiplication reported).
* Budgets: 300 s per multiplication, 4 GB peak working set per process; a configuration that
  exceeds a budget is stopped and larger sizes of the same method and input are not attempted.
* Memory metric: peak working set of the process (Windows `PeakWorkingSetSize`, psutil).
* Tile size 64 for the blocked kernel, fixed after the pre-study in `results/tuning/`.

## Raw data format

`results/raw/measurements.csv`:
`experiment,method,input,n,density_pct,structure,block,phase,rep,inner,batch_time_s,time_per_mult_s,checksum,nnz_c`
(`phase` is `warmup`, `measure`, `to_csr` or `to_dense`).

`results/raw/excluded.csv`: rows removed from the raw files, with the reason (a Modern Standby
interruption during the first ijk n=3072 run; both affected configurations were rerun).
An interrupted campaign can be continued with `run_benchmarks.py --resume`.

`results/raw/runs.csv`: one row per configuration with status (`ok`, `timeout`,
`memory_limit`, `error`, `skipped`), wall time, peak working set, nnz of A, B and C and the
CSR storage of each.

## Use of generative AI

Claude (Anthropic) was used to help write the implementations, scripts and a draft of the
report. All measurements in `results/` come from actually running the code on the machine
described in `results/environment.json`.
