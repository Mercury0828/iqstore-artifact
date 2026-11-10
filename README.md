# IQSTORE

Code and derived results for the paper *IQSTORE: Storing Qubit Readout Data for Queries Under Later Discriminators*
(Jiachen Shen, Hui Zhong).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24120145.svg)](https://doi.org/10.5281/zenodo.24120145)

IQSTORE stores kerneled qubit readout data (IQ points) so that counts, marginals, parities and readout-mitigated
expectations can be answered later under discriminators chosen after ingest. Points that every discriminator of a
declared range labels alike are stored as one bit, the rest as quadtree cells, and every query returns an interval
that contains the answer of the raw points.

## Requirements

Python 3.12 (the scripts use 3.12 syntax), Git (the evaluation drivers record the code commit) and the packages in
`requirements.txt`:

```
python -m pip install -r requirements.txt
```

## Layout

| path | content |
|---|---|
| `iqstore/` | the store: frames and declared sets, quadtree construction with the worst-case check, file layout, query engine, views |
| `baselines/` | storage baselines B0, B1, B2 (zfp, SZ3), B4, B4q, ablation A7, and the A6 view maintainer |
| `eval/` | evaluation drivers (storage, correctness, query timing, view maintenance, coverage) |
| `collect/` | circuits and job submission used to collect the campaign on `ibm_cleveland`, and the decoders |
| `tests/` | unit and end-to-end tests (`python -m pytest tests`) |
| `prereg/` | the fixed evaluation workload and the settings selected on the development split (kept byte-identical: every result file records its SHA-256) |
| `runs/m2_test/` | archived evaluation outputs (see below) |
| `paper/` | scripts that turn `runs/m2_test` into the result numbers, result tables and data figures of the paper |

## From archived results to the paper

```
python paper/gen_results.py      # writes paper/generated/
```

| paper element | produced from | output file |
|---|---|---|
| Table I sizes, Table III (storage), Sec. VII-B numbers | `runs/m2_test/q1/*.json` | `numbers.tex`, `tab_storage.tex` |
| Sec. VII-C ingest cost | `runs/m2_test/hc/events_*.jsonl` (ingest events) | `numbers.tex` |
| Sec. VII-D correctness | `runs/m2_test/he/*.json` | `numbers.tex` |
| Table IV and Fig. 3 (queries) | `runs/m2_test/q2/time/*.json`, `*.npz`, `q2/plan`, B0 headers in `q2/containers` | `tab_query.tex`, `fig_query.pdf` |
| Fig. 4 and Sec. VII-F (view maintenance) | `runs/m2_test/hc/events_*.jsonl` | `fig_maint.pdf`, `numbers.tex` |
| Sec. VII-G coverage | `paper/coverage.json` (from `paper/coverage_test.py`) | `numbers.tex` |

Figures 1 and 2 and Table II of the paper are drawings or definitions, not data. `gen_results.py` also writes
`fig_storage.pdf` (the data of Table III against the width target) and `tab_datasets.tex`, which the paper states in
the text of Sec. VII-A. `runs/m2_test/q2/containers/` holds the stored file of every method for every evaluation
dataset, so the query engines can be run on them directly (`iqstore/e2e.py`, `eval/b0q.py`).

## Re-running from the raw data

The raw readout data of the campaign (14 epochs, 35 target datasets and their calibration circuits, about 1 GB) are
archived on Zenodo (https://doi.org/10.5281/zenodo.24120145); `data/MANIFEST.csv` lists every file with its SHA-256.
Place the files under `data/` as listed in the manifest; `qpu/ledger.csv` lists every job of the campaign and maps
job ids to raw-result directories. Then decode the jobs with `collect/decode_m1.py` and run the drivers in `eval/` (`m2_storage.py`,
`m2_test.py`, `m2_q2.py`, `m2_hc.py`, `m2_cov.py`). Write new runs to a fresh output directory; the archived results in
`runs/m2_test` record the code version that produced them. Timed runs expect an otherwise idle machine: set
`IQSTORE_MACHINE_IDLE=1` and the thread variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`NUMBA_NUM_THREADS` to `1` (for example `export OMP_NUM_THREADS=1` in a POSIX shell).

## License

MIT (see `LICENSE`).
