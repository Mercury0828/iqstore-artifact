"""Assemble prereg/M2_dev_freeze.json from the DEV selection outputs (prereg/M2.md v4 §3, §5; AMENDMENT_C1 §5).

Inputs (DEV epochs 1-5 only): runs/m2_dev/fdrift.json, runs/m2_dev/r2/{iqgrid,b3a6,b0q,entropy,cov_summary}.json. The evidence
files are copied to prereg/dev_evidence/ (committed with the freeze) and their sha256 recorded. Writes the freeze file
only; committing it (and printing its sha256) is a separate step. Refuses to overwrite an existing freeze.

Usage: python eval/m2_freeze.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import m2_data as M  # noqa: E402
from eval.m2_cov import BUDGETS, MULTS  # noqa: E402
from iqstore.ingest import FACTORS  # noqa: E402

DEV = ROOT / "runs" / "m2_dev"
R1 = DEV / "r2"  # re-selection in a fresh directory (r1 evidence is invalid)
EVID = ROOT / "prereg" / "dev_evidence"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    fd = json.loads((DEV / "fdrift.json").read_text(encoding="utf-8"))
    grid = json.loads((R1 / "iqgrid.json").read_text(encoding="utf-8"))
    b3a6 = json.loads((R1 / "b3a6.json").read_text(encoding="utf-8"))
    b0q = json.loads((R1 / "b0q.json").read_text(encoding="utf-8"))
    cov = json.loads((R1 / "cov_summary.json").read_text(encoding="utf-8"))
    ent = json.loads((R1 / "entropy.json").read_text(encoding="utf-8"))
    n_dev = sum(1 for e in M.workload()["datasets"] if e["split"] == "DEV")
    # validation: complete DEV corpus, consistent provenance, exact engines
    if cov["datasets"] != n_dev or len(grid["datasets"]) != n_dev or len(b3a6["B3"]["datasets"]) != n_dev:
        raise RuntimeError("an input does not cover every DEV dataset")
    if cov["iqstore_settings"] != grid["selected"]:
        raise RuntimeError("the sweep did not use the selected IQSTORE settings")
    if b0q["mismatches"][b0q["primary"]] != 0:
        raise RuntimeError("the primary B0q engine is not exact")
    dev = sorted(e["id"] for e in M.workload()["datasets"] if e["split"] == "DEV")
    provs = {json.dumps(grid.get("provenance"), sort_keys=True), json.dumps(b3a6.get("provenance"), sort_keys=True),
             json.dumps(b0q.get("provenance"), sort_keys=True), json.dumps(ent.get("provenance"), sort_keys=True),
             cov["code_commit"]}
    if len(provs) != 1 or "null" in provs or "+dirty" in next(iter(provs)):
        raise RuntimeError("DEV evidence comes from different / unknown / dirty code or environment")
    for name, ids in (("iqgrid", sorted(grid["datasets"])), ("B3", sorted(b3a6["B3"]["datasets"])),
                      ("A6", sorted(b3a6["A6"]["datasets"])), ("b0q", sorted(b0q["datasets"])),
                      ("entropy", sorted(ent["datasets"]))):
        if ids != dev:
            raise RuntimeError(f"{name} does not cover exactly the DEV datasets")
    jobs = M.ledger_jobs()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    deployed = {base: b["deployed"] for base, b in cov["bases"].items()}
    return {
        "prereg": "prereg/M2.md v4",
        "amendments": ["prereg/AMENDMENT_C1_cadence.md", "docs/guide_amendments.md A11 (W3 repair r1)",
                       "docs/guide_amendments.md A12 (staged (c1) timing ordering)",
                       "docs/guide_amendments.md A13 (indexes derived in memory)",
                       "docs/guide_amendments.md A14 (prereg coder set)",
                       "docs/guide_amendments.md A15 (B0q engine selection sample)",
                       "docs/guide_amendments.md A16 (paper-impact triage, process)",
                       "docs/guide_amendments.md A17 (selective-read B0q engine candidate)"],
        "dev_epochs": list(M.DEV_EPOCHS),
        "dev_jobs": {str(e): jobs[e]["job_id"] for e in M.DEV_EPOCHS},
        "code_commit_at_freeze": head,
        "F-drift": {ctx: {k: c[k] for k in ("q99_abs_dtheta", "q99_abs_dbeta", "dtheta_half_width", "dbeta_half_width")}
                    for ctx, c in fd["contexts"].items()},
        "IQSTORE": {"d0": grid["selected"]["d0"], "max_depth": grid["selected"]["max_depth"],
                    "split_fraction": grid["selected"]["split_fraction"],
                    "dev_median_bits_selected": grid["selected"]["median_bits"],
                    "ingest_search": {"factors": list(FACTORS), "stop_rule": "two consecutive not smaller than the best, "
                                      "or one > 1.05 x best, or budget 0 (iqstore/ingest.py)"}},
        "B3": {"bins_per_axis": b3a6["B3"]["selected"], "dev_median_bits": b3a6["B3"]["median_bits"]},
        "A6": {"block_size": b3a6["A6"]["selected"], "dev_median_bits": b3a6["A6"]["median_bits"]},
        "B0q_engine": {"primary": b0q["primary"], "dev_median_seconds": b0q["median_s"],
                       "mismatches_vs_exact": b0q["mismatches"]},
        "Q6_label_entropy_tertiles_bits": ent["tertile_boundaries_bits"],
        "dev_sweep_code_commit": cov["code_commit"],
        "tradeoff_grid": {"multipliers": list(MULTS), "bases": list(deployed), "budgets_bits": list(BUDGETS),
                          "tau": 1e-3},
        "deployed": deployed,
        "competitor_timing_order_by_dev_query_time": "staged per amendment A12: per-configuration DEV median warm query "
                                                     "times of every competitor grid configuration (an ordering, not one "
                                                     "winner), measured on DEV with the final query engine and committed "
                                                     "as prereg/M2_dev_freeze_c1.json before any TEST timing",
        "indexes": "per amendment A13: cell -> shots CSR and cell-tuple index are derived in memory at load from the "
                   "stored bytes, identically for every method; not persisted, not in b-bar; their build is inside every "
                   "end-to-end timed run (warm and cold) and construction time, their memory in peak RSS; retained "
                   "indexes only in the labelled resident mode",
        "evidence": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if M.FREEZE.exists():
        raise SystemExit("freeze exists; refusing to overwrite")
    fz = build()
    if not a.dry_run:
        EVID.mkdir(parents=True, exist_ok=True)
        for src in (DEV / "fdrift.json", R1 / "iqgrid.json", R1 / "b3a6.json", R1 / "b0q.json", R1 / "cov_summary.json",
                    R1 / "entropy.json"):
            dst = EVID / src.name
            shutil.copyfile(src, dst)
            fz["evidence"][dst.relative_to(ROOT).as_posix()] = sha(dst)
        for f in sorted(R1.glob("cov*.json")):
            if f.name != "cov_summary.json":
                dst = EVID / f.name
                shutil.copyfile(f, dst)
                fz["evidence"][dst.relative_to(ROOT).as_posix()] = sha(dst)
        M.FREEZE.write_text(json.dumps(fz, indent=1), encoding="utf-8")
        print("written", M.FREEZE, sha(M.FREEZE))
    else:
        print(json.dumps(fz, indent=1)[:3000])


if __name__ == "__main__":
    main()
