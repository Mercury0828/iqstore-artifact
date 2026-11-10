"""M2 workload manifest generator (prereg/M2.md v4 §0, §4; frozen before any TEST ingest).

Writes prereg/M2_workload.json: everything random in the M2 workload, generated from the prereg seed streams
(numpy SeedSequence([20261107, STREAM, dataset_index, extra])), plus the deterministic rules that are resolved against
the stored data at evaluation time (positions are classical bits; their physical qubits come from the stored
measurement map of each job).

Discriminators are stored in NORMALIZED family coordinates (u, v): theta = th_lo + u (th_hi - th_lo),
beta = b_lo + v (b_hi - b_lo), with the family of that dataset / position / family rule. The family is centred on the
reference, so (0.5, 0.5) is the reference. Out-of-family draws use u, v in [-1, 2] (= +-3 half-widths around the
reference) with rejection of points inside [0, 1]^2.

Usage: python eval/workload.py [--out prereg/M2_workload.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collect.m1_circuits import N_DAYS, TARGET_NAMES, schedule  # noqa: E402

SEED = 20261107
STREAM = {"subsets": 1, "in_family": 2, "out_family": 3, "he_alloc": 4, "synthetic": 5, "subsample": 6,
          "bootstrap_ci": 7, "timing_order": 8, "fboot": 9, "sscale": 10}
DEV_DAYS = range(1, 6)
N_SHOTS = 20_000
N_IN_FAMILY = 20
N_OUT_FAMILY = 10
N_SUBSETS = 50
SWEEP_U = [0.1, 0.3, 0.5, 0.7, 0.9]
SWEEP_V = [0.05 + 0.1 * i for i in range(10)]
DELTAS = [0.01, 0.03, 0.1, 0.3, 1.0]
SUBSAMPLE_N = [1_000, 2_000, 5_000, 10_000, 20_000]
N_SUBSAMPLE_RANDOM = 5
HE_TOTAL = 1_000_000


def rng(stream: str, dataset_index: int, extra: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, STREAM[stream], dataset_index, extra]))


def positions(name: str) -> list[list]:
    """Canonical position list [register, bit] in register order (= the stored circuit's classical bits)."""
    if name.startswith("REP_"):
        r = int(name.split("_r")[1].split("_")[0])
        return [[f"a{t}", i] for t in range(1, r + 1) for i in range(4)] + [["data", i] for i in range(5)]
    return [["meas", i] for i in range(16)]


def pair_rules(name: str, pos: list[list]) -> dict:
    if name.startswith("REP_"):
        r = int(name.split("_r")[1].split("_")[0])
        idx = {(p[0], p[1]): k for k, p in enumerate(pos)}
        data_nb = [[idx[("data", i)], idx[("data", i + 1)]] for i in range(4)]
        anc_rounds = [[idx[(f"a{t}", i)], idx[(f"a{t + 1}", i)]] for t in range(1, r) for i in range(4)]
        return {"rule": "explicit", "pairs": data_nb + anc_rounds}
    return {"rule": "every frozen coupler (qpu/M1_qubits.json edges) whose two physical qubits are both measured, "
                    "resolved through the stored measurement map; pairs ordered by position index"}


def mitigated_rule(name: str, pos: list[list]) -> dict:
    if name.startswith("REP_"):
        return {"rule": "explicit", "S": [k for k, p in enumerate(pos) if p[0] == "data" and p[1] < 4]}
    return {"rule": "positions measuring the first 4 qubits of the frozen longest path (qpu/M1_qubits.json), "
                    "in path order, resolved through the stored measurement map"}


def out_family_uv(g: np.random.Generator, n: int) -> list[list[float]]:
    out = []
    while len(out) < n:
        u, v = g.uniform(-1.0, 2.0, 2)
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            continue  # inside the family: rejected
        out.append([float(u), float(v)])
    return out


def build() -> dict:
    days = schedule()
    day_of = {t: d + 1 for d, ts in enumerate(days) for t in ts}
    test_idx = [i for i in range(len(TARGET_NAMES)) if day_of[i] not in DEV_DAYS]
    per_test = -(-HE_TOTAL // len(test_idx))
    datasets = []
    digest = hashlib.sha256()
    for i, name in enumerate(TARGET_NAMES):
        pos = positions(name)
        m = len(pos)
        sub = {}
        for k in (3, 4):
            g = rng("subsets", i, k)
            sub[str(k)] = [sorted(g.choice(m, k, replace=False).tolist()) for _ in range(N_SUBSETS)]
        g = rng("in_family", i)
        in_fam = g.uniform(0.0, 1.0, (N_IN_FAMILY, m, 2)).tolist()
        g = rng("out_family", i)
        out_fam = [out_family_uv(g, m) for _ in range(N_OUT_FAMILY)]
        g = rng("synthetic", i)
        # real-stream variants: per later event day, the positions changed under '1' and '4'
        later = [d for d in range(day_of[i] + 1, N_DAYS + 1)]
        ev1 = {str(d): [int(g.integers(m))] for d in later}
        ev4 = {str(d): sorted(g.choice(m, min(4, m), replace=False).tolist()) for d in later}
        synth = [{"kind": kind, "delta": dl, "order": n_} for n_, (kind, dl) in
                 enumerate([("beta", d_) for d_ in DELTAS] + [("theta", d_) for d_ in DELTAS])]
        g = rng("subsample", i)
        # seeds of the random shot subsets (each subset = default_rng(seed).choice(N_SHOTS, n, replace=False))
        subs = {str(n): [int(g.integers(1 << 31)) for _ in range(N_SUBSAMPLE_RANDOM)] for n in SUBSAMPLE_N}
        ds = {"index": i, "id": name, "campaign_day": day_of[i], "split": "DEV" if day_of[i] in DEV_DAYS else "TEST",
              "kind": "R-MID" if name.startswith("REP_") else "R-FINAL", "m": m, "positions": pos,
              "queries": {"1_single": "every position",
                          "2_pairs": pair_rules(name, pos),
                          "3_marginals": sub,
                          "4_mitigated": mitigated_rule(name, pos),
                          "5_sweep": {"u": SWEEP_U, "v": [round(v, 2) for v in SWEEP_V],
                                      "vectors": "50 = 5 u x 10 v, the same (u, v) at every position of S"}},
              "vectors": {"a_reference": [0.5, 0.5],
                          "b_later_refits": f"one vector per later day {later} (none on day {N_DAYS})",
                          "c_in_family_uv": in_fam, "d_out_family_uv": out_fam},
              "maintenance": {"real_stream_positions_1": ev1, "real_stream_positions_4": ev4,
                              "synthetic_events": synth, "synthetic_reset": "to the ingest-day reference, untimed"},
              "subsamples": {"first_n": SUBSAMPLE_N, "random_subset_seeds": subs}}
        if ds["split"] == "TEST":
            g = rng("he_alloc", i)
            cls = g.integers(1, 6, per_test)
            src = g.choice(np.array([2, 1, 3]), per_test, p=[0.5, 0.25, 0.25])  # 2 = (c), 1 = (b), 3 = (d)
            sel = g.integers(0, 1 << 30, per_test)
            src[cls == 5] = 0  # class 5 carries its own prescribed sweep vectors (prereg §4 item 5; sources are for 1-4)
            digest.update(cls.astype(np.int8).tobytes() + src.astype(np.int8).tobytes() + sel.astype(np.int64).tobytes())
            ds["he_allocation"] = {"n_pairs": per_test, "stream": "he_alloc", "class_counts":
                                   {str(c): int((cls == c).sum()) for c in range(1, 6)},
                                   "source_codes": {"0": "class 5 only: its own sweep vector (no other source)",
                                                    "1": "b later refit (falls back to c if none)", "2": "c in-family",
                                                    "3": "d out-of-family"},
                                   "selector": "classes 1-4: query instance = sel mod (#instances of the class, in "
                                               "manifest order); vector = (sel >> 15) mod (#vectors of the source). "
                                               "class 5: instance = sel mod (#pairs x 50), enumerated pair-major "
                                               "(pair p, sweep point i) -> p * 50 + i, sweep points u-major in the "
                                               "order of 5_sweep.u x 5_sweep.v; the vector is that sweep point"}
        datasets.append(ds)
    return {"prereg": "prereg/M2.md v4", "seed": SEED, "streams": STREAM, "n_shots": N_SHOTS, "dev_days": list(DEV_DAYS),
            "schedule": days, "datasets": datasets, "he_total_requested": HE_TOTAL,
            "he_per_test_dataset": per_test, "he_stream_sha256": digest.hexdigest(),
            "normalized_coordinates": __doc__.split("Discriminators are stored")[1].split("Usage")[0].strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "prereg" / "M2_workload.json"))
    a = ap.parse_args()
    w = build()
    data = json.dumps(w, indent=1, sort_keys=True).encode("utf-8")  # LF bytes on every platform
    out = pathlib.Path(a.out)
    out.write_bytes(data)
    print(f"wrote {a.out}: {len(w['datasets'])} datasets, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
