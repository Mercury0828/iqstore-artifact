"""Q3 / H-c driver: the chronological campaign event stream of prereg/M2.md v4 section 5 Q3 (amendment C1: campaign
epochs in place of days), IQSTORE (Theorem 3, incremental) vs B6 (full recompute from the same representation), on the
maintenance engine of `eval/m2_q3.py`, and A6 (Hazy-style maintenance on the raw IQ points, `eval/a6_maint.py`, block
size from the DEV freeze) on the same events.

Stream (per family, IQSTORE at the frozen settings, tau = 1e-3):
  for epoch d = 1 ... 14 in order:
    (i)  ingest the datasets of epoch d (container bytes decoded once at load) and materialize their views (classes 1-3
         of section 4 under the ingest reference); the synthetic events of section 4 run here, from the reference state:
         beta-moves then theta-moves, delta ascending, all positions changed, each followed by an untimed reset;
    (ii) apply the epoch-d re-fitted reference (LDA of epoch d calibration, in the dataset's own frame) as one version
         change to every previously ingested dataset, all changed positions of a dataset in one batch.
  Variations: real events with the changed positions restricted to the workload lists (`real_stream_positions_1`,
  `_4`) or all positions (each variation is its own replay with its own state); views x1 and x4 (x4 = every pair view
  materialized four times, one per pattern variant).
Timing (section 6): per event 2 warm-ups + 10 measured runs, each run from the pre-event state (restored untimed); the
order of the three methods in each run is a permutation drawn from STREAM 8 (`timing_order`, extra = 10**6 + event
serial of the dataset), one thread, CPU lock held. Equality of the incremental views with the full recompute after every run
(integer counts, exact); A6's exact views are checked against an exact recount from the raw points and must lie
inside IQSTORE's intervals. An event is in-family iff every changed position's new discriminator lies in the stored family.

Usage: python eval/m2_hc.py stream [--dev] [--family F-boot|F-drift] [--out DIR]
       python eval/m2_hc.py summary [--out DIR]
--dev: DEV datasets and DEV epochs only (screen; no TEST access). Without it the DEV freeze must be committed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import m2_data as M  # noqa: E402
from eval.m2_cov import queries  # noqa: E402
from eval.a6_maint import A6  # noqa: E402
from eval.m2_q3 import Maintainer, Store  # noqa: E402
from eval.workload import rng  # noqa: E402
from iqstore.family import Discriminator, family_from_widths, fit_lda  # noqa: E402
from iqstore.ingest import ingest_position  # noqa: E402
from iqstore.rep import PositionRep  # noqa: E402
from iqstore.storage import decode_iqstore_position  # noqa: E402

TAU = 1e-3
N_EPOCHS = 14
WARM, MEAS = 2, 10
REAL = ("1", "4", "all")
MULTS = (1, 4)


def _settings(dev: bool) -> tuple[dict, dict, str, int]:
    """(IQSTORE settings, F-drift half-widths per context, freeze sha or 'DEV', A6 block size)."""
    if dev:
        fd = json.loads((ROOT / "runs" / "m2_dev" / "fdrift.json").read_text(encoding="utf-8"))["contexts"]
        return {"d0": 3, "max_depth": 16, "split_fraction": 0.5}, fd, "DEV", 1024
    from eval.m2_test import freeze
    fz, sha = freeze()
    return fz["IQSTORE"], fz["F-drift"], sha, int(fz["A6"]["block_size"])


def _families(ds, fname: str, fd: dict):
    if fname == "F-drift":
        return [family_from_widths(p.cal.ref, fd[p.context]["dtheta_half_width"], fd[p.context]["dbeta_half_width"])
                for p in ds.positions]
    return [p.fams[fname] for p in ds.positions]


def _store(ds, fams, iq: dict) -> tuple[Store, float]:
    """Ingest at the frozen settings, then decode the container bytes (the resident representation is built from the
    persisted bytes, as at load time)."""
    t0 = time.perf_counter()
    reps = []
    for p, fam in zip(ds.positions, fams):
        x, y = p.cal.frame.apply(p.raw)
        _, ib, _ = ingest_position(p.cal.frame, fam, x, y, p.raw, TAU, iq["d0"], iq["max_depth"], iq["split_fraction"])
        fr, dfam, part, codes, exc_index, exc_raw = decode_iqstore_position(ib)
        ex, ey = fr.apply(exc_raw) if exc_raw.size else (np.zeros(0), np.zeros(0))
        reps.append(PositionRep(fr, dfam, part, codes, exc_index, ex, ey, exc_raw))
    return Store(reps), time.perf_counter() - t0


def _views(entry: dict, ds, ep, mult: int) -> list[list[int]]:
    q = queries(entry, ds, ep)
    return q["1"] + q["2"] * mult + q["3"]


def _refit(ds, ep2) -> list[Discriminator]:
    out = []
    for p in ds.positions:
        c2 = ep2.cal[(p.qubit, p.context)]
        x0, y0 = p.cal.frame.apply(c2.z0)
        x1, y1 = p.cal.frame.apply(c2.z1)
        out.append(fit_lda(x0, y0, x1, y1))
    return out


def _synthetic(fams, h0, kind: str, delta: float) -> list[Discriminator]:
    out = []
    for fa, ref in zip(fams, h0):
        if kind == "beta":
            out.append(Discriminator(ref.theta, ref.beta + delta * (fa.b_hi - ref.beta)))
        else:
            out.append(Discriminator(ref.theta + delta * (fa.th_hi - ref.theta), ref.beta))
    return out


def _summ(t: list[float]) -> dict:
    w, m = np.array(t[:WARM]), np.array(t[WARM:])
    return {"median": float(np.median(m)), "iqr": [float(np.percentile(m, 25)), float(np.percentile(m, 75))],
            "runs": [float(v) for v in m], "warmups": [float(v) for v in w]}


class _DS:
    """One ingested dataset: its store, families, reference and one Maintainer per (real variation, view multiple)."""

    def __init__(self, entry, ep, ds, fams, iq, block):
        self.entry, self.ds, self.fams = entry, ds, fams
        self.h0 = [p.cal.ref for p in ds.positions]
        self.st, self.t_build = _store(ds, fams, iq)
        self.vsets = {mult: _views(entry, ds, ep, mult) for mult in MULTS}
        self.mt = {(v, mult): Maintainer(self.st, self.vsets[mult], self.h0) for v in REAL for mult in MULTS}
        xy = [p.cal.frame.apply(p.raw) for p in ds.positions]
        self.xs = [np.ascontiguousarray(a[0]) for a in xy]
        self.ys = [np.ascontiguousarray(a[1]) for a in xy]
        self.block = block
        self.a6 = {k: self.new_a6(mt) for k, mt in self.mt.items()}
        self.serial = 0

    def new_a6(self, mt: Maintainer) -> A6:
        return A6(self.xs, self.ys, mt.h, [list(s[:k]) for s, k in zip(mt.views_iq.S.tolist(), mt.views_iq.k.tolist())],
                  self.block, mt.vptr, mt.vidx)

    def order(self) -> np.ndarray:
        self.serial += 1
        g = rng("timing_order", int(self.entry["index"]), 10 ** 6 + self.serial)
        return np.stack([g.permutation(3) for _ in range(WARM + MEAS)])

    def in_family(self, J, hs) -> bool:
        return all(self.st.reps[j].family.contains(hs[j]) for j in J)


def stream(dev: bool, fname: str, out: pathlib.Path) -> None:
    iq, fd, sha, block = _settings(dev)
    w = M.workload()
    avail = set(M.DEV_EPOCHS) if dev else set(M.ledger_jobs())
    epochs = [d for d in range(1, N_EPOCHS + 1) if d in avail]
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"events_{fname}.jsonl"
    if f.exists():
        raise FileExistsError(f"{f} exists (the stream is one chronological replay; remove it to re-run)")
    live: list[_DS] = []
    t_start = time.time()
    with open(f, "w", encoding="utf-8") as fo:
        fo.write(json.dumps({"header": True, "freeze_sha256": sha, "family": fname, "tau": TAU, "iqstore": iq, "a6_block": block,
                             "epochs": epochs, "dev": dev, "warmups": WARM, "measured": MEAS}) + "\n")

        def emit(dsx: _DS, rec: dict) -> None:
            rec |= {"dataset": dsx.ds.name, "split": dsx.entry["split"], "ingest_epoch": dsx.ds.epoch,
                    "m": len(dsx.ds.positions), "N": dsx.st.N}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()

        for d in epochs:
            ep = M.load_epoch(d, allow_test=not dev)
            # (ii) the epoch-d re-fit applied to every previously ingested dataset
            for dsx in live:
                hs = _refit(dsx.ds, ep)
                m = len(dsx.ds.positions)
                mnt = dsx.entry["maintenance"]
                for v in REAL:
                    J = list(range(m)) if v == "all" else list(mnt[f"real_stream_positions_{v}"][str(d)])
                    for mult in MULTS:
                        r = dsx.mt[(v, mult)].event_timed(J, hs, dsx.order(), dsx.a6[(v, mult)])
                        emit(dsx, {"kind": "real", "variation": v, "views_x": mult, "event_epoch": d, "J": J,
                                   "in_family": dsx.in_family(J, hs), "n_views": len(dsx.vsets[mult]),
                                   "iq": _summ(r["t_iq"]), "b6": _summ(r["t_b6"]), "a6": _summ(r["t_a6"]),
                                   "a6_work": r["a6_work"],
                                   "tables_median": float(np.median(r["t_tables"][WARM:])),
                                   **{k: r[k] for k in ("cells", "shots", "view_shot_updates", "views_recomputed")}})
            # (i) ingest the datasets of epoch d; synthetic events from the reference state
            for entry in w["datasets"]:
                if int(entry["campaign_day"]) != d or (dev and entry["split"] != "DEV"):
                    continue
                ds = M.load_dataset(entry, ep, families=(fname == "F-boot"))
                fams = _families(ds, fname, fd)
                dsx = _DS(entry, ep, ds, fams, iq, block)
                emit(dsx, {"kind": "ingest", "build_s": dsx.t_build, "n_views": {str(k): len(v) for k, v in
                                                                                dsx.vsets.items()}})
                m = len(ds.positions)
                for mult in MULTS:
                    mt = Maintainer(dsx.st, dsx.vsets[mult], dsx.h0)
                    a6 = dsx.new_a6(mt)
                    a6_ref = a6.snapshot()
                    for ev in entry["maintenance"]["synthetic_events"]:
                        hs = _synthetic(fams, dsx.h0, ev["kind"], ev["delta"])
                        J = list(range(m))
                        r = mt.event_timed(J, hs, dsx.order(), a6)
                        emit(dsx, {"kind": "synthetic", "variation": f"{ev['kind']} {ev['delta']:g}", "views_x": mult,
                                   "J": J, "in_family": dsx.in_family(J, hs), "n_views": len(dsx.vsets[mult]),
                                   "iq": _summ(r["t_iq"]), "b6": _summ(r["t_b6"]), "a6": _summ(r["t_a6"]),
                                   "a6_work": r["a6_work"],
                                   "tables_median": float(np.median(r["t_tables"][WARM:])),
                                   **{k: r[k] for k in ("cells", "shots", "view_shot_updates", "views_recomputed")}})
                        mt.reset(dsx.h0)  # untimed
                        a6.restore(a6_ref)
                live.append(dsx)
                print(f"epoch {d} ingest {ds.name} (m={m}) build {dsx.t_build:.1f}s; live {len(live)}; "
                      f"elapsed {time.time() - t_start:.0f}s", flush=True)
        fo.write(json.dumps({"footer": True, "all_runs_equal": True, "seconds": time.time() - t_start}) + "\n")
    print("done", f, flush=True)


# ---------------------------------------------------------------------------------------------------- summary

def _ratio_by_dataset(rows, other: str = "b6") -> dict[str, float]:
    tot: dict[str, list[float]] = {}
    for r in rows:
        a = tot.setdefault(r["dataset"], [0.0, 0.0])
        a[0] += r[other]["median"]
        a[1] += r["iq"]["median"]
    return {k: v[0] / v[1] for k, v in tot.items() if v[1] > 0}


def _day_ci(ratio: dict[str, float], day_of: dict[str, int], seed_rng) -> tuple[float, list[float]]:
    names = sorted(ratio)
    if not names:
        return float("nan"), [float("nan")] * 2
    med = float(np.median([ratio[n] for n in names]))
    days = sorted({day_of[n] for n in names})
    by = {d: [ratio[n] for n in names if day_of[n] == d] for d in days}
    bs = []
    for _ in range(10_000):
        pick = seed_rng.choice(len(days), len(days), replace=True)
        vals = [v for i in pick for v in by[days[i]]]
        bs.append(np.median(vals))
    return med, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summary(out: pathlib.Path) -> None:
    res = {}
    for f in sorted(out.glob("events_*.jsonl")):
        lines = [json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
        head = lines[0]
        if not lines[-1].get("footer"):
            print(f.name, "incomplete (no footer)")
            continue
        ev = [r for r in lines if r.get("kind") in ("real", "synthetic")]
        split = "DEV" if head["dev"] else "TEST"
        ev = [r for r in ev if r["split"] == split]
        day_of = {r["dataset"]: r["ingest_epoch"] for r in ev}
        groups = {"pooled (every event)": ev}
        for v in REAL:
            for mult in MULTS:
                groups[f"real {v} x{mult}"] = [r for r in ev if r["kind"] == "real" and r["variation"] == v
                                               and r["views_x"] == mult]
        for mult in MULTS:
            groups[f"synthetic x{mult}"] = [r for r in ev if r["kind"] == "synthetic" and r["views_x"] == mult]
        groups["in-family events"] = [r for r in ev if r["in_family"]]
        groups["out-of-family events"] = [r for r in ev if not r["in_family"]]
        fam = {}
        for gname, rows in groups.items():
            ratio = _ratio_by_dataset(rows)
            med, ci = _day_ci(ratio, day_of, rng("bootstrap_ci", 0, 3))
            ev_r = [r["b6"]["median"] / r["iq"]["median"] for r in rows]
            r6 = _ratio_by_dataset(rows, "a6")
            med6, ci6 = _day_ci(r6, day_of, rng("bootstrap_ci", 0, 4))
            fam[gname] = {"datasets": len(ratio), "events": len(rows), "median_dataset_ratio": med, "ci95": ci,
                          "a6_over_iq_median_dataset_ratio": med6, "a6_ci95": ci6,
                          "event_ratio_median": float(np.median(ev_r)) if ev_r else None,
                          "event_ratio_min": float(np.min(ev_r)) if ev_r else None,
                          "events_with_views_recomputed": sum(r["views_recomputed"] > 0 for r in rows)}
            print(f"{head['family']:8s} {gname:24s} datasets {len(ratio):2d} events {len(rows):4d}  B6/IQ median "
                  f"{med:7.1f} CI [{ci[0]:.1f}, {ci[1]:.1f}]   A6/IQ {med6:6.1f} CI [{ci6[0]:.1f}, {ci6[1]:.1f}]")
        res[head["family"]] = {"split": split, "freeze_sha256": head["freeze_sha256"], "groups": fam}
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["stream", "summary"])
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--family", default="F-boot", choices=["F-boot", "F-drift"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out) if a.out else ROOT / "runs" / ("m2_dev/hc" if a.dev else "m2_test/hc")
    if a.part == "stream":
        stream(a.dev, a.family, out)
    else:
        summary(out)


if __name__ == "__main__":
    main()
