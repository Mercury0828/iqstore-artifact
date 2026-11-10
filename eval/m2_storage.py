"""M2 Q1 storage at equal certificate (prereg/M2.md v4 §1, §3, §5 Q1; component breakdown [F12]).

Per dataset, family and tau: b-bar (8 x container bytes / (N m), everything included) for IQSTORE (DEV-frozen
partition settings), B4q (ingest-time grid search over the 64-value step set), B4, B2-zfp / B2-SZ3 (ingest-time eps
search over logspace(-4, 0.5, 136)), B0 (lossless complex128), B1 (reference labels, uncertified). Every lossy method is
certified per position by the same certifier (`iqstore.partition.certify`); IQSTORE positions are round-tripped from
their bytes. Same container and coder for every method (`iqstore.storage`).

Library module; `eval/m2_dev.py storage` runs it on DEV datasets.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.storage_baselines import (b0_blobs, b1_bitarray_bytes, b1_blobs, b2_position, b4_blobs,  # noqa: E402
                                         b4_slab, b4q_position, fixed_margins)
from iqstore.ingest import ingest_position  # noqa: E402
from iqstore.storage import bits_per_shot_position, decode_iqstore_position, write_container  # noqa: E402

EPS_GRID = np.logspace(-4, 0.5, 136)  # prereg/M2.md section 3, B2
DELTAS = np.array(sorted({c * 2.0 ** -p for p in range(0, 16) for c in (1.0, 1.25, 1.5, 1.75)}))  # 64 values, B4q/B5
assert DELTAS.size == 64


def _grouped(blobs: list[bytes]) -> list[bytes]:
    return [len(blobs).to_bytes(2, "little")] + blobs


def _cell_keys(part, codes):
    out = np.stack([codes.astype(np.int64), np.full(codes.size, -1), np.full(codes.size, -1), np.full(codes.size, -1)], 1)
    band = codes >= 3
    k = codes[band] - 3
    out[band] = np.stack([np.full(k.size, 3), part.depth[k], part.ix[k], part.iy[k]], 1)
    return out


def _measure(store: pathlib.Path, name: str, blobs: list[bytes], N: int, m: int, keep: bool) -> float:
    p = store / name
    nb = write_container(p, blobs, name.split("_")[0], N, m)
    if not keep:
        p.unlink()
    return bits_per_shot_position(nb, N, m)


def storage_row(ds, fname: str, tau: float, iq: dict, store: pathlib.Path, methods=("IQSTORE", "B4q", "B4", "B2-zfp",
                "B2-SZ3"), keep: bool = False, families: dict | None = None) -> dict:
    """b-bar of each method for one dataset / family / tau. `iq` = {d0, max_depth, split_fraction}. `families`
    optionally overrides the family per position (list of Family), e.g. F-drift."""
    N, m = ds.positions[0].raw.size, len(ds.positions)
    t0 = time.time()
    blobs = {k: [] for k in methods}
    st = {"exceptions": 0, "leaves": 0, "band_shots": 0, "roundtrip_ok": True, "b4q": [], "b2": {},
          "iq_bytes": {"header": 0, "tree": 0, "columns": 0, "exceptions": 0}, "fallbacks": {}}
    for j, p in enumerate(ds.positions):
        fam = families[j] if families is not None else p.fams[fname]
        fr, ref = p.cal.frame, p.cal.ref
        x, y = fr.apply(p.raw)
        if "IQSTORE" in methods:
            rep, ib, info = ingest_position(fr, fam, x, y, p.raw, tau, iq["d0"], iq["max_depth"], iq["split_fraction"])
            part, exc = info["part"], info["exc"]
            st.setdefault("iq_factor", []).append(info["factor"])
            dfr, dfam, dpart, dcodes, dexc, dexr = decode_iqstore_position(ib)
            st["roundtrip_ok"] &= bool(np.array_equal(_cell_keys(dpart, dcodes), _cell_keys(part, rep.codes))
                                       and np.array_equal(dexc, rep.exc_index) and np.array_equal(dexr, rep.exc_raw)
                                       and dpart.n_leaves == part.n_leaves and dfr == fr and dfam == fam)
            blobs["IQSTORE"] += _grouped(ib)
            bd = st["iq_bytes"]
            bd["header"] += len(ib[0]) + 4
            bd["tree"] += len(ib[1]) + 4
            ncol = len(ib) - 2 - (1 if rep.exc_raw.size else 0)
            bd["columns"] += sum(len(b) + 4 for b in ib[2:2 + ncol])
            bd["exceptions"] += sum(len(b) + 4 for b in ib[2 + ncol:])
            st["exceptions"] += int(exc.sum())
            st["leaves"] += part.n_leaves
            st["band_shots"] += int((rep.codes >= 3).sum())
        if "B4" in methods or "B4q" in methods:
            M0, M1 = fixed_margins(ref, fam, (float(x.min()), float(y.min()), float(x.max()), float(y.max())))
            in_slab, side1 = b4_slab(x, y, ref, M0, M1)
            if "B4" in methods:
                blobs["B4"] += _grouped(b4_blobs(p.raw, in_slab, side1, fr, ref, M0, M1, fam))
            if "B4q" in methods:
                qb, qinfo = b4q_position(x, y, in_slab, side1, p.raw, fr, ref, M0, M1, fam, tau, DELTAS)
                blobs["B4q"] += _grouped(qb)
                st["b4q"].append(qinfo | {"slab_shots": int(in_slab.sum())})
                if qinfo["mode"] != "grid":
                    st["fallbacks"].setdefault("B4q", []).append(j)
        if "A7" in methods:
            from baselines.a7 import a7_position
            a7rep, a7b, a7i = a7_position(fr, fam, x, y, p.raw, tau)
            blobs["A7"] += _grouped(a7b)
            st.setdefault("a7", []).append({k: v for k, v in a7i.items() if k != "tried"})
            if a7rep is None:
                st["fallbacks"].setdefault("A7", []).append(j)
        for codec in ("zfp", "SZ3"):
            if f"B2-{codec}" in methods:
                bb, binfo = b2_position(x, y, p.raw, fr, fam, tau, codec.lower(), EPS_GRID)
                blobs[f"B2-{codec}"] += _grouped(bb)
                st["b2"].setdefault(codec, []).append({k: v for k, v in binfo.items() if k not in ("rx", "ry")})
                if binfo["mode"] != "lossy":
                    st["fallbacks"].setdefault(f"B2-{codec}", []).append(j)
    row = {meth: _measure(store, f"{meth}_{ds.name}_{fname}_{tau:g}.bin", bl, N, m, keep) for meth, bl in blobs.items()}
    row["stats"] = st
    row["seconds"] = round(time.time() - t0, 1)
    return row


def lossless_row(ds, store: pathlib.Path, keep: bool = False) -> dict:
    """B0 (raw complex128, frame + F-boot family header) and B1 (reference labels; no certificate)."""
    N, m = ds.positions[0].raw.size, len(ds.positions)
    raw = ds.raw
    b0 = b0_blobs(raw, [p.cal.frame for p in ds.positions], [p.fams["F-boot"] for p in ds.positions])
    lab = np.column_stack([p.cal.ref.label(*p.cal.frame.apply(p.raw)) for p in ds.positions]).astype(np.int8)
    return {"B0": _measure(store, f"B0_{ds.name}.bin", b0, N, m, keep),
            "B1_zstd": _measure(store, f"B1_{ds.name}.bin", b1_blobs(lab), N, m, keep),
            "B1_bitarray": bits_per_shot_position(b1_bitarray_bytes(N, m), N, m)}
