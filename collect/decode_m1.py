"""Decode a stored M1 job (data/raw/<job_id>/) and run the prereg/M1.md §2 acceptance checks. Reads only stored files.

Writes data/raw/<job_id>/decoded/<pub>_<circuit>_<register>.npy (complex, as returned) and decoded/checks.json:
  pass_shape     every PUB present; every register complex of shape (shots_of_pub, register size), all finite; the
                 classical-bit -> physical-qubit map recomputed from the stored QPY equals the recorded map
  clusters       per qubit: "final" from CAL0/CAL1 `meas`; "mid-circuit" from CALM rounds 1-2 (even index prepared |1>
                 in round 1, |0> in round 2; odd index the reverse); LDA fidelity >= 0.90 and separation >= 3
                 (failures listed, never dropped)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collect.decode import _cluster_metrics  # noqa: E402


def _safe_metrics(z0, z1) -> dict:
    """Per-position cluster metrics; an undefined frame / LDA / separation is recorded as a failure with its reason
    (the other positions are still assessed and the report is always written)."""
    try:
        return _cluster_metrics(z0, z1)
    except Exception as e:  # noqa: BLE001
        return {"lda_fidelity": None, "separation": None, "pass": False, "error": repr(e)[:200]}


def decode(job_dir: pathlib.Path) -> dict:
    from qiskit import qpy
    from qiskit_ibm_runtime import RuntimeDecoder

    from collect.submit import measurement_map

    meta = json.loads((job_dir / "job_metadata.json").read_text(encoding="utf-8"))
    result = json.loads((job_dir / "result_runtime_encoded.json").read_text(encoding="utf-8"), cls=RuntimeDecoder)
    with open(job_dir / "circuits.qpy", "rb") as f:
        circs = qpy.load(f)
    shots = meta["options"]["shots_per_pub"]
    out_dir = job_dir / "decoded"
    out_dir.mkdir(exist_ok=True)
    names = meta["circuit_names"]
    chk: dict = {"job_id": meta["job_id"], "plan": meta["plan"], "day": meta["options"].get("campaign_day"),
                 "qubits": meta["qubits"], "pubs": [], "problems": []}
    if len(result) != len(names) or len(circs) != len(names):
        chk["problems"].append(f"expected {len(names)} PUBs, got {len(result)} results / {len(circs)} circuits")
    for c in circs:
        if measurement_map(c) != meta["measurement_map"].get(c.name):
            chk["problems"].append(f"measurement map of {c.name} differs from the stored QPY circuit")
    arrays: dict = {}
    for i, pub in enumerate(result):
        cname = names[i] if i < len(names) else f"pub{i}"
        entry = {"circuit": cname, "registers": {}}
        for reg, qubits in meta["measurement_map"].get(cname, {}).items():
            val = getattr(pub.data, reg, None)
            if val is None:
                chk["problems"].append(f"{cname}.{reg} missing")
                continue
            arr = np.asarray(val)
            info = {"dtype": str(arr.dtype), "shape": list(arr.shape), "all_finite": bool(np.all(np.isfinite(arr))),
                    "physical_qubits": qubits}
            if not (np.iscomplexobj(arr) and info["shape"] == [shots[cname], len(qubits)] and info["all_finite"]):
                chk["problems"].append(f"{cname}.{reg}: unexpected array {info}")
            arrays[(cname, reg)] = arr
            np.save(out_dir / f"{i}_{cname}_{reg}.npy", arr)
            entry["registers"][reg] = info
        chk["pubs"].append(entry)
    chk["pass_shape"] = not chk["problems"]
    per = []
    if chk["pass_shape"]:
        q_final = meta["measurement_map"]["CAL0"]["meas"]
        c0, c1 = arrays[("CAL0", "meas")], arrays[("CAL1", "meas")]
        for k, q in enumerate(q_final):
            per.append({"qubit": q, "context": "final"} | _safe_metrics(c0[:, k], c1[:, k]))
        r1, r2 = arrays[("CALM", "r1")], arrays[("CALM", "r2")]
        for k, q in enumerate(meta["measurement_map"]["CALM"]["r1"]):
            z0, z1 = (r2[:, k], r1[:, k]) if k % 2 == 0 else (r1[:, k], r2[:, k])
            per.append({"qubit": q, "context": "mid-circuit"} | _safe_metrics(z0, z1))
    chk["clusters"] = per
    chk["cluster_failures"] = [p for p in per if not p["pass"]]
    (out_dir / "checks.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
    return chk


def main() -> None:
    jd = pathlib.Path(sys.argv[1])
    if not jd.is_absolute():
        jd = ROOT / jd
    c = decode(jd)
    print(json.dumps({k: v for k, v in c.items() if k not in ("pubs", "clusters")}, indent=1))


if __name__ == "__main__":
    main()
