"""Decode a stored raw job result (data/raw/<job_id>/) and run the pre-registered D0 acceptance checks (PREREG §D0).

Reads ONLY the stored files (never the network). Writes data/raw/<job_id>/decoded/<pub>_<circuit>_<register>.npy (as
returned: complex for kerneled, packed uint8 BitArray for classified) and decoded/checks.json with explicit pass / fail
flags:
  d0_pass_shape     every expected PUB present; kerneled registers complex, shape (4096, 16), all finite; classified
                    registers BitArray with 16 bits, 2 bytes per shot, 4096 shots; the classical-bit -> physical-qubit
                    map recomputed from the stored QPY circuits equals the recorded map
  d0_pass_clusters  (kerneled only) every position: LDA assignment fidelity >= 0.90 and |mu1 - mu0| / sigma_pooled >= 3,
                    final positions from the C0/C1 pair, mid-circuit positions from pooled rounds 1-2 of the same qubit
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHOTS = 4096
K = 16


def _cluster_metrics(z0: np.ndarray, z1: np.ndarray) -> dict:
    from iqstore.family import fit_lda
    from iqstore.frame import fit_frame

    fr = fit_frame(z0, z1)
    x0, y0 = fr.apply(z0)
    x1, y1 = fr.apply(z1)
    h = fit_lda(x0, y0, x1, y1)
    fid = 1.0 - 0.5 * (np.mean(h.label(x0, y0)) + np.mean(~h.label(x1, y1)))
    a = np.column_stack([x0, y0])
    b = np.column_stack([x1, y1])
    s = (np.cov(a, rowvar=False) * (len(a) - 1) + np.cov(b, rowvar=False) * (len(b) - 1)) / (len(a) + len(b) - 2)
    d = b.mean(axis=0) - a.mean(axis=0)
    sep = float(np.sqrt(d @ np.linalg.solve(s, d)))  # Mahalanobis distance = |mu1 - mu0| / sigma_pooled
    return {"lda_fidelity": float(fid), "separation": sep,
            "pass": bool(fid >= 0.90 and sep >= 3.0)}


def decode(job_dir: pathlib.Path) -> dict:
    from qiskit import qpy
    from qiskit_ibm_runtime import RuntimeDecoder

    from collect.submit import measurement_map

    meta = json.loads((job_dir / "job_metadata.json").read_text(encoding="utf-8"))
    result = json.loads((job_dir / "result_runtime_encoded.json").read_text(encoding="utf-8"), cls=RuntimeDecoder)
    with open(job_dir / "circuits.qpy", "rb") as f:
        circs = qpy.load(f)
    out_dir = job_dir / "decoded"
    out_dir.mkdir(exist_ok=True)
    names = list(meta["measurement_map"].keys())  # circuit names in PUB order
    mt = meta["options"]["meas_type"]
    checks: dict = {"job_id": meta["job_id"], "plan": meta["plan"], "meas_type": mt, "pubs": [], "problems": []}
    expect_pubs = 3 if meta["plan"].startswith("D0") else 2
    if len(result) != expect_pubs or len(circs) != expect_pubs:
        checks["problems"].append(f"expected {expect_pubs} PUBs, got {len(result)} results / {len(circs)} circuits")
    for c in circs:
        if measurement_map(c) != meta["measurement_map"].get(c.name):
            checks["problems"].append(f"measurement map of {c.name} differs from the stored QPY circuit")
    arrays: dict = {}
    for pub_idx, pub in enumerate(result):
        cname = names[pub_idx] if pub_idx < len(names) else f"pub{pub_idx}"
        entry = {"circuit": cname, "registers": {}}
        for reg, qubits in meta["measurement_map"].get(cname, {}).items():
            val = getattr(pub.data, reg, None)
            if val is None:
                checks["problems"].append(f"{cname}.{reg} missing")
                continue
            if hasattr(val, "array") and hasattr(val, "num_bits"):  # BitArray (classified)
                arr = np.asarray(val.array)
                info = {"type": "BitArray", "shape": list(arr.shape), "num_bits": int(val.num_bits),
                        "num_shots": int(val.num_shots), "bytes_per_shot": int(arr.shape[-1])}
                if (mt != "classified" or info["num_bits"] != K or info["bytes_per_shot"] != 2
                        or info["num_shots"] != SHOTS or info["shape"] != [SHOTS, 2]):
                    checks["problems"].append(f"{cname}.{reg}: unexpected BitArray {info}")
            else:
                arr = np.asarray(val)
                info = {"type": "ndarray", "dtype": str(arr.dtype), "shape": list(arr.shape),
                        "all_finite": bool(np.all(np.isfinite(arr))), "nbytes": int(arr.nbytes),
                        "is_complex": bool(np.iscomplexobj(arr))}
                if not (mt == "kerneled" and info["is_complex"] and info["shape"] == [SHOTS, K] and info["all_finite"]):
                    checks["problems"].append(f"{cname}.{reg}: unexpected array {info}")
                arrays[(cname, reg)] = arr
            info["physical_qubits"] = qubits
            np.save(out_dir / f"{pub_idx}_{cname}_{reg}.npy", arr)
            entry["registers"][reg] = info
        checks["pubs"].append(entry)
    checks["d0_pass_shape"] = len(checks["problems"]) == 0
    if mt == "kerneled" and checks["d0_pass_shape"]:
        per = []
        c0 = arrays[("C0_all0", "meas")]
        c1 = arrays[("C1_all1", "meas")]
        for i in range(K):
            per.append({"position": f"meas[{i}]", "context": "final"} | _cluster_metrics(c0[:, i], c1[:, i]))
        if ("C2_mid3", "r1") in arrays:
            r1, r2 = arrays[("C2_mid3", "r1")], arrays[("C2_mid3", "r2")]
            for i in range(K):
                z0, z1 = (r2[:, i], r1[:, i]) if i % 2 == 0 else (r1[:, i], r2[:, i])
                per.append({"position": f"r1/r2[{i}]", "context": "mid-circuit"} | _cluster_metrics(z0, z1))
        checks["clusters"] = per
        checks["d0_pass_clusters"] = all(p["pass"] for p in per)
    (out_dir / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    return checks


def main() -> None:
    jd = pathlib.Path(sys.argv[1])
    if not jd.is_absolute():
        jd = ROOT / jd
    c = decode(jd)
    print(json.dumps({k: v for k, v in c.items() if k not in ("pubs", "clusters")}, indent=1))


if __name__ == "__main__":
    main()
