"""Free (zero-QPU-second) probes of the IBM account: usage, backend configuration, today's calibration snapshot.

Writes qpu/probe_<UTC date>.json and data/calib/<backend>_<UTC timestamp>_properties.json.
The API token is read from the saved account in ~/.qiskit/ and is never printed or written.

Usage: python collect/probe.py [--backend ibm_cleveland]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _jsonable(x):
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, (set, tuple)):
        return list(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_cleveland")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService, __version__ as rt_version

    service = QiskitRuntimeService()  # saved account; token never touched here
    now = dt.datetime.now(dt.timezone.utc)
    out: dict = {"utc": now.isoformat(), "qiskit_ibm_runtime": rt_version, "backend": args.backend}

    try:
        out["usage"] = service.usage()
    except Exception as e:  # noqa: BLE001 - record and continue
        out["usage_error"] = repr(e)
    out["backends_visible"] = [b.name for b in service.backends()]

    backend = service.backend(args.backend)
    cfg = backend.configuration()
    keys = [
        "meas_levels", "meas_kernels", "discriminators", "rep_delay_range", "default_rep_delay", "max_shots",
        "max_experiments", "dynamic_reprate_enabled", "supported_features", "supported_instructions", "n_qubits",
        "basis_gates", "processor_type", "meas_map",
    ]
    out["configuration"] = {k: getattr(cfg, k, None) for k in keys}
    out["meas_map_len"] = len(getattr(cfg, "meas_map", []) or [])

    props = backend.properties()
    pdict = props.to_dict()
    calib_dir = ROOT / "data" / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    calib_path = calib_dir / f"{args.backend}_{now.strftime('%Y%m%dT%H%M%SZ')}_properties.json"
    calib_path.write_text(json.dumps(pdict, default=_jsonable, indent=1), encoding="utf-8")
    out["properties_path"] = str(calib_path.relative_to(ROOT))
    out["properties_last_update"] = _jsonable(props.last_update_date)

    # per-qubit readout summary
    ro = []
    for q in range(cfg.n_qubits):
        try:
            ro.append({
                "qubit": q,
                "readout_error": props.readout_error(q),
                "prob_meas1_prep0": props.qubit_property(q).get("prob_meas1_prep0", (None,))[0],
                "prob_meas0_prep1": props.qubit_property(q).get("prob_meas0_prep1", (None,))[0],
                "readout_length": props.readout_length(q),
                "operational": props.is_qubit_operational(q),
            })
        except Exception as e:  # noqa: BLE001
            ro.append({"qubit": q, "error": repr(e)})
    out["readout"] = ro

    # measured-instruction durations (for t_circ in the cost model)
    try:
        tgt = backend.target
        out["measure_duration_s_q0"] = tgt["measure"][(0,)].duration
        out["x_duration_s_q0"] = tgt["x"][(0,)].duration
    except Exception as e:  # noqa: BLE001
        out["target_error"] = repr(e)

    out_path = ROOT / "qpu" / f"probe_{now.strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(out, default=_jsonable, indent=1), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)} and {calib_path.relative_to(ROOT)}")
    print("usage:", json.dumps(out.get("usage", out.get("usage_error")), default=_jsonable))
    print("meas_levels:", out["configuration"]["meas_levels"], "| backends:", out["backends_visible"])


if __name__ == "__main__":
    main()
