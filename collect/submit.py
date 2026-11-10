"""Tier-P0 job submission for D0 and D3.

SAFETY RULES (code-enforced):
  * dry-run by default: builds the circuits, prints the plan and the cost projection, submits nothing;
  * refuses to submit without `--approved P0 --cap 60`;
  * an exclusive lock file (qpu/.submit.lock) serializes all ledger transitions; a stale lock is never auto-removed;
  * CHARGING: a job counts against the cap with its final per-job quantum seconds once billing is final (> 0);
    until then it counts with its RESERVATION = the server-side `max_execution_time` it was submitted with;
  * CEILING: every job is submitted with `max_execution_time = floor(cap - charged(all other jobs))`, so the service
    cancels it before the tier cap can be exceeded; a job is refused if that ceiling is < projected cost;
  * IDEMPOTENT per plan id (D0-K, D0-C, D3-YYYYMMDD): before `sampler.run` the plan is CLAIMED in the ledger and its
    immutable inputs (QPY circuits, measurement map, options, calibration snapshot, qubits) are persisted under
    data/raw/_plans/<plan>/; any existing row for a plan (claimed / failed) blocks a new submission; a `submitted`
    plan is RESUMED by job id with the persisted inputs; a `claimed` row without a job id is ambiguous (crash between
    claim and run) and blocks until a human reconciles it;
  * after completion the verbatim service payload is fetched and written FIRST, then decoded; a failed archive is
    recorded as a failure (never `ok`);
  * measurement twirling and dynamical decoupling are off (kerneled + measurement twirling fail together, issue #2958).
The API token is never read, printed or written here (the saved account in ~/.qiskit/ is used).

Usage:
  python collect/submit.py --plan D0                                # dry run
  python collect/submit.py --plan D0 --approved P0 --cap 60
  python collect/submit.py --plan D3 --approved P0 --cap 60         # one calibration pair for today (UTC date)
  python collect/submit.py --reconcile --approved P0 --cap 60       # update final usage of jobs with pending billing
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "qpu" / "ledger.csv"
LOCK = ROOT / "qpu" / ".submit.lock"
RAW = ROOT / "data" / "raw"
PLANS = RAW / "_plans"
MANIFEST = ROOT / "data" / "MANIFEST.csv"
TIER = "P0"
APPROVED_CAP = 60.0
SHOTS = 4096
PER_SHOT_S = 260e-6  # cost model per shot (qpu/P0_estimate.md)
N_QUBITS = 16
FIELDS = ["date_utc", "job_id", "tier", "purpose", "circuits", "shots", "meas_type", "usage_before_s", "usage_after_s",
          "calibration_snapshot", "raw_result_path", "status", "notes"]


# ------------------------------------------------------------------------------------------------ ledger + lock

@contextlib.contextmanager
def ledger_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"REFUSED: lock {LOCK} exists (another submission running, or a crash). Check the ledger and "
                         f"the IBM job list by hand, then remove the lock.")
    try:
        os.write(fd, f"{os.getpid()} {dt.datetime.now(dt.timezone.utc).isoformat()}".encode())
        os.close(fd)
        yield
    finally:
        LOCK.unlink(missing_ok=True)


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_ledger(rows: list[dict]) -> None:
    tmp = LEDGER.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(LEDGER)


def notes_get(row: dict, key: str):
    for part in row.get("notes", "").split(";"):
        part = part.strip()
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return None


def notes_set(row: dict, key: str, val) -> None:
    parts = [p for p in row.get("notes", "").split(";") if p.strip() and not p.strip().startswith(key + "=")]
    parts.append(f"{key}={val}")
    row["notes"] = ";".join(parts)


def charged(row: dict) -> float:
    """Final per-job quantum seconds if billing is final, else the reservation (server-side ceiling)."""
    final = notes_get(row, "final_quantum_s")
    if final is not None:
        return float(final)
    res = notes_get(row, "reserved_s")
    return float(res) if res is not None else APPROVED_CAP  # unknown reservation: fail closed


def tier_charged(rows: list[dict]) -> float:
    return sum(charged(r) for r in rows if r["tier"] == TIER)


def measured_t_job(rows: list[dict]) -> float | None:
    for r in rows:
        v = notes_get(r, "t_job_s")
        if v is not None:
            return float(v)
    return None


# ------------------------------------------------------------------------------------------------ circuits

def choose_qubits(props, n_qubits: int, k: int) -> list[int]:
    cand = []
    for q in range(n_qubits):
        try:
            if props.is_qubit_operational(q):
                cand.append((props.readout_error(q), q))
        except Exception:  # noqa: BLE001
            continue
    cand.sort()
    return [q for _, q in cand[:k]]


def build_circuits(n_total: int, qubits: list[int]):
    """C0 (all |0>), C1 (all |1>), C2 (3 rounds; rounds 1-2 mid-circuit). Built on physical qubit indices so the
    classical-bit -> physical-qubit map is known by construction (premise correction ii)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qr = QuantumRegister(n_total, "q")
    k = len(qubits)
    c0 = QuantumCircuit(qr, ClassicalRegister(k, "meas"), name="C0_all0")
    c0.measure([qr[q] for q in qubits], c0.cregs[0])
    c1 = QuantumCircuit(qr, ClassicalRegister(k, "meas"), name="C1_all1")
    for q in qubits:
        c1.x(qr[q])
    c1.measure([qr[q] for q in qubits], c1.cregs[0])
    r1, r2, r3 = ClassicalRegister(k, "r1"), ClassicalRegister(k, "r2"), ClassicalRegister(k, "r3")
    c2 = QuantumCircuit(qr, r1, r2, r3, name="C2_mid3")
    for i, q in enumerate(qubits):
        if i % 2 == 0:
            c2.x(qr[q])
    c2.measure([qr[q] for q in qubits], r1)
    for q in qubits:
        c2.x(qr[q])
    c2.measure([qr[q] for q in qubits], r2)
    for q in qubits:
        c2.x(qr[q])
    c2.measure([qr[q] for q in qubits], r3)
    return c0, c1, c2


def measurement_map(circ) -> dict:
    """register -> list of physical qubits, in classical-bit order (from the circuit's measure instructions)."""
    out: dict = {}
    for inst in circ.data:
        if inst.operation.name == "measure":
            q = circ.find_bit(inst.qubits[0]).index
            reg, idx = circ.find_bit(inst.clbits[0]).registers[0]
            out.setdefault(reg.name, {})[idx] = q
    return {r: [m[i] for i in sorted(m)] for r, m in out.items()}


# ------------------------------------------------------------------------------------------------ archive

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_add(path: pathlib.Path, job_id: str, kind: str) -> None:
    """Append a checksum row; idempotent: an identical (path, sha256) row is never added twice (a resumed archive of the
    same job rewrites identical files)."""
    new = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    digest = sha256(path)
    if not new:
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            if any(row.get("path") == rel and row.get("sha256") == digest for row in csv.DictReader(f)):
                return
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "sha256", "bytes", "job_id", "kind", "created_utc", "in_git"])
        w.writerow([rel, digest, path.stat().st_size, job_id, kind,
                    dt.datetime.now(dt.timezone.utc).isoformat(), "yes" if path.stat().st_size < 20e6 else "no"])


def persist_plan(plan: str, circuits, meas_maps, options: dict, calib_rel: str, qubits: list[int]) -> pathlib.Path:
    from qiskit import qpy

    d = PLANS / plan
    if d.exists():
        raise SystemExit(f"REFUSED: persisted inputs for {plan} already exist at {d} (ledger and plans disagree)")
    d.mkdir(parents=True)
    with open(d / "circuits.qpy", "wb") as f:
        qpy.dump(list(circuits), f)
    (d / "plan.json").write_text(json.dumps({"plan": plan, "measurement_map": meas_maps, "options": options,
                                             "calibration_snapshot": calib_rel, "qubits": qubits,
                                             "circuit_names": [c.name for c in circuits]}, indent=1),
                                 encoding="utf-8")
    return d


def load_plan(plan: str):
    from qiskit import qpy

    d = PLANS / plan
    with open(d / "circuits.qpy", "rb") as f:
        circuits = qpy.load(f)
    return circuits, json.loads((d / "plan.json").read_text(encoding="utf-8"))


def _terminal(job) -> bool:
    s = str(job.status()).upper()
    return any(t in s for t in ("DONE", "ERROR", "CANCELLED"))


def archive(job, plan: str) -> pathlib.Path:
    """Wait for the final state, write the verbatim payload FIRST, then the decoded result and metadata."""
    from qiskit_ibm_runtime import RuntimeEncoder

    jid = job.job_id()
    while not _terminal(job):
        time.sleep(10)
    d = RAW / jid
    d.mkdir(parents=True, exist_ok=True)
    pd = PLANS / plan
    for name in ("circuits.qpy", "plan.json"):
        shutil.copy2(pd / name, d / name)
        manifest_add(d / name, jid, "plan_inputs")
    meta = json.loads((pd / "plan.json").read_text(encoding="utf-8"))
    meta.update({"job_id": jid, "status": str(job.status()), "creation_date": str(getattr(job, "creation_date", ""))})
    try:
        meta["backend"] = job.backend().name
    except Exception:  # noqa: BLE001
        meta["backend"] = "ibm_cleveland"
    try:
        meta["metrics"] = job.metrics()
    except Exception as e:  # noqa: BLE001
        meta["metrics_error"] = repr(e)
    if "DONE" in str(job.status()).upper():
        raw_payload = job._api_client.job_results(job_id=jid)  # noqa: SLF001 - verbatim payload, read-only
        p = d / "result_raw_payload.json"
        p.write_text(raw_payload if isinstance(raw_payload, str) else json.dumps(raw_payload), encoding="utf-8")
        manifest_add(p, jid, "raw_payload")
    else:
        try:
            meta["error_message"] = job.error_message()
        except Exception as e:  # noqa: BLE001
            meta["error_message"] = repr(e)
    p = d / "job_metadata.json"
    p.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    manifest_add(p, jid, "job_metadata")
    if "DONE" not in str(job.status()).upper():
        raise RuntimeError(f"job ended in state {job.status()} (metadata archived)")
    result = job.result()  # decoding happens only after the verbatim payload is on disk
    p = d / "result_runtime_encoded.json"
    p.write_text(json.dumps(result, cls=RuntimeEncoder), encoding="utf-8")
    manifest_add(p, jid, "runtime_encoded_result")
    return d


def final_quantum_seconds(job) -> float | None:
    """Per-job quantum seconds if billing is final (> 0); None while pending (0 or unavailable)."""
    for getter in (lambda: job.usage(), lambda: job.metrics()["usage"]["quantum_seconds"]):
        try:
            v = getter()
            if v is not None and float(v) > 0:
                return float(v)
        except Exception:  # noqa: BLE001
            continue
    return None


def _finish(job, plan: str, service, compute: float):
    """Archive (payload first), record usage (final, or the reservation is kept), update the ledger."""
    try:
        d = archive(job, plan)
        status = "ok"
    except Exception as e:  # noqa: BLE001 - technical failure: record, never auto-resubmit
        d = RAW / job.job_id()
        status = f"failed: {e!r}"[:300]
    q = final_quantum_seconds(job)
    rows = read_ledger()
    for r in rows:
        if r["purpose"] == plan:
            r["usage_after_s"] = service.usage().get("usage_consumed_seconds")
            r["status"] = status
            r["raw_result_path"] = str(d.relative_to(ROOT)).replace("\\", "/") if d.exists() else ""
            if q is not None:
                notes_set(r, "final_quantum_s", q)
                if measured_t_job(rows) is None:
                    notes_set(r, "t_job_s", f"{max(0.0, q - compute):.3f}")
            else:
                notes_set(r, "billing", "pending (reservation kept; run --reconcile later)")
    write_ledger(rows)
    print(f"{plan}: status {status}; final quantum seconds {q}; tier charged {tier_charged(rows):.2f} s")
    return status, d


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["D0", "D3"])
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--approved", default=None)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--backend", default="ibm_cleveland")
    a = ap.parse_args()
    live = a.approved is not None or a.cap is not None
    if live and not (a.approved == TIER and a.cap is not None and abs(a.cap - APPROVED_CAP) < 1e-9):
        print(f"REFUSED: live submission needs exactly --approved {TIER} --cap {APPROVED_CAP:g}")
        return 2
    if not a.reconcile and a.plan is None:
        print("need --plan or --reconcile")
        return 2

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    if a.reconcile:
        if not live:
            print("reconcile needs --approved P0 --cap 60 (it writes the ledger)")
            return 2
        service = QiskitRuntimeService()
        with ledger_lock():
            rows = read_ledger()
            for r in rows:
                if r["job_id"] and notes_get(r, "final_quantum_s") is None:
                    q = final_quantum_seconds(service.job(r["job_id"]))
                    if q is not None:
                        notes_set(r, "final_quantum_s", q)
            write_ledger(rows)
            print(f"tier {TIER} charged {tier_charged(rows):.2f} s")
        return 0

    rows = read_ledger()
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    d0 = [r for r in rows if r["purpose"] == "D0-K" and r["status"] == "ok"]
    if a.plan == "D0":
        jobs = [("D0-K", "kerneled", 3), ("D0-C", "classified", 3)]
    else:
        if not d0:
            print("REFUSED: D3 needs the day-1 qubit set from a completed D0-K job")
            return 2
        if d0[0]["date_utc"][:10].replace("-", "") == today:
            print("SKIP: today's calibration pair is the D0-K job (day 1)")
            return 0
        jobs = [(f"D3-{today}", "kerneled", 2)]
    print(f"tier {TIER}: charged {tier_charged(rows):.2f} s of cap {APPROVED_CAP:g} s; t_job = {measured_t_job(rows)}")

    service = QiskitRuntimeService()
    backend = service.backend(a.backend)
    n_total = backend.configuration().n_qubits
    for plan, meas_type, ncirc in jobs:
        rows = read_ledger()
        prev = [r for r in rows if r["purpose"] == plan]
        if prev and prev[0]["status"] == "ok":
            print(f"{plan}: already done -> skip")
            continue
        if prev and not (prev[0]["status"] == "submitted" and prev[0]["job_id"]):
            print(f"REFUSED {plan}: ledger row in state '{prev[0]['status']}' (job id '{prev[0]['job_id']}') — "
                  f"diagnose and record the diagnosis before any re-submission")
            return 4
        compute = ncirc * SHOTS * PER_SHOT_S
        t_job = measured_t_job(rows)
        projected = compute * 1.3 + (t_job if t_job is not None else 2.0)
        if prev:  # resume a submitted job with its persisted inputs
            print(f"{plan}: resuming job {prev[0]['job_id']} (persisted inputs in {PLANS / plan})")
            if not live:
                print(f"{plan}: DRY RUN - would resume")
                continue
            with ledger_lock():
                status, _ = _finish(service.job(prev[0]["job_id"]), plan, service, compute)
            if status != "ok":
                return 4
            continue
        props = backend.properties()
        now = dt.datetime.now(dt.timezone.utc)
        calib = ROOT / "data" / "calib" / f"{a.backend}_{now.strftime('%Y%m%dT%H%M%SZ')}_properties.json"
        if a.plan == "D0":
            qubits = choose_qubits(props, n_total, N_QUBITS)
        else:
            qubits = json.loads((ROOT / d0[0]["raw_result_path"] / "plan.json").read_text(encoding="utf-8"))["qubits"]
        circuits = list(build_circuits(n_total, qubits))[:ncirc]
        print(f"{plan}: qubits {qubits}; {ncirc} circuits x {SHOTS} shots, meas_type={meas_type}; "
              f"projected {projected:.2f} s")
        if not live:
            print(f"{plan}: DRY RUN - nothing submitted")
            continue
        with ledger_lock():
            rows = read_ledger()
            if any(r["purpose"] == plan for r in rows):
                print(f"REFUSED {plan}: claimed concurrently")
                return 4
            ceiling = math.floor(APPROVED_CAP - tier_charged(rows))
            if ceiling < projected:
                print(f"REFUSED {plan}: remaining ceiling {ceiling} s < projected {projected:.2f} s")
                return 3
            calib.parent.mkdir(parents=True, exist_ok=True)
            calib.write_text(json.dumps(props.to_dict(), default=str, indent=1), encoding="utf-8")
            calib_rel = str(calib.relative_to(ROOT)).replace("\\", "/")
            options = {"default_shots": SHOTS, "meas_type": meas_type, "twirling": "off", "dd": "off",
                       "max_execution_time": ceiling}
            persist_plan(plan, circuits, {c.name: measurement_map(c) for c in circuits}, options, calib_rel, qubits)
            row = {"date_utc": now.isoformat(), "job_id": "", "tier": TIER, "purpose": plan,
                   "circuits": "+".join(c.name for c in circuits), "shots": SHOTS, "meas_type": meas_type,
                   "usage_before_s": service.usage().get("usage_consumed_seconds"), "calibration_snapshot": calib_rel,
                   "status": "claimed", "notes": f"qubits={'/'.join(map(str, qubits))};reserved_s={ceiling}"}
            write_ledger(rows + [row])
            sampler = SamplerV2(mode=backend)
            sampler.options.default_shots = SHOTS
            sampler.options.execution.meas_type = meas_type
            sampler.options.twirling.enable_gates = False
            sampler.options.twirling.enable_measure = False
            sampler.options.dynamical_decoupling.enable = False
            sampler.options.max_execution_time = int(ceiling)
            job = sampler.run(circuits)
            rows = read_ledger()
            for r in rows:
                if r["purpose"] == plan:
                    r["job_id"] = job.job_id()
                    r["status"] = "submitted"
            write_ledger(rows)
            print(f"{plan}: submitted job {job.job_id()} with max_execution_time={ceiling}")
            status, _ = _finish(job, plan, service, compute)
        if status != "ok":
            print("STOP: technical failure recorded; diagnose before any re-submission")
            return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
