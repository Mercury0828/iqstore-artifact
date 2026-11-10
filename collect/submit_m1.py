"""Tier-M1 job submission: one job per campaign epoch (qpu/M1_estimate.md v3; prereg/M1.md + AMENDMENT_C1_cadence).

SAFETY RULES (code-enforced; same mechanisms as collect/submit.py, reused from it):
  * dry-run by default: builds and checks the day's circuits, prints the plan and the projected cost, submits nothing;
  * refuses to submit without `--approved M1 --cap 600`, AND unless sha256(qpu/M1_estimate.md) equals APPROVED_SHA
    (the approved hash; None = not approved yet -> live mode refused);
  * the exclusive ledger lock, plan claim + persisted inputs before `sampler.run`, payload-first archive and the
    "never auto-resubmit" rule are the ones of collect/submit.py;
  * CHARGING per M1 job: final quantum seconds once billed (> 0), else the reservation; unknown -> the whole cap;
  * CEILING: every job runs with max_execution_time = floor(600 - charged(all other M1 jobs)); a job is refused if
    the ceiling is below its projected cost (measured all-in rate 407 us/shot + 1 s billing rounding);
  * CADENCE (v3): a new epoch is claimed only if >= MIN_GAP (110 min) passed since the claim timestamp of the
    latest M1 row; plan id M1-YYYYMMDDTHHMM from the SAME timestamp (taken under the lock) that is written to the ledger
    row; the two v2 rows claimed before CUTOVER keep their ids M1-YYYYMMDD; gap, campaign epoch and ledger state are
    re-validated UNDER the lock right before the claim; at most 14 campaign epochs ("day" below = epoch);
  * an outstanding `submitted` M1 job (any date) is resumed first, before any new work (one action per invocation);
  * WHOLE-CAMPAIGN budget: a new day is admitted only if charged + projected(all remaining days of the frozen schedule)
    <= 600 s, with the projection at max(measured D0 rate, highest per-shot rate billed so far in M1) + 1 s per job;
    otherwise STOP and report;
  * a job is `ok` only after the archive AND the structural acceptance check of prereg/M1.md section 2
    (decode_m1.pass_shape); weak clusters are reported, never a reason to re-run;
  * the qubit set is chosen once (first job) by the pre-registered rule and frozen with its couplers and the
    repetition-code path in qpu/M1_qubits.json AND inside every plan's persisted options; if any M1 row exists, the
    file must exist and equal the freeze stored with the first plan (never re-selected); a day whose circuits leave the
    set's healthy couplers or use an instruction the runtime Target does not support is refused BEFORE the claim.
The API token is never read, printed or written here (the saved account in ~/.qiskit/ is used).

Usage:
  python collect/submit_m1.py                              # dry run for today's campaign day
  python collect/submit_m1.py --approved M1 --cap 600      # live
  python collect/submit_m1.py --reconcile --approved M1 --cap 600
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collect import m1_circuits as M  # noqa: E402
from collect.submit import (PLANS, RAW, archive, final_quantum_seconds, ledger_lock, measurement_map,  # noqa: E402
                            notes_get, notes_set, persist_plan, read_ledger, sha256, write_ledger)

TIER = "M1"
CAP = 600.0
ESTIMATE = ROOT / "qpu" / "M1_estimate.md"
APPROVED_SHA = "d079d6cb1412aedf9f4ee4d308e6297c477095c95c9307b18f5a9aed0e45768e"  # sha256 of the job plan this script may run
MIN_GAP = dt.timedelta(minutes=110)  # AMENDMENT_C1 section 2
CUTOVER = dt.datetime(2026, 11, 8, 1, 0, tzinfo=dt.timezone.utc)  # rows claimed before it use the v2 id M1-YYYYMMDD
QUBITS_FILE = ROOT / "qpu" / "M1_qubits.json"
RATE_S_PER_SHOT = 5.0 / 12288  # measured all-in on D0 (qpu/M1_estimate.md)
BASIS = ["cz", "rz", "sx", "x", "id", "measure", "reset"]


def m1_charged(row: dict) -> float:
    final = notes_get(row, "final_quantum_s")
    if final is not None:
        return float(final)
    res = notes_get(row, "reserved_s")
    return float(res) if res is not None else CAP  # unknown reservation: fail closed


def m1_tier_charged(rows: list[dict]) -> float:
    return sum(m1_charged(r) for r in rows if r["tier"] == TIER)


def _row_shots(row: dict) -> int:
    return sum(int(x) for x in str(row.get("shots", "")).split("/") if x.strip().isdigit())


def projection_rate(rows: list[dict]) -> float:
    """max(measured D0 all-in rate, highest per-shot rate billed so far in M1)."""
    rate = RATE_S_PER_SHOT
    for r in rows:
        q = notes_get(r, "final_quantum_s")
        if r["tier"] == TIER and q is not None and _row_shots(r) > 0:
            rate = max(rate, float(q) / _row_shots(r))
    return rate


def day_shots(day: int) -> int:
    """Shots of campaign day `day` from the frozen schedule (names only; no device needed)."""
    names = ["CAL0", "CAL1", "CALM"] + [M.TARGET_NAMES[i] for i in M.schedule()[day - 1]]
    return sum(M.shots_for(n) for n in names)


def remaining_projection(day: int, rate: float) -> float:
    return sum(day_shots(d) * rate + 1.0 for d in range(day, M.N_DAYS + 1))


SERVER_TIME_URLS = ("https://quantum.cloud.ibm.com/", "https://iam.cloud.ibm.com/identity/keys")


def server_now() -> dt.datetime | None:
    """Current UTC time from the HTTP Date header of IBM's endpoints (independent of the local clock). Any HTTP status carries a Date header; None if no endpoint answers (the caller then refuses)."""
    import email.utils
    import urllib.error
    import urllib.request

    for u in SERVER_TIME_URLS:
        try:
            try:
                h = urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=30).headers
            except urllib.error.HTTPError as e:
                h = e.headers
            t = email.utils.parsedate_to_datetime(h["Date"])
            if t.tzinfo is not None:
                return t.astimezone(dt.timezone.utc)
        except Exception:  # noqa: BLE001 - try the next endpoint
            continue
    return None


def server_gap_ok(service, m1_rows: list[dict]) -> tuple[bool, str]:
    """Cadence gap measured on IBM's clocks: server-side creation time of the latest M1 job vs the server's current time.
    Both local and server gaps must reach MIN_GAP; a local clock jump forward therefore cannot shorten the gap."""
    if not m1_rows:
        return True, "first epoch"
    latest = max(m1_rows, key=lambda r: dt.datetime.fromisoformat(r["date_utc"]))
    if not latest["job_id"]:
        return False, f"latest M1 row {latest['purpose']} has no job id"
    created = service.job(latest["job_id"]).creation_date
    now = server_now()
    if created is None or created.tzinfo is None or now is None:
        return False, "server time or the latest job's server-side creation time unavailable"
    gap = now - created.astimezone(dt.timezone.utc)
    return gap >= MIN_GAP, f"server-side gap {gap} since job {latest['job_id']}"


def plan_id(t: dt.datetime) -> str:
    return "M1-" + t.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M")


def ledger_state(rows: list[dict]):
    """(campaign epoch, unresolved M1 rows, all M1 rows, latest claim timestamp or None)."""
    m1 = [r for r in rows if r["tier"] == TIER]
    day = sum(1 for r in m1 if r["status"] == "ok") + 1
    # a row's plan id is derived from its claim timestamp by construction (same timestamp); refuse inconsistent history
    claims = []
    for r in m1:
        t = dt.datetime.fromisoformat(r["date_utc"])
        if t.tzinfo is None:
            raise SystemExit(f"REFUSED: ledger row {r['purpose']} has a claim timestamp without time zone")
        legacy = t < CUTOVER and r["purpose"] == "M1-" + t.astimezone(dt.timezone.utc).strftime("%Y%m%d")
        if r["purpose"] != plan_id(t) and not legacy:
            raise SystemExit(f"REFUSED: ledger row {r['purpose']} does not match its claim timestamp {r['date_utc']}")
        claims.append(t)
    unresolved = [r for r in m1 if r["status"] != "ok"]
    return day, unresolved, m1, (max(claims) if claims else None)


def load_freeze(m1_rows: list[dict], props: dict | None, target, live: bool) -> dict:
    """The frozen set. Fresh selection only if NO M1 row exists; otherwise the file must exist and equal the freeze
    persisted with the first M1 plan."""
    if m1_rows:
        if not QUBITS_FILE.exists():
            raise SystemExit("REFUSED: M1 rows exist but qpu/M1_qubits.json is missing - restore it from the first "
                             "plan's persisted options (never re-select mid-campaign)")
        fz = json.loads(QUBITS_FILE.read_text(encoding="utf-8"))
        first = sorted(m1_rows, key=lambda r: r["date_utc"])[0]
        cands = [PLANS / first["purpose"] / "plan.json"]
        if first.get("raw_result_path"):
            cands.append(ROOT / first["raw_result_path"] / "plan.json")
        pj = next((c for c in cands if c.exists()), None)
        if pj is None:
            raise SystemExit("REFUSED: the first M1 plan's persisted inputs (plan.json) are missing - the freeze cannot "
                             "be verified; restore them before any further collection")
        stored = json.loads(pj.read_text(encoding="utf-8"))["options"].get("freeze")
        keys = ("qubits", "edges_virtual", "path_virtual")
        if stored is None or any(stored.get(k) != fz.get(k) for k in keys):
            raise SystemExit("REFUSED: qpu/M1_qubits.json differs from the freeze stored with the first M1 plan")
        return fz
    if QUBITS_FILE.exists():
        raise SystemExit("REFUSED: qpu/M1_qubits.json exists but no M1 row does - remove it only after diagnosis")
    qset = M.choose_connected_set(props, target=target)
    edges = M.set_edges(props, qset)
    fz = {"qubits": qset, "edges_virtual": [list(e) for e in edges], "path_virtual": M.longest_path(edges, len(qset)),
          "rule": "greedy growth by readout error over CZ couplers with error < 5 %; lowest mean readout error",
          "chosen_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    if live:
        QUBITS_FILE.write_text(json.dumps(fz, indent=1), encoding="utf-8")
    return fz


def build_frozen_day(props: dict, fz: dict, day: int, n_total: int, target=None):
    """Circuits of campaign day `day` with the FROZEN couplers and path (identical logical circuits across days)."""
    qset = fz["qubits"]
    edges = [tuple(e) for e in fz["edges_virtual"]]
    path = fz["path_virtual"]
    if len(path) < M.REP_PATH:
        raise SystemExit(f"REFUSED: frozen path {len(path)} < {M.REP_PATH}")
    targets = M.target_list(edges, path)
    logical = M.cal_circuits() + [targets[i] for i in M.schedule()[day - 1]]
    out = []
    for qc in logical:
        e = M.embed(M.transpile_in_set(qc, edges, BASIS), qset, n_total)
        probs = M.check_embedded(e, qset, props, target)  # TODAY's couplers and Target: a degraded day is refused
        if probs:
            raise SystemExit("REFUSED (before any claim): " + "; ".join(probs[:5]))
        out.append(e)
    return out


def finish(job, plan: str, service) -> str:
    """Archive (payload first), then the structural acceptance check; `ok` only if both pass."""
    from collect.decode_m1 import decode

    try:
        done = RAW / job.job_id() / "decoded" / "checks.json"
        if done.exists():  # a previous invocation archived and decoded this job completely: reuse, never rewrite
            d = done.parent.parent
            chk = json.loads(done.read_text(encoding="utf-8"))
            print(f"{plan}: archive and acceptance check already complete in {d} -> reused")
        else:
            d = archive(job, plan)
            chk = decode(d)
        status = "ok" if chk["pass_shape"] else f"failed: acceptance {chk['problems'][:3]}"[:300]
    except Exception as e:  # noqa: BLE001 - technical failure: record, never auto-resubmit
        d = RAW / job.job_id()
        status = f"failed: {e!r}"[:300]
    try:
        q = final_quantum_seconds(job)
    except Exception:  # noqa: BLE001 - billing is reconciled later (--reconcile); never lose the completion transition
        q = None
    try:
        usage_after = service.usage().get("usage_consumed_seconds")
    except Exception as e:  # noqa: BLE001 - optional telemetry
        usage_after = f"unavailable: {type(e).__name__}"
    rows = read_ledger()
    for r in rows:
        if r["purpose"] == plan:
            r["usage_after_s"] = usage_after
            r["status"] = status
            r["raw_result_path"] = d.relative_to(ROOT).as_posix() if d.exists() else ""
            if q is not None:
                notes_set(r, "final_quantum_s", q)
            else:
                notes_set(r, "billing", "pending (reservation kept; run --reconcile later)")
    write_ledger(rows)
    print(f"{plan}: status {status}; final quantum seconds {q}; tier {TIER} charged {m1_tier_charged(rows):.2f} s")
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--approved", default=None)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--backend", default="ibm_cleveland")
    a = ap.parse_args()
    live = a.approved is not None or a.cap is not None
    if live:
        if not (a.approved == TIER and a.cap is not None and abs(a.cap - CAP) < 1e-9):
            print(f"REFUSED: live submission needs exactly --approved {TIER} --cap {CAP:g}")
            return 2
        have = sha256(ESTIMATE)
        if APPROVED_SHA is None or have != APPROVED_SHA:
            print(f"REFUSED: qpu/M1_estimate.md sha256 {have} is not the approved hash ({APPROVED_SHA})")
            return 2

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    if a.reconcile:
        if not live:
            print("reconcile needs --approved M1 --cap 600 (it writes the ledger)")
            return 2
        service = QiskitRuntimeService()
        with ledger_lock():
            rows = read_ledger()
            for r in rows:
                if r["tier"] == TIER and r["job_id"] and notes_get(r, "final_quantum_s") is None:
                    q = final_quantum_seconds(service.job(r["job_id"]))
                    if q is not None:
                        notes_set(r, "final_quantum_s", q)
            write_ledger(rows)
            print(f"tier {TIER} charged {m1_tier_charged(rows):.2f} s")
        return 0

    rows = read_ledger()
    day, unresolved, m1, last = ledger_state(rows)
    # 1) an outstanding submitted job (any date) is resumed first
    pending = [r for r in unresolved if r["status"] == "submitted" and r["job_id"]]
    if pending:
        r = pending[0]
        print(f"{r['purpose']}: resuming job {r['job_id']} (persisted inputs in {PLANS / r['purpose']})")
        if not live:
            print("DRY RUN - would resume")
            return 0
        service = QiskitRuntimeService()
        with ledger_lock():
            # re-validate under the lock: another invocation may have finished this job meanwhile (polling)
            cur = [x for x in read_ledger() if x["purpose"] == r["purpose"]]
            if not cur or cur[0]["status"] != "submitted" or cur[0]["job_id"] != r["job_id"]:
                print(f"{r['purpose']}: state changed under the lock ({cur[0]['status'] if cur else 'missing'}) "
                      f"-> nothing to do")
                return 0
            return 0 if finish(service.job(r["job_id"]), r["purpose"], service) == "ok" else 4
    # 2) any other unresolved row blocks (claimed without job id, failed): diagnose first
    if unresolved:
        r = unresolved[0]
        print(f"REFUSED: M1 row {r['purpose']} in state '{r['status']}' (job '{r['job_id']}') - diagnose and record "
              f"it first")
        return 4
    if day > M.N_DAYS:
        print(f"campaign complete ({M.N_DAYS} epochs)")
        return 0
    if last is not None and dt.datetime.now(dt.timezone.utc) - last < MIN_GAP:
        print(f"epoch {day} not due before {(last + MIN_GAP).isoformat(timespec='seconds')} -> nothing to do")
        return 0
    plan = f"M1-epoch{day}"  # provisional label; the plan id is fixed from the claim timestamp under the lock
    service = QiskitRuntimeService()
    rate = projection_rate(rows)
    need = remaining_projection(day, rate)
    charged = m1_tier_charged(rows)
    print(f"tier {TIER}: charged {charged:.2f} s of cap {CAP:g} s; campaign day {day}; remaining schedule projected "
          f"{need:.1f} s at {rate * 1e6:.0f} us/shot")
    if charged + need > CAP:
        print(f"STOP: the remaining frozen schedule ({need:.1f} s) no longer fits the cap ({CAP - charged:.1f} s "
              f"left) - report; the cap is never raised and the corpus never cut by the session")
        return 3

    backend = service.backend(a.backend)
    target = backend.target
    n_total = backend.configuration().n_qubits
    props = json.loads(json.dumps(backend.properties().to_dict(), default=str))
    fz = load_freeze(m1, props, target, live=False)
    circuits = build_frozen_day(props, fz, day, n_total, target)
    projected = sum(M.shots_for(c.name) for c in circuits) * rate + 1.0
    print(f"{plan}: qubits {fz['qubits']}; PUBs {[(c.name, M.shots_for(c.name)) for c in circuits]}; "
          f"projected {projected:.1f} s")
    if not live:
        print(f"{plan}: DRY RUN - nothing submitted")
        return 0
    with ledger_lock():
        # ONE timestamp, taken under the lock, defines the claim: the plan id is derived from it and the same timestamp is
        # written to the ledger row; the cadence gap is re-checked against it
        now = dt.datetime.now(dt.timezone.utc)
        rows = read_ledger()
        day2, unresolved2, m1_2, last2 = ledger_state(rows)
        plan = plan_id(now)
        if day2 != day or unresolved2 or any(r["purpose"] == plan for r in m1_2):
            print("REFUSED: campaign epoch / ledger changed during preparation - run again")
            return 4
        if last2 is not None and now - last2 < MIN_GAP:
            print("REFUSED: cadence gap not reached at claim time - run again later")
            return 4
        ok, why = server_gap_ok(service, m1_2)
        if not ok:
            print(f"not due by IBM server time ({why}) -> nothing claimed; run again later")
            return 0
        print(f"cadence check: {why}")
        charged = m1_tier_charged(rows)
        if charged + remaining_projection(day, projection_rate(rows)) > CAP:
            print("STOP: the remaining schedule no longer fits the cap - report")
            return 3
        ceiling = math.floor(CAP - charged)
        if ceiling < projected:
            print(f"REFUSED {plan}: remaining ceiling {ceiling} s < projected {projected:.1f} s - report")
            return 3
        fz = load_freeze(m1_2, props, target, live=True)  # freezes the set on the first live day only
        calib = ROOT / "data" / "calib" / f"{a.backend}_{now.strftime('%Y%m%dT%H%M%SZ')}_properties.json"
        calib.parent.mkdir(parents=True, exist_ok=True)
        calib.write_text(json.dumps(props, indent=1), encoding="utf-8")
        calib_rel = calib.relative_to(ROOT).as_posix()
        shots = {c.name: M.shots_for(c.name) for c in circuits}
        options = {"shots_per_pub": shots, "meas_type": "kerneled", "twirling": "off", "dd": "off",
                   "max_execution_time": ceiling, "campaign_day": day, "campaign_epoch": day,
                   "cadence": "epoch >= 110 min after the previous claim (AMENDMENT_C1)",
                   "estimate_sha256": APPROVED_SHA,
                   "schedule_seed": M.schedule_seed(), "freeze": fz}
        persist_plan(plan, circuits, {c.name: measurement_map(c) for c in circuits}, options, calib_rel, fz["qubits"])
        row = {"date_utc": now.isoformat(), "job_id": "", "tier": TIER, "purpose": plan,
               "circuits": "+".join(c.name for c in circuits), "shots": "/".join(str(shots[c.name]) for c in circuits),
               "meas_type": "kerneled", "usage_before_s": service.usage().get("usage_consumed_seconds"),
               "calibration_snapshot": calib_rel, "status": "claimed",
               "notes": f"day={day};reserved_s={ceiling}"}
        write_ledger(rows + [row])
        sampler = SamplerV2(mode=backend)
        sampler.options.execution.meas_type = "kerneled"
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.max_execution_time = int(ceiling)
        job = sampler.run([(c, None, shots[c.name]) for c in circuits])
        rows = read_ledger()
        for r in rows:
            if r["purpose"] == plan:
                r["job_id"] = job.job_id()
                r["status"] = "submitted"
        write_ledger(rows)
        print(f"{plan}: submitted job {job.job_id()} with max_execution_time={ceiling}")
        status = finish(job, plan, service)
    if status != "ok":
        print("STOP: technical failure recorded; diagnose before any re-submission")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
