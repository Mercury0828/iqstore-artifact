"""M2 data access (prereg/M2.md v4 §1, §2; prereg/M1.md §4; AMENDMENT_C1).

Loads M1 campaign epochs from the stored, decoded jobs. **DEV guard** (AMENDMENT_C1 §5, prereg/M2.md §1 [F3]): until
`prereg/M2_dev_freeze.json` is committed, only epochs 1-5 may be ingested. `load_epoch(e)` refuses any e > 5 unless
`allow_test=True`, and `allow_test=True` itself refuses unless the freeze file is committed and unmodified. Calibration PUBs and target
PUBs are read only through this module.

Per (epoch, physical qubit, context):
  final        frame and LDA reference from that epoch's CAL0 / CAL1 `meas`
  mid-circuit  frame and LDA reference from that epoch's CALM rounds 1-2 (even index prepared |1> in round 1 and |0>
               in round 2; odd index the reverse; collect/decode_m1.py)
Per dataset position: the F-boot family (prereg/M2.md §2: 200 bootstrap LDA refits of the calibration shots of that
context, STREAM 9, dataset_index = target index, extra = position index) and F-plan.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.workload import rng  # noqa: E402
from iqstore.family import Discriminator, Family, bootstrap_widths, family_from_widths, fit_lda  # noqa: E402
from iqstore.frame import Frame, fit_frame  # noqa: E402

DEV_EPOCHS = (1, 2, 3, 4, 5)
FREEZE = ROOT / "prereg" / "M2_dev_freeze.json"
F_PLAN = (0.1, 0.2)
N_BOOT = 200
MID_REGS_PREFIX = "a"  # R-MID ancilla registers a1..a10 are mid-circuit; `data` and `meas` are final


def ledger_jobs() -> dict[int, dict]:
    """Accepted M1 jobs keyed by campaign epoch (qpu/ledger.csv, status ok, notes day=<e>)."""
    out: dict[int, dict] = {}
    with open(ROOT / "qpu" / "ledger.csv", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 13 or row[2] != "M1" or row[11] != "ok":
                continue
            notes = dict(kv.split("=", 1) for kv in row[12].split(";") if "=" in kv)
            e = int(notes["day"])
            if e in out:
                raise RuntimeError(f"two accepted M1 jobs for epoch {e}")
            out[e] = {"epoch": e, "job_id": row[1], "plan": row[3], "claim_utc": row[0], "dir": ROOT / row[10]}
    return out


def workload() -> dict:
    return json.loads((ROOT / "prereg" / "M2_workload.json").read_text(encoding="utf-8"))


def _check_access(epoch: int, allow_test: bool) -> None:
    if epoch in DEV_EPOCHS:
        return
    if not allow_test:
        raise PermissionError(f"epoch {epoch} is TEST; refused (DEV guard)")
    if not freeze_committed():
        raise PermissionError("TEST access refused: prereg/M2_dev_freeze.json is not committed unmodified "
                              "(AMENDMENT_C1 section 5)")


def freeze_committed() -> bool:
    """The DEV freeze exists, is tracked by git, and equals its committed version (checked at every TEST access)."""
    import subprocess
    if not FREEZE.exists():
        return False
    rel = FREEZE.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, capture_output=True).returncode == 0
    staged = subprocess.run(["git", "diff", "--quiet", "--cached", "--", rel], cwd=ROOT, capture_output=True).returncode == 0
    return tracked and clean and staged


@dataclass
class Cal:
    frame: Frame
    ref: Discriminator
    x0: np.ndarray  # class-0 calibration points in this frame
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    z0: np.ndarray  # raw complex calibration shots (for refits in another epoch's frame)
    z1: np.ndarray


@dataclass
class Epoch:
    epoch: int
    job_id: str
    job_dir: pathlib.Path
    meta: dict
    cal: dict = field(default_factory=dict)  # (qubit, context) -> Cal

    def _arr(self, circuit: str, reg: str) -> np.ndarray:
        hits = sorted((self.job_dir / "decoded").glob(f"*_{circuit}_{reg}.npy"))
        hits = [h for h in hits if h.name.split("_", 1)[1] == f"{circuit}_{reg}.npy"]
        if len(hits) != 1:
            raise FileNotFoundError(f"{circuit}.{reg} in {self.job_dir}")
        return np.load(hits[0])

    def target(self, name: str) -> dict[str, np.ndarray]:
        """Target registers {register: (N, bits) complex128}; only targets of this epoch exist."""
        regs = self.meta["measurement_map"][name]
        return {r: self._arr(name, r) for r in regs}


def _cal(z0: np.ndarray, z1: np.ndarray) -> Cal:
    fr = fit_frame(z0, z1)
    x0, y0 = fr.apply(z0)
    x1, y1 = fr.apply(z1)
    return Cal(fr, fit_lda(x0, y0, x1, y1), x0, y0, x1, y1, z0, z1)


def load_epoch(epoch: int, allow_test: bool = False) -> Epoch:
    _check_access(epoch, allow_test)
    j = ledger_jobs()[epoch]
    jd = ROOT / "data" / "raw" / j["job_id"]
    chk = json.loads((jd / "decoded" / "checks.json").read_text(encoding="utf-8"))
    if not chk.get("pass_shape") or chk.get("cluster_failures"):
        raise RuntimeError(f"epoch {epoch} job {j['job_id']} did not pass acceptance")
    meta = json.loads((jd / "job_metadata.json").read_text(encoding="utf-8"))
    ep = Epoch(epoch, j["job_id"], jd, meta)
    c0, c1 = ep._arr("CAL0", "meas"), ep._arr("CAL1", "meas")
    for k, q in enumerate(meta["measurement_map"]["CAL0"]["meas"]):
        ep.cal[(q, "final")] = _cal(c0[:, k], c1[:, k])
    r1, r2 = ep._arr("CALM", "r1"), ep._arr("CALM", "r2")
    for k, q in enumerate(meta["measurement_map"]["CALM"]["r1"]):
        z0, z1 = (r2[:, k], r1[:, k]) if k % 2 == 0 else (r1[:, k], r2[:, k])
        ep.cal[(q, "mid")] = _cal(z0, z1)
    return ep


@dataclass
class Position:
    index: int
    register: str
    bit: int
    qubit: int
    context: str  # "final" | "mid"
    raw: np.ndarray  # (N,) complex128
    cal: Cal
    fams: dict  # "F-boot" / "F-plan" -> Family


@dataclass
class Dataset:
    name: str
    index: int
    epoch: int
    kind: str
    positions: list

    @property
    def raw(self) -> np.ndarray:
        return np.column_stack([p.raw for p in self.positions])


def load_dataset(entry: dict, ep: Epoch, families: bool = True) -> Dataset:
    name, idx = entry["id"], int(entry["index"])
    if int(entry["campaign_day"]) != ep.epoch:
        raise ValueError(f"{name} belongs to epoch {entry['campaign_day']}, not {ep.epoch}")
    regs = ep.target(name)
    mmap = ep.meta["measurement_map"][name]
    pos = []
    for i, (reg, bit) in enumerate(entry["positions"]):
        ctx = "mid" if reg.startswith(MID_REGS_PREFIX) else "final"
        q = mmap[reg][bit]
        cal = ep.cal[(q, ctx)]
        fams = {}
        if families:
            dth, dbe, ref = bootstrap_widths(cal.x0, cal.y0, cal.x1, cal.y1, n_boot=N_BOOT, seed=rng("fboot", idx, i))
            assert ref == cal.ref
            fams = {"F-boot": family_from_widths(ref, 2 * dth, 2 * dbe), "F-plan": family_from_widths(ref, *F_PLAN)}
        pos.append(Position(i, reg, bit, q, ctx, np.ascontiguousarray(regs[reg][:, bit]), cal, fams))
    m_decl = int(entry["m"])
    if len(pos) != m_decl or sum(a.shape[1] for a in regs.values()) != m_decl:
        raise AssertionError(f"{name}: m mismatch (prereg/M2.md section 1)")
    return Dataset(name, idx, ep.epoch, entry["kind"], pos)


def dev_datasets(families: bool = True):
    """Yield (Epoch, Dataset) for every DEV dataset, in workload order."""
    w = workload()
    eps = {e: load_epoch(e) for e in DEV_EPOCHS}
    for entry in w["datasets"]:
        if entry["split"] != "DEV":
            continue
        e = int(entry["campaign_day"])
        yield eps[e], load_dataset(entry, eps[e], families)
