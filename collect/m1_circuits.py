"""M1 campaign circuits (qpu/M1_estimate.md v2, prereg/M1.md). Pure construction; no network, no submission.

Logical circuits are built on 16 virtual qubits (index i = i-th qubit of the fixed M1 set), transpiled against the
coupling map restricted to the set (any SWAP stays inside the set), then embedded into the full device register by
virtual -> physical = set[i]. `check_embedded` verifies that every 2-qubit gate acts on a real coupler of the set and that
every measured physical qubit lies in the set.
"""
from __future__ import annotations

import collections
import math

import numpy as np

SEED = 20261107
N_SET = 16
N_TARGETS = 35
N_DAYS = 14
CAL_SHOTS = 20_000
CALM_SHOTS = 4_096
TARGET_SHOTS = 20_000
CZ_ERR_MAX = 0.05
REP_PATH = 9


# ------------------------------------------------------------------------------------------------ device graph

def device_graph(props: dict):
    """From a BackendProperties dict: readout error, operational flag, CZ couplers with their gate error."""
    ro, oper = {}, {}
    for i, q in enumerate(props["qubits"]):
        for p in q:
            if p["name"] == "readout_error":
                ro[i] = float(p["value"])
            elif p["name"] == "operational":
                oper[i] = bool(p["value"])
    cz = {}
    for g in props["gates"]:
        if g["gate"] == "cz" and len(g["qubits"]) == 2:
            err = next((float(p["value"]) for p in g["parameters"] if p["name"] == "gate_error"), 1.0)
            cz[tuple(sorted(g["qubits"]))] = err
    return ro, oper, cz


def choose_connected_set(props: dict, k: int = N_SET, target=None) -> list[int]:
    """Greedy growth from every operational seed (lowest readout error neighbour over a CZ coupler with error < 5 %);
    keep the set with the lowest mean readout error; ties -> smallest sorted list. With a Target, only qubits that
    support `measure` and couplers that support `cz` in both directions are eligible."""
    ro, oper, cz = device_graph(props)
    ok = {q for q in ro if oper.get(q, True)
          and (target is None or target.instruction_supported("measure", (q,)))}
    adj = collections.defaultdict(set)
    for (a, b), e in cz.items():
        if target is not None and not (target.instruction_supported("cz", (a, b))
                                       and target.instruction_supported("cz", (b, a))):
            continue
        if e < CZ_ERR_MAX and a in ok and b in ok:
            adj[a].add(b)
            adj[b].add(a)
    best = None
    for seed in sorted(ok):
        s = {seed}
        while len(s) < k:
            cand = {v for u in s for v in adj[u] if v not in s}
            if not cand:
                break
            s.add(min(cand, key=lambda v: (ro[v], v)))
        if len(s) < k:
            continue
        key = (sum(ro[q] for q in s) / k, sorted(s))
        if best is None or key < best:
            best = key
    if best is None:
        raise RuntimeError("no connected set of the requested size")
    return best[1]


def set_edges(props: dict, qset: list[int]) -> list[tuple[int, int]]:
    """Couplers inside the set (CZ error < 5 %), as virtual index pairs (i < j)."""
    _, _, cz = device_graph(props)
    pos = {q: i for i, q in enumerate(qset)}
    return sorted((min(pos[a], pos[b]), max(pos[a], pos[b])) for (a, b), e in cz.items()
                  if e < CZ_ERR_MAX and a in pos and b in pos)


def longest_path(edges: list[tuple[int, int]], n: int) -> list[int]:
    """Longest simple path (virtual indices); ties -> lexicographically smallest list (exhaustive DFS, n = 16)."""
    adj = collections.defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    best: list[int] = []

    def dfs(p, seen):
        nonlocal best
        if len(p) > len(best) or (len(p) == len(best) and p < best):
            best = list(p)
        for v in sorted(adj[p[-1]]):
            if v not in seen:
                seen.add(v)
                p.append(v)
                dfs(p, seen)
                p.pop()
                seen.remove(v)

    for s in range(n):
        dfs([s], {s})
    return best


# ------------------------------------------------------------------------------------------------ logical circuits

def _qc(n, regs, name):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    return QuantumCircuit(QuantumRegister(n, "v"), *[ClassicalRegister(s, r) for r, s in regs], name=name)


def cal_circuits(n: int = N_SET):
    c0 = _qc(n, [("meas", n)], "CAL0")
    c0.measure(range(n), range(n))
    c1 = _qc(n, [("meas", n)], "CAL1")
    c1.x(range(n))
    c1.measure(range(n), range(n))
    cm = _qc(n, [("r1", n), ("r2", n), ("r3", n)], "CALM")
    cm.x([i for i in range(n) if i % 2 == 0])
    cm.measure(range(n), cm.cregs[0])
    cm.x(range(n))
    cm.measure(range(n), cm.cregs[1])
    cm.x(range(n))
    cm.measure(range(n), cm.cregs[2])
    return [c0, c1, cm]


def _bfs_tree(edges, n, root=0):
    adj = collections.defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    order, parent, seen = [root], {}, {root}
    for u in order:
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                order.append(v)
    return order, parent


def ghz(edges, n_active: int, tag: str, n: int = N_SET):
    order, parent = _bfs_tree(edges, n)
    act = order[:n_active]
    qc = _qc(n, [("meas", n)], f"GHZ{n_active}_{tag}")
    qc.h(act[0])
    for v in act[1:]:
        qc.cx(parent[v], v)
    qc.measure(range(n), range(n))
    return qc


def random_clifford_layers(edges, depth: int, seed: int, n: int = N_SET):
    from qiskit.quantum_info import random_clifford

    rng = np.random.default_rng([SEED, depth, seed])
    qc = _qc(n, [("meas", n)], f"RCL_d{depth}_s{seed}")
    for _ in range(depth):
        for q in range(n):
            qc.append(random_clifford(1, seed=int(rng.integers(2**31))).to_instruction(), [q])
        used, perm = set(), rng.permutation(len(edges))
        for k in perm:  # random maximal matching of the set's couplers
            a, b = edges[k]
            if a not in used and b not in used:
                used |= {a, b}
                qc.cz(a, b)
    qc.measure(range(n), range(n))
    return qc


def product_state(seed: int, n: int = N_SET):
    rng = np.random.default_rng([SEED, 999, seed])
    qc = _qc(n, [("meas", n)], f"PROD_s{seed}")
    qc.x([int(i) for i in np.flatnonzero(rng.integers(0, 2, n))])
    qc.measure(range(n), range(n))
    return qc


def repetition_code(path: list[int], rounds: int, logical: int, n: int = N_SET):
    data, anc = path[0::2], path[1::2]  # 5 data, 4 ancillas on a 9-qubit path
    regs = [(f"a{t}", len(anc)) for t in range(1, rounds + 1)] + [("data", len(data))]
    qc = _qc(n, regs, f"REP_d{len(data)}_r{rounds}_L{logical}")
    if logical:
        qc.x(data)
    for t in range(rounds):
        for i, a in enumerate(anc):
            qc.cx(data[i], a)
            qc.cx(data[i + 1], a)
        qc.measure(anc, qc.cregs[t])
        qc.reset(anc)
    qc.measure(data, qc.cregs[rounds])
    return qc


def target_list(edges, path) -> list:
    t = []
    for n_act in (8, 12, 16):
        for rep in ("a", "b"):
            t.append(ghz(edges, n_act, rep))
    for depth in range(2, 21, 2):
        for s in (0, 1):
            t.append(random_clifford_layers(edges, depth, s))
    for s in range(4):
        t.append(product_state(s))
    for rounds, logical in ((5, 0), (5, 1), (10, 0), (10, 1), (8, 0)):
        t.append(repetition_code(path[:REP_PATH], rounds, logical))
    assert len(t) == N_TARGETS and [c.name for c in t] == TARGET_NAMES
    return t


TARGET_NAMES = ([f"GHZ{n}_{r}" for n in (8, 12, 16) for r in ("a", "b")]
                + [f"RCL_d{d}_s{s}" for d in range(2, 21, 2) for s in (0, 1)]
                + [f"PROD_s{s}" for s in range(4)]
                + [f"REP_d5_r{r}_L{l}" for r, l in ((5, 0), (5, 1), (10, 0), (10, 1), (8, 0))])
GHZ_PAIRS = ((0, 1), (2, 3), (4, 5))  # target indices of GHZ8 a/b, GHZ12 a/b, GHZ16 a/b


def _days(order, n_targets, n_days):
    return [order[math.floor((d - 1) * n_targets / n_days): math.floor(d * n_targets / n_days)]
            for d in range(1, n_days + 1)]


def schedule(n_targets: int = N_TARGETS, n_days: int = N_DAYS) -> list[list[int]]:
    """Fixed permutation: the first seed s = 20261107, 20261108, ... whose permutation puts the two runs of every GHZ
    size on DIFFERENT campaign days; campaign day d (1-based) takes order[floor((d-1)N/D) : floor(dN/D)]."""
    for s in range(SEED, SEED + 10_000):
        order = np.random.default_rng(s).permutation(n_targets).tolist()
        days = _days(order, n_targets, n_days)
        day_of = {t: d for d, ts in enumerate(days) for t in ts}
        if all(day_of[a] != day_of[b] for a, b in GHZ_PAIRS):
            return days
    raise RuntimeError("no admissible schedule")


def schedule_seed(n_targets: int = N_TARGETS, n_days: int = N_DAYS) -> int:
    days = schedule(n_targets, n_days)
    for s in range(SEED, SEED + 10_000):
        if _days(np.random.default_rng(s).permutation(n_targets).tolist(), n_targets, n_days) == days:
            return s
    raise RuntimeError("unreachable")


# ------------------------------------------------------------------------------------------------ transpile + embed

def transpile_in_set(qc, edges, basis, seed: int = SEED):
    """Transpile a logical circuit against the coupling map restricted to the set (virtual indices)."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    cm = CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
    return transpile(qc, coupling_map=cm, basis_gates=basis, initial_layout=list(range(qc.num_qubits)),
                     optimization_level=1, seed_transpiler=seed)


def embed(tqc, qset: list[int], n_total: int):
    """Map virtual qubit i -> physical qset[i] in a device-wide circuit; classical registers kept by name/order."""
    from qiskit import QuantumCircuit, QuantumRegister

    out = QuantumCircuit(QuantumRegister(n_total, "q"), *tqc.cregs, name=tqc.name)
    phys = [out.qubits[qset[tqc.find_bit(q).index]] for q in tqc.qubits]
    out.compose(tqc, qubits=phys, clbits=out.clbits, inplace=True)
    return out


def check_embedded(circ, qset: list[int], props: dict, target=None) -> list[str]:
    """Problems: 2-qubit gates off the set's healthy couplers, operations or measurements outside the set, and (if the
    backend Target is given) any instruction the Target does not support on those qubits (faulty qubits / couplers are
    absent from the runtime Target)."""
    _, _, cz = device_graph(props)
    allowed = {e for e, err in cz.items() if err < CZ_ERR_MAX and e[0] in qset and e[1] in qset}
    s = set(qset)
    probs = []
    for inst in circ.data:
        qs = [circ.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "barrier":
            continue
        if any(q not in s for q in qs):
            probs.append(f"{circ.name}: {inst.operation.name} on {qs} outside the set")
        if len(qs) == 2 and tuple(sorted(qs)) not in allowed:
            probs.append(f"{circ.name}: 2-qubit {inst.operation.name} on {qs} is not a coupler of the set")
        if target is not None and not target.instruction_supported(inst.operation.name, tuple(qs)):
            probs.append(f"{circ.name}: {inst.operation.name} on {qs} not supported by today's Target (faulty?)")
    return probs


def build_day(props: dict, qset: list[int], day: int, n_total: int, basis) -> list:
    """Device-wide circuits for campaign day `day` (1-based): CAL0, CAL1, CALM, then that day's targets."""
    edges = set_edges(props, qset)
    path = longest_path(edges, len(qset))
    if len(path) < REP_PATH:
        raise RuntimeError(f"longest path {len(path)} < {REP_PATH}: repetition code does not fit the set")
    targets = target_list(edges, path)
    idx = schedule()[day - 1]
    logical = cal_circuits() + [targets[i] for i in idx]
    out = []
    for qc in logical:
        e = embed(transpile_in_set(qc, edges, basis), qset, n_total)
        probs = check_embedded(e, qset, props)
        if probs:
            raise RuntimeError("; ".join(probs[:5]))
        out.append(e)
    return out


def shots_for(name: str) -> int:
    return CAL_SHOTS if name in ("CAL0", "CAL1") else CALM_SHOTS if name == "CALM" else TARGET_SHOTS
