"""End-to-end query engines: the timed IQSTORE engine `IQFile` (iqstore/lean.py) equals the
reference path `IQFileRef`; the allocation-free tree decoder equals `fastdec._decode_tree`; the B4q engine equals the
label-set rule of `eval.m2_cov.b4q_label_sets`. Synthetic positions (no campaign data)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from baselines.b4q_query import B4qFile  # noqa: E402
from baselines.storage_baselines import b4_slab, b4q_position, fixed_margins  # noqa: E402
from eval.m2_cov import b4q_label_sets  # noqa: E402
from iqstore.e2e import IQFile, IQFileRef, _HDR, _zload, read_groups  # noqa: E402
from iqstore.family import Discriminator, family_from_widths, fit_lda  # noqa: E402
from iqstore.fastdec import _decode_tree  # noqa: E402
from iqstore.frame import fit_frame  # noqa: E402
from iqstore.ingest import ingest_position  # noqa: E402
from iqstore.lean import _decode_tree3  # noqa: E402
from iqstore.query import exact_dot, exact_value, interval_from_tuples, intervals_multi  # noqa: E402
from iqstore.storage import write_container  # noqa: E402

TAU = 1e-3
DELTAS = np.array([c * 2.0 ** -p for p in range(16) for c in (1.0, 1.25, 1.5, 1.75)])


def _grouped(blobs):
    return [len(blobs).to_bytes(2, "little")] + blobs


def _positions(m=4, N=4000, seed=3):
    g = np.random.default_rng(seed)
    out = []
    for j in range(m):
        c0 = complex(g.normal(), g.normal())
        c1 = c0 + complex(2.0 + g.normal() * 0.2, 1.0 + g.normal() * 0.2)
        cal0 = c0 + (g.normal(size=2000) + 1j * g.normal(size=2000)) * 0.45
        cal1 = c1 + (g.normal(size=2000) + 1j * g.normal(size=2000)) * 0.45
        lab = g.random(N) < 0.4
        raw = np.where(lab, c1, c0) + (g.normal(size=N) + 1j * g.normal(size=N)) * 0.45
        fr = fit_frame(cal0, cal1)
        x0, y0 = fr.apply(cal0)
        x1, y1 = fr.apply(cal1)
        ref = fit_lda(x0, y0, x1, y1)
        fam = family_from_widths(ref, 0.03 + 0.02 * j, 0.05 + 0.03 * j)
        out.append((fr, fam, ref, raw.astype(np.complex128)))
    return out


def _ref_counts(lss):
    k = len(lss)
    P = 1 << k
    lo = np.zeros(P, np.int64)
    hi = np.zeros(P, np.int64)
    plo = phi = 0
    N = lss[0].size
    can0 = np.zeros(N, np.int64)
    can1 = np.zeros(N, np.int64)
    for i, ls in enumerate(lss):
        can0 |= ((ls & 1) > 0).astype(np.int64) << i
        can1 |= ((ls & 2) > 0).astype(np.int64) << i
    for s in range(N):
        b = int(can0[s] & can1[s])
        base = int(can1[s]) & ~b
        if b == 0:
            lo[base] += 1
            hi[base] += 1
            par = -1 if bin(base).count("1") % 2 else 1
            plo += par
            phi += par
        else:
            sub = b
            while True:
                hi[base | sub] += 1
                if sub == 0:
                    break
                sub = (sub - 1) & b
            plo -= 1
            phi += 1
    return lo, hi, plo, phi


def _draws(g, fams, S, wide):
    hs = []
    for j in S:
        f = fams[j]
        u, v = g.uniform(-0.5, 1.5, 2) if wide else g.uniform(0, 1, 2)
        hs.append(Discriminator(f.th_lo + u * (f.th_hi - f.th_lo), f.b_lo + v * (f.b_hi - f.b_lo)))
    return hs


def _same(a, b):
    return (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and int(a[2]) == int(b[2])
            and int(a[3]) == int(b[3]))


def test_lean_engine_equals_reference(tmp_path):
    pos = _positions()
    N = pos[0][3].size
    blobs = []
    for fr, fam, ref, raw in pos:
        x, y = fr.apply(raw)
        rep, ib, info = ingest_position(fr, fam, x, y, raw, TAU, 3, 16, 0.5)
        blobs += _grouped(ib)
    path = tmp_path / "iq.bin"
    write_container(path, blobs, "IQSTORE", N, len(pos))
    new, old = IQFile(path), IQFileRef(path)
    fams = [p[1] for p in pos]
    g = np.random.default_rng(11)
    for q in range(120):
        k = 1 + q % 4
        S = sorted(g.choice(len(pos), k, replace=False).tolist())
        hs = _draws(g, fams, S, wide=q % 3 == 0)
        a, b = new.query(S, hs), old.query(S, hs)
        assert _same(a, b) and np.array_equal(a[4], b[4]), (S, hs)
        # label-set tuples: every shot once; definite tuples equal the lower pattern counts
        assert int(a[4].sum()) == N
        for p_ in range(1 << k):
            key = sum((2 if (p_ >> i) & 1 else 1) << (2 * i) for i in range(k))
            assert int(a[4][key]) == int(a[0][p_])
    # tree decoder: allocation-free version equals the reference decoder on every stored tree
    _, groups = read_groups(path, list(range(len(pos))))
    for gr in groups:
        hd = _HDR.unpack(gr[0])
        tb = np.frombuffer(_zload(gr[1]) + b"\0", dtype=np.uint8)
        args = (hd[12], hd[0], hd[1], hd[2], float(np.cos(hd[7])), float(np.sin(hd[7])), float(np.cos(hd[8])),
                float(np.sin(hd[8])), hd[7], hd[8], hd[9], hd[10], hd[14] >= 4)
        a = _decode_tree(tb, *args)
        b = _decode_tree3(tb, 0, *args)
        assert all(np.array_equal(u, v) for u, v in zip(a[:4], b[:4])) and a[4] == b[4]


def test_b4q_engine_equals_label_set_rule(tmp_path):
    pos = _positions(m=3, N=3000, seed=5)
    N = pos[0][3].size
    blobs = []
    meta = []
    for fr, fam, ref, raw in pos:
        x, y = fr.apply(raw)
        box = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
        M0, M1 = fixed_margins(ref, fam, box)
        in_slab, side1 = b4_slab(x, y, ref, M0, M1)
        qb, qinfo = b4q_position(x, y, in_slab, side1, raw, fr, ref, M0, M1, fam, TAU, DELTAS)
        blobs += _grouped(qb)
        meta.append((x, y, in_slab, side1, qinfo, fam))
    path = tmp_path / "b4q.bin"
    write_container(path, blobs, "B4q", N, len(pos))
    eng = B4qFile(path)
    fams = [p[1] for p in pos]
    g = np.random.default_rng(13)
    for q in range(60):
        k = 1 + q % 3
        S = sorted(g.choice(len(pos), k, replace=False).tolist())
        hs = _draws(g, fams, S, wide=q % 3 == 0)
        lss = [b4q_label_sets(meta[j][0], meta[j][1], meta[j][2], meta[j][3], meta[j][4], meta[j][5], h,
                              meta[j][4]["mode"] != "grid") for j, h in zip(S, hs)]
        out = eng.query(S, hs)
        assert _same(out, _ref_counts(lss)), (S, hs)
        gv = g.normal(size=1 << k) * 10.0 ** g.integers(-3, 4)
        assert interval_from_tuples(out[4], k, gv, exact=True) == intervals_multi(np.column_stack(lss), [gv],
                                                                                  exact=True)[0]


def test_exact_dot_equals_fraction_accumulation():
    g = np.random.default_rng(17)
    for _ in range(200):
        k = int(g.integers(1, 5))
        labels = (g.random((500, k)) < 0.4).astype(np.int8)
        gv = g.normal(size=1 << k) * 10.0 ** g.integers(-8, 9, size=1 << k)
        idx = (labels.astype(np.int64) << np.arange(k)).sum(1)
        cnt = np.bincount(idx, minlength=1 << k)
        assert exact_dot(cnt, gv, exact=True) == exact_value(labels, gv, exact=True)
        assert exact_dot(cnt, gv) == exact_value(labels, gv)


def test_vector_trig_equals_scalar():
    """lean.IQFile computes cos / sin of all angles in one vectorized call; the encoder and the reference use scalar
    calls. Bit identity over many sizes and magnitudes (the lean-vs-reference test checks it end to end)."""
    g = np.random.default_rng(5)
    for sc in (1e-3, 1.0, 3.2, 100.0):
        x = g.uniform(-sc, sc, 3000)
        for n in (1, 2, 3, 6, 9, 12, 18):
            for s in range(0, 600, n):
                v = x[s:s + n].reshape(-1, 1) if n % 3 else x[s:s + n].reshape(-1, 3)
                for fn in (np.cos, np.sin):
                    a = fn(v).ravel()
                    b = np.array([fn(t) for t in v.ravel()])
                    assert (a.view(np.int64) == b.view(np.int64)).all()
