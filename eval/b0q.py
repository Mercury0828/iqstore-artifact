"""B0q: re-classification of the raw complex128 IQ (B0) with a vectorized columnar engine (prereg/M2.md v4 §3, [F9]).

Engines answer a pattern-count query (S, h_S) -> counts of every label pattern over S (every class 1-5 statistic
follows: single counts, pair counts and <ZZ>, marginals, mitigated <f, M p>). All are EXACT: a float filter decides
shots far from the boundary, the rest are re-decided by the reference path (`Frame.apply` + `exact_sign`).
  numpy+numba   frame map by `Frame.apply` (NumPy, the reference arithmetic), labels and histogram in one numba pass.
  duckdb        DuckDB 2.0.0, threads = 1; frame map and decision value in SQL; uncertain row ids re-decided exactly.
  folded        (strongest fair B0q for Q2, not part of the engine selection) frame map and discriminator folded into
                one affine predicate per position.
End-to-end mode (prereg §6): `B0File` / `DuckFile` start from the persisted file on disk (B0 container / DuckDB database
file) and decode what the query needs on every run; nothing decoded is retained between runs. The primary engine is
selected in `eval/m2_dev.py b0q`.
"""
from __future__ import annotations

import numpy as np
import numba

from iqstore.family import Discriminator, exact_sign, unit
from iqstore.frame import Frame

_EPS = np.finfo(np.float64).eps


@numba.njit(cache=True)
def _pattern_counts(xs, ys, ps, qs, bs, k, out_unsure):
    n = xs.shape[0]
    cnt = np.zeros(1 << k, dtype=np.int64)
    n_uns = 0
    tiny = 2.2250738585072014e-308
    for s in range(n):
        idx = 0
        uns = False
        for i in range(k):
            px = ps[i] * xs[s, i]
            qy = qs[i] * ys[s, i]
            d = (px + qy) - bs[i]
            err = 2.0 * 2.220446049250313e-16 * (abs(px) + abs(qy) + abs(bs[i])) + tiny
            if not (abs(d) > err):
                uns = True
            if d > 0:
                idx |= 1 << i
        if uns:
            out_unsure[n_uns] = s
            n_uns += 1
        else:
            cnt[idx] += 1
    return cnt, n_uns


def numba_counts(raw_cols: list[np.ndarray], frames: list[Frame], hs: list[Discriminator]) -> np.ndarray:
    k = len(raw_cols)
    xy = [f.apply(z) for z, f in zip(raw_cols, frames)]
    xs = np.ascontiguousarray(np.column_stack([a for a, _ in xy]))
    ys = np.ascontiguousarray(np.column_stack([b for _, b in xy]))
    pq = [unit(h.theta) for h in hs]
    ps = np.array([a for a, _ in pq])
    qs = np.array([b for _, b in pq])
    bs = np.array([h.beta for h in hs])
    uns = np.empty(xs.shape[0], dtype=np.int64)
    cnt, n_uns = _pattern_counts(xs, ys, ps, qs, bs, k, uns)
    if n_uns:
        u = uns[:n_uns]
        idx = np.zeros(n_uns, dtype=np.int64)
        for i in range(k):
            idx |= (exact_sign(xs[u, i], ys[u, i], ps[i], qs[i], bs[i]) > 0).astype(np.int64) << i
        cnt += np.bincount(idx, minlength=1 << k)
    return cnt


@numba.njit(cache=True)
def _folded_counts(re, im, Sx, A, B, C, R, k, out_unsure):
    """Pattern counts with the frame map folded into one affine predicate per position: d = A re + B im + C. Shots with
    |d| <= R(re, im) (a generous bound on |d - reference decision value|) are returned for the exact path."""
    n = re.shape[1]
    cnt = np.zeros(1 << k, dtype=np.int64)
    n_uns = 0
    for s in range(n):
        idx = 0
        uns = False
        for i in range(k):
            r = re[Sx[i], s]
            m = im[Sx[i], s]
            d = A[i] * r + B[i] * m + C[i]
            err = (R[i, 0] * (abs(A[i] * r) + abs(B[i] * m) + R[i, 1]) + R[i, 2] * (abs(r - R[i, 3]) + abs(m - R[i, 4]))
                   + 1e-300)  # absolute allowance: relative bounds fail in the subnormal range
            if not (abs(d) > err) or (abs(r - R[i, 3]) + abs(m - R[i, 4])) < 1e-280:
                uns = True  # near the boundary, or near the frame centre where the reference map may underflow
            if d > 0:
                idx |= 1 << i
        if uns:
            out_unsure[n_uns] = s
            n_uns += 1
        else:
            cnt[idx] += 1
    return cnt, n_uns


class FoldedEngine:
    """Strongest fair B0q: raw columns as float64 arrays; per query the frame map and the
    discriminator are folded into d = A re + B im + C (A = (p cos phi - q sin phi)/s, B = (p sin phi + q cos phi)/s,
    C = -(A cx + B cy) - beta). A float filter with a 64x margin over the rounding of both the folded value and the
    reference frame map sends near-boundary shots to the exact reference path (`Frame.apply` + `exact_sign`), so answers
    equal the exact reference (checked on DEV)."""

    def __init__(self, raw: np.ndarray):
        self.re = np.ascontiguousarray(raw.real.T)  # (m, N): one contiguous row per position
        self.im = np.ascontiguousarray(raw.imag.T)
        self.raw = raw

    def counts(self, S: list[int], frames: list[Frame], hs: list[Discriminator]) -> np.ndarray:
        k = len(S)
        A = np.empty(k)
        B = np.empty(k)
        C = np.empty(k)
        R = np.empty((k, 5))
        for i, (f, h) in enumerate(zip(frames, hs)):
            p, q = unit(h.theta)
            cs, sn = np.cos(f.phi), np.sin(f.phi)
            A[i] = (p * cs - q * sn) / f.scale
            B[i] = (p * sn + q * cs) / f.scale
            C[i] = -(A[i] * f.cx + B[i] * f.cy) - h.beta
            R[i] = (64 * _EPS, abs(C[i]) + abs(A[i] * f.cx) + abs(B[i] * f.cy) + abs(h.beta),
                    64 * _EPS * (abs(p) + abs(q)) / f.scale, f.cx, f.cy)
        uns = np.empty(self.re.shape[1], dtype=np.int64)
        cnt, n_uns = _folded_counts(self.re, self.im, np.asarray(S, dtype=np.int64), A, B, C, R, k, uns)
        if n_uns:
            u = uns[:n_uns]
            idx = np.zeros(n_uns, dtype=np.int64)
            for i, (j, f, h) in enumerate(zip(S, frames, hs)):
                x, y = f.apply(self.raw[u, j])
                p, q = unit(h.theta)
                idx |= (exact_sign(x, y, p, q, h.beta) > 0).astype(np.int64) << i
            cnt += np.bincount(idx, minlength=1 << k)
        return cnt


class DuckEngine:
    """DuckDB 2.0.0 engine (threads = 1). The frame map and the decision value d = p x + q y - beta are computed in SQL;
    a row is decided in SQL only if |d| exceeds a generous float error bound at every position of S, otherwise its row id
    is returned and the row is re-decided exactly by the reference path (`Frame.apply` + `exact_sign`), so answers equal
    the exact reference."""

    def __init__(self, raw: np.ndarray):
        import duckdb
        import pyarrow as pa

        self.con = duckdb.connect()
        self.con.execute("SET threads = 1")
        self.raw = raw
        self.m = raw.shape[1]
        tbl = pa.table({"rid": np.arange(raw.shape[0], dtype=np.int64)}
                       | {f"re{j}": np.ascontiguousarray(raw[:, j].real) for j in range(self.m)}
                       | {f"im{j}": np.ascontiguousarray(raw[:, j].imag) for j in range(self.m)})
        self.con.register("arrow_src", tbl)
        self.con.execute("CREATE TABLE iq AS SELECT * FROM arrow_src")
        self.con.unregister("arrow_src")

    def counts(self, S: list[int], frames: list[Frame], hs: list[Discriminator]) -> np.ndarray:
        cols, conds, terms = [], [], []
        for i, (j, f, h) in enumerate(zip(S, frames, hs)):
            p, q = unit(h.theta)
            cs, sn, sc = float(np.cos(f.phi)), float(np.sin(f.phi)), float(f.scale)
            dx, dy = f"(re{j} - {f.cx!r})", f"(im{j} - {f.cy!r})"
            x = f"(({dx} * {cs!r} + {dy} * {sn!r}) / {sc!r})"
            y = f"((-{dx} * {sn!r} + {dy} * {cs!r}) / {sc!r})"
            cols.append(f"{p!r} * {x} + {q!r} * {y} - {h.beta!r} AS d{i}")
            cols.append(f"1e-12 * (abs({p!r} * {x}) + abs({q!r} * {y}) + {abs(h.beta)!r} "
                        f"+ (abs({dx}) + abs({dy})) / {sc!r}) + 1e-300 AS e{i}")
            conds.append(f"abs(d{i}) > e{i}")
            terms.append(f"(CASE WHEN d{i} > 0 THEN {1 << i} ELSE 0 END)")
        base = f"WITH t AS (SELECT rid, {', '.join(cols)} FROM iq)"
        sure = " AND ".join(conds)
        out = np.zeros(1 << len(S), dtype=np.int64)
        for pat, c in self.con.execute(f"{base} SELECT {' + '.join(terms)} AS pat, count(*) FROM t WHERE {sure} "
                                       "GROUP BY pat").fetchall():
            out[int(pat)] = int(c)
        uns = np.array([r[0] for r in self.con.execute(f"{base} SELECT rid FROM t WHERE NOT ({sure})").fetchall()],
                       dtype=np.int64)
        if uns.size:
            idx = np.zeros(uns.size, dtype=np.int64)
            for i, (j, f, h) in enumerate(zip(S, frames, hs)):
                x, y = f.apply(self.raw[uns, j])
                p, q = unit(h.theta)
                idx |= (exact_sign(x, y, p, q, h.beta) > 0).astype(np.int64) << i
            out += np.bincount(idx, minlength=1 << len(S))
        return out


# ---------------------------------------------------------------------------------------------------- end-to-end

class B0File:
    """numpy+numba B0q from the persisted B0 container (iqstore.storage container of `b0_blobs`)."""

    def __init__(self, path):
        self.path = path

    def counts(self, S: list[int], hs: list[Discriminator]) -> np.ndarray:
        import struct
        from iqstore.storage import DTYPES, read_container, zload
        _, N, m, blobs = read_container(self.path)
        dcode, n, m2, mode = struct.unpack("<BqqB", blobs[0])
        side = np.frombuffer(blobs[1], dtype=np.float64).reshape(m2, 8)
        dt = DTYPES[dcode]
        if mode == 1:
            cols = [np.frombuffer(zload(blobs[2 + j]), dtype=dt) for j in S]
        else:
            allr = np.frombuffer(zload(blobs[2]), dtype=dt).reshape(n, m2)
            cols = [np.ascontiguousarray(allr[:, j]) for j in S]
        frames = [Frame.from_array(side[j, :4]) for j in S]
        return numba_counts(cols, frames, hs)


class B0FileSel:
    """numpy+numba B0q that reads only the bytes it needs (amendment A17): the container header, the frame blob, and,
    when B0 stores per-position columns, only the touched columns (seek past the others); when B0 stores the whole
    matrix as one blob, that blob. Same arithmetic as B0File."""

    def __init__(self, path):
        self.path = path

    def counts(self, S: list[int], hs: list[Discriminator]) -> np.ndarray:
        import os
        import struct
        from iqstore.storage import DTYPES, zload
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            def blob_at(pos):
                os.lseek(fd, pos, 0)
                (L,) = struct.unpack("<I", os.read(fd, 4))
                return L, pos + 4

            ln = os.read(fd, 1)[0]
            os.lseek(fd, 1 + ln, 0)
            N, m, nb = struct.unpack("<qqI", os.read(fd, 20))
            pos = 1 + ln + 20
            L, p0 = blob_at(pos)
            hdr = os.read(fd, L)
            pos = p0 + L
            L, p0 = blob_at(pos)
            side = np.frombuffer(os.read(fd, L), dtype=np.float64).reshape(-1, 8)
            pos = p0 + L
            dcode, n, m2, mode = struct.unpack("<BqqB", hdr)
            dt = DTYPES[dcode]
            if mode == 1:
                need = set(S)
                got = {}
                for j in range(max(S) + 1):
                    L, p0 = blob_at(pos)
                    if j in need:
                        got[j] = os.read(fd, L)
                    pos = p0 + L
                cols = [np.frombuffer(zload(got[j]), dtype=dt) for j in S]
            else:
                L, p0 = blob_at(pos)
                allr = np.frombuffer(zload(os.read(fd, L)), dtype=dt).reshape(n, m2)
                cols = [np.ascontiguousarray(allr[:, j]) for j in S]
        finally:
            os.close(fd)
        frames = [Frame.from_array(side[j, :4]) for j in S]
        return numba_counts(cols, frames, hs)


class DuckFile:
    """DuckDB B0q from a persisted DuckDB database file (tables iq(rid, re_j, im_j) and frames(j, cx, cy, phi, scale))."""

    def __init__(self, path):
        self.path = str(path)

    @staticmethod
    def write(path, raw: np.ndarray, frames: list[Frame]) -> None:
        import duckdb
        import pyarrow as pa
        con = duckdb.connect(str(path))
        con.execute("SET threads = 1")
        tbl = pa.table({"rid": np.arange(raw.shape[0], dtype=np.int64)}
                       | {f"re{j}": np.ascontiguousarray(raw[:, j].real) for j in range(raw.shape[1])}
                       | {f"im{j}": np.ascontiguousarray(raw[:, j].imag) for j in range(raw.shape[1])})
        fr = pa.table({"j": np.arange(len(frames), dtype=np.int64), "cx": [f.cx for f in frames],
                       "cy": [f.cy for f in frames], "phi": [f.phi for f in frames], "scale": [f.scale for f in frames]})
        con.register("a", tbl)
        con.register("b", fr)
        con.execute("CREATE TABLE iq AS SELECT * FROM a")
        con.execute("CREATE TABLE frames AS SELECT * FROM b")
        con.close()

    def counts(self, S: list[int], hs: list[Discriminator]) -> np.ndarray:
        import duckdb
        con = duckdb.connect(self.path, read_only=True)
        try:
            con.execute("SET threads = 1")
            fr = {int(r[0]): Frame(*map(float, r[1:])) for r in
                  con.execute("SELECT j, cx, cy, phi, scale FROM frames").fetchall()}
            cols, conds, terms = [], [], []
            for i, (j, h) in enumerate(zip(S, hs)):
                f = fr[j]
                p, q = unit(h.theta)
                cs, sn, sc = float(np.cos(f.phi)), float(np.sin(f.phi)), float(f.scale)
                dx, dy = f"(re{j} - {f.cx!r})", f"(im{j} - {f.cy!r})"
                x = f"(({dx} * {cs!r} + {dy} * {sn!r}) / {sc!r})"
                y = f"((-{dx} * {sn!r} + {dy} * {cs!r}) / {sc!r})"
                cols.append(f"{p!r} * {x} + {q!r} * {y} - {h.beta!r} AS d{i}")
                cols.append(f"1e-12 * (abs({p!r} * {x}) + abs({q!r} * {y}) + {abs(h.beta)!r} "
                            f"+ (abs({dx}) + abs({dy})) / {sc!r}) + 1e-300 AS e{i}")
                cols.append(f"abs({dx}) + abs({dy}) AS g{i}")
                conds.append(f"abs(d{i}) > e{i} AND g{i} >= 1e-280")
                terms.append(f"(CASE WHEN d{i} > 0 THEN {1 << i} ELSE 0 END)")
            base = f"WITH t AS (SELECT rid, {', '.join(cols)} FROM iq)"
            sure = " AND ".join(f"({c})" for c in conds)
            out = np.zeros(1 << len(S), dtype=np.int64)
            for pat, c in con.execute(f"{base} SELECT {' + '.join(terms)} AS pat, count(*) FROM t WHERE {sure} "
                                      "GROUP BY pat").fetchall():
                out[int(pat)] = int(c)
            rows = con.execute(f"SELECT {', '.join(f're{j}, im{j}' for j in S)} FROM iq WHERE rid IN "
                               f"(SELECT rid FROM ({base} SELECT rid FROM t WHERE NOT ({sure})))").fetchall()
        finally:
            con.close()
        if rows:
            arr = np.array(rows, dtype=np.float64)
            idx = np.zeros(arr.shape[0], dtype=np.int64)
            for i, (j, h) in enumerate(zip(S, hs)):
                x, y = fr[j].apply(arr[:, 2 * i] + 1j * arr[:, 2 * i + 1])
                p, q = unit(h.theta)
                idx |= (exact_sign(x, y, p, q, h.beta) > 0).astype(np.int64) << i
            out += np.bincount(idx, minlength=1 << len(S))
        return out
