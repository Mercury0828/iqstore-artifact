"""Phase-0 storage baselines B0, B1, B2 (zfp, SZ3), B4, B4q, same container and coder as IQSTORE.

Certificates use the SAME sound certifier as IQSTORE (`iqstore.partition.certify`): a method's stored cell for a shot is a
rectangle (B2: the closed L-inf square of half-width e around the reconstruction, e = measured max error + frame
rounding (amendment A4); B4q: its grid cell), and the certified worst-case uncertain count over the family must be
<= floor(tau * N). Every payload gets the same general-purpose coder choice (`zbest`) as IQSTORE; every
header carries the actual side information (frame, family, reference, margins, steps, dtypes) needed to answer queries; lossless paths keep the raw dtype as returned.
"""
from __future__ import annotations

import struct

import numpy as np

from iqstore.family import Discriminator, Family, rect_decided
from iqstore.frame import Frame
from iqstore.partition import certify
from iqstore.storage import DTYPE_CODE, mask_blob, zbest

# ---------------------------------------------------------------------------------------------------- B0, B1


def b0_blobs(raw: np.ndarray, frames: list[Frame], fams: list[Family]) -> list[bytes]:
    """raw: (N, m) complex IQ exactly as returned (dtype kept). Best of per-position columns vs whole matrix. The
    per-position frame and family are stored too: a frame-parameterized query needs them."""
    raw = np.ascontiguousarray(raw)
    fdt = np.float32 if raw.dtype == np.complex64 else np.float64
    per_col = [zbest(np.ascontiguousarray(raw[:, j]).view(fdt)) for j in range(raw.shape[1])]
    whole = [zbest(raw.view(fdt))]
    use_cols = sum(len(b) + 4 for b in per_col) <= sum(len(b) + 4 for b in whole)
    hdr = struct.pack("<BqqB", DTYPE_CODE[raw.dtype], raw.shape[0], raw.shape[1], 1 if use_cols else 2)
    side = b"".join(struct.pack("<8d", *f.to_array(), *fa.to_array()) for f, fa in zip(frames, fams))
    return [hdr, side] + (per_col if use_cols else whole)


def b1_bitarray_bytes(N: int, m: int) -> int:
    """Qiskit BitArray in-memory packing: ceil(m/8) bytes per shot (G2)."""
    return N * (m // 8 + (m % 8 > 0))


def b1_blobs(labels: np.ndarray) -> list[bytes]:
    """labels (N, m) in {0,1}: bit-packed rows + zstd (header: N, m)."""
    return [struct.pack("<qq", *labels.shape), zbest(np.packbits(labels.astype(np.uint8), axis=1).tobytes())]


# ---------------------------------------------------------------------------------------------------- B2


def _zfp_roundtrip(a: np.ndarray, eps: float) -> tuple[bytes, np.ndarray]:
    import zfpy

    comp = zfpy.compress_numpy(np.ascontiguousarray(a), tolerance=eps)
    return comp, zfpy.decompress_numpy(comp)


def _sz3_roundtrip(a: np.ndarray, eps: float) -> tuple[bytes, np.ndarray]:
    from pysz import sz, szConfig, szErrorBoundMode

    cfg = szConfig()
    cfg.errorBoundMode = szErrorBoundMode.ABS
    cfg.absErrorBound = float(eps)
    comp, _ratio = sz.compress(np.ascontiguousarray(a), cfg)
    comp_b = bytes(np.asarray(comp).tobytes())
    rec, _cfg = sz.decompress(comp, a.dtype, a.shape)
    return comp_b, np.asarray(rec).reshape(a.shape)


def b2_position(x: np.ndarray, y: np.ndarray, raw_col: np.ndarray, frame: Frame, fam: Family, tau: float, codec: str,
                eps_grid: np.ndarray):
    """Certified candidate search: every eps is tried, its actual reconstruction is certified with
    e = measured max error (+ float32 rounding of the frame coordinates); the certified candidate with the smallest
    complete stored size wins (no bisection). If no candidate certifies, B2 stores the raw column losslessly (exact).
    Returns (blobs, info)."""
    N = x.size
    budget = int(np.floor(tau * N))
    xf = x.astype(np.float32)
    yf = y.astype(np.float32)
    rnd = float(max(np.max(np.abs(xf.astype(np.float64) - x)), np.max(np.abs(yf.astype(np.float64) - y))))
    rt = _zfp_roundtrip if codec == "zfp" else _sz3_roundtrip
    cands = []
    for eps in eps_grid:
        cx, rx = rt(xf, float(eps))
        cy, ry = rt(yf, float(eps))
        rx = rx.astype(np.float64)
        ry = ry.astype(np.float64)
        e = float(max(np.max(np.abs(rx - xf)), np.max(np.abs(ry - yf)))) + rnd
        hdr = struct.pack("<B4dd4dq", 1 if codec == "zfp" else 2, *frame.to_array(), e, *fam.to_array(), N)
        blobs = [hdr, zbest(cx), zbest(cy)]
        cands.append((sum(len(b) + 4 for b in blobs), float(eps), e, rx, ry, blobs))
    for size, eps, e, rx, ry, blobs in sorted(cands, key=lambda c: c[0]):
        res = certify(rx - e, ry - e, rx + e, ry + e, np.ones(N, dtype=np.int64), fam, budget)
        if res.ok:
            return blobs, {"mode": "lossy", "eps": eps, "e": e, "certified_upper": res.upper, "bytes": size,
                           "rx": rx, "ry": ry}
    raw_col = np.ascontiguousarray(raw_col)
    hdr = struct.pack("<B4dd4dqB", 0, *frame.to_array(), 0.0, *fam.to_array(), N, DTYPE_CODE[raw_col.dtype])
    blobs = [hdr, zbest(raw_col)]
    return blobs, {"mode": "lossless-fallback", "bytes": sum(len(b) + 4 for b in blobs)}


# ---------------------------------------------------------------------------------------------------- B4, B4q


def _clip_box_halfplane(X0, Y0, X1, Y1, ux, uy, c, ge: bool) -> np.ndarray:
    """Vertices of [X0,X1]x[Y0,Y1] ∩ {ux x + uy y >= c} (ge) or {<= c} (not ge) (Sutherland-Hodgman)."""
    pts = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]
    s = 1.0 if ge else -1.0
    out = []
    for i in range(4):
        p, q = pts[i], pts[(i + 1) % 4]
        fp = s * (ux * p[0] + uy * p[1] - c)
        fq = s * (ux * q[0] + uy * q[1] - c)
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return np.array(out) if out else np.zeros((0, 2))


def _region_decided(verts: np.ndarray, fam: Family, want: int) -> bool:
    if verts.shape[0] == 0:
        return True
    st = rect_decided(verts[:, 0], verts[:, 1], verts[:, 0], verts[:, 1], fam.th_lo, fam.th_hi, fam.b_lo, fam.b_hi)
    return bool(np.all(st == want))


def fixed_margins(ref: Discriminator, fam: Family, box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Smallest margins (M0, M1) such that root ∩ {v <= -M0} is decided-0 and root ∩ {v >= M1} is decided-1 for the
    whole family, v = <u_ref, z> - beta_ref (bisection; decidedness is monotone under shrinking the region). The
    clipped polygon is convex, so its vertices decide it (T2). `box` = the TIGHT bounding box of the stored points
    (closed), which gives B4 its smallest valid margins (baseline at its best)."""
    X0, Y0, X1, Y1 = box
    side = max(X1 - X0, Y1 - Y0)
    ux, uy = np.cos(ref.theta), np.sin(ref.theta)
    out = []
    for want, ge in ((0, False), (1, True)):
        lo, hi = 0.0, 4.0 * side
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            c = ref.beta + mid if ge else ref.beta - mid
            if _region_decided(_clip_box_halfplane(X0, Y0, X1, Y1, ux, uy, c, ge), fam, want):
                hi = mid
            else:
                lo = mid
        out.append(hi)
    return out[0], out[1]


def b4_slab(x, y, ref: Discriminator, M0: float, M1: float) -> tuple[np.ndarray, np.ndarray]:
    """(in_slab mask, side bit for points outside the slab)."""
    v = np.cos(ref.theta) * x + np.sin(ref.theta) * y - ref.beta
    side1 = v >= M1
    side0 = v <= -M0
    return ~(side0 | side1), side1


def _b4_header(kind: int, frame: Frame, ref: Discriminator, M0: float, M1: float, d: float, N: int, dt: int,
               fam: Family) -> bytes:
    return struct.pack("<B4d2d2dd4dqB", kind, *frame.to_array(), ref.theta, ref.beta, M0, M1, d, *fam.to_array(), N, dt)


def b4_blobs(raw_col: np.ndarray, in_slab: np.ndarray, side1: np.ndarray, frame: Frame, ref: Discriminator,
             M0: float, M1: float, fam: Family) -> list[bytes]:
    """B4: side bits outside the slab + lossless raw IQ (raw dtype) of slab points (exact in-family)."""
    raw_col = np.ascontiguousarray(raw_col)
    return [_b4_header(4, frame, ref, M0, M1, 0.0, raw_col.size, DTYPE_CODE[raw_col.dtype], fam),
            mask_blob(in_slab), zbest(np.packbits(side1[~in_slab]).tobytes()),
            zbest(np.ascontiguousarray(raw_col[in_slab]))]


def b4q_position(x, y, in_slab, side1, raw_col, frame: Frame, ref: Discriminator, M0: float, M1: float, fam: Family,
                 tau: float, deltas: np.ndarray):
    """B4q: slab points quantized on a uniform grid of step delta (anchored at 0); the certified candidate with the
    smallest complete stored size wins. Falls back to B4 (exact band) if no step certifies. Returns (blobs, info)."""
    N = x.size
    budget = int(np.floor(tau * N))
    xs, ys = x[in_slab], y[in_slab]
    cands = []
    for d in deltas:
        kx = np.floor(xs / d).astype(np.int64)
        ky = np.floor(ys / d).astype(np.int64)
        kx = np.where(xs < kx * d, kx - 1, kx)
        kx = np.where(xs >= (kx + 1) * d, kx + 1, kx)
        ky = np.where(ys < ky * d, ky - 1, ky)
        ky = np.where(ys >= (ky + 1) * d, ky + 1, ky)
        keys = np.stack([kx, ky], axis=1)
        uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        idt = np.uint8 if uk.shape[0] < 256 else (np.uint16 if uk.shape[0] < 65536 else np.uint32)
        blobs = [_b4_header(5, frame, ref, M0, M1, float(d), N, DTYPE_CODE[np.dtype(idt)], fam),
                 mask_blob(in_slab), zbest(np.packbits(side1[~in_slab]).tobytes()),
                 struct.pack("<q", uk.shape[0]) + zbest(uk.astype(np.int32)), zbest(inv.ravel().astype(idt))]
        cands.append((sum(len(b) + 4 for b in blobs), float(d), uk, cnt, blobs))
    for size, d, uk, cnt, blobs in sorted(cands, key=lambda c: c[0]):
        res = certify(uk[:, 0] * d, uk[:, 1] * d, (uk[:, 0] + 1) * d, (uk[:, 1] + 1) * d, cnt, fam, budget)
        if res.ok:
            return blobs, {"mode": "grid", "delta": d, "certified_upper": res.upper, "bytes": size}
    bl = b4_blobs(raw_col, in_slab, side1, frame, ref, M0, M1, fam)
    return bl, {"mode": "B4-fallback", "bytes": sum(len(b) + 4 for b in bl)}


# ---------------------------------------------------------------------------------------------------- B3, A6 storage


def b3_blobs(x: np.ndarray, y: np.ndarray, frame: Frame, bins: int) -> list[bytes]:
    """B3: per-position 2-D histogram (bins x bins) over the closed data box in frame coordinates; counts uint32
    (zbest) + frame + box. No per-shot information (joint queries cannot be answered); no certificate."""
    box = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[[box[0], box[1]], [box[2], box[3]]])
    hdr = struct.pack("<4d4dq", *frame.to_array(), *box, bins)
    return [hdr, zbest(h.astype(np.uint32))]


def a6_blobs(raw_col: np.ndarray, frame: Frame, ref: Discriminator, block: int, cache: dict | None = None) -> list[bytes]:
    """A6 (Hazy-style, prereg/M2.md section 3) persistent bytes of one position: raw column (B0 coder), anchor (theta,
    beta, p, q), anchor distances d~ (float64), the sorting permutation (int32), per block of `block` sorted shots the
    certified radius and error bounds (float64 x 2). Bounds computed with outward rounding."""
    from iqstore.family import unit

    raw_col = np.ascontiguousarray(raw_col)
    x, y = frame.apply(raw_col)
    p, q = unit(ref.theta)
    d = (p * x + q * y) - ref.beta
    perm = np.argsort(d, kind="stable").astype(np.int32)
    ds_ = d[perm]
    eps = np.finfo(np.float64).eps
    up = lambda v: np.nextafter(v, np.inf)  # noqa: E731  (one ulp outward after every rounded step)
    s2 = up(up(up(x * x) + up(y * y)) * (1 + 4 * eps))
    r = up(up(np.sqrt(s2)) * (1 + 4 * eps))[perm]
    e = up(up(2 * eps * up(up(up(np.abs(p * x)) + up(np.abs(q * y))) + abs(ref.beta)) * (1 + 4 * eps))
           + np.finfo(np.float64).tiny)[perm]
    nb = -(-x.size // block)
    R = np.array([np.nextafter(r[i * block:(i + 1) * block].max(), np.inf) for i in range(nb)])
    E = np.array([np.nextafter(e[i * block:(i + 1) * block].max(), np.inf) for i in range(nb)])
    fdt = np.float32 if raw_col.dtype == np.complex64 else np.float64
    hdr = struct.pack("<4d4dqq", *frame.to_array(), ref.theta, ref.beta, p, q, x.size, block)
    if cache is None:
        cache = {}
    if "fixed" not in cache:  # the block-size-independent blobs (identical bytes for every block size)
        cache["fixed"] = (zbest(raw_col.view(fdt)), zbest(ds_), zbest(perm))
    return [hdr, *cache["fixed"], zbest(R), zbest(E)]
