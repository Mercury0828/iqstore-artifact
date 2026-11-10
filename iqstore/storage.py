"""Byte-exact storage accounting.

Every method serializes its per-position blobs into one container file; b̄ = 8 * file_bytes / (N * m).
The container is self-describing: a counted header carries the method name, N, m and the number of blobs;
every per-position header carries all dimensions, dtype codes and mode flags needed to reconstruct the stored data.
`decode_iqstore_position` rebuilds the partition and the per-shot cell codes from the bytes alone (D1 checks the round
trip for every position). The general-purpose coder is the same for every method: zstd at the best of levels
1-22 (prereg/M2.md section 3), with and without a byte-shuffle for multi-byte arrays (`zbest`), plus the raw (uncompressed) option.
"""
from __future__ import annotations

import os
import pathlib
import struct

import numpy as np
import zstandard as zstd

from .family import Family
from .frame import Frame
from .rep import BAND_BASE, CODE_EXCEPTION, CODE_OMEGA0, CODE_OMEGA1, Partition, PositionRep

LEVELS = tuple(range(1, 23))  # prereg/M2.md section 3: best of zstd levels 1-22, with and without byte-shuffle
_ZC = {lv: zstd.ZstdCompressor(level=lv) for lv in LEVELS}
DTYPES = {0: np.uint8, 1: np.uint16, 2: np.uint32, 3: np.int32, 4: np.float32, 5: np.float64, 6: np.complex64,
          7: np.complex128, 8: np.int64}
DTYPE_CODE = {np.dtype(v): k for k, v in DTYPES.items()}


def _shuffle(a: np.ndarray) -> bytes:
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize)
    return np.ascontiguousarray(b.T).tobytes()


def _unshuffle(raw: bytes, itemsize: int) -> bytes:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(itemsize, -1)
    return np.ascontiguousarray(b.T).tobytes()


def zbest(arr_or_bytes) -> bytes:
    """Smallest of: raw bytes, zstd over levels, zstd of byte-shuffled data (typed arrays, itemsize > 1).
    Tag byte: 0 raw, 1 zstd, 2 zstd+shuffle; for tag 2 the next byte is the item size."""
    if isinstance(arr_or_bytes, (bytes, bytearray)):
        raw, a = bytes(arr_or_bytes), None
    else:
        a = np.ascontiguousarray(arr_or_bytes)
        raw = a.tobytes()
    cands = [b"\x00" + raw]
    sh = _shuffle(a) if a is not None and a.dtype.itemsize > 1 else None
    for lv in LEVELS:
        cands.append(b"\x01" + _ZC[lv].compress(raw))
        if sh is not None:
            cands.append(b"\x02" + bytes([a.dtype.itemsize]) + _ZC[lv].compress(sh))
    return min(cands, key=len)


def zload(blob: bytes) -> bytes:
    tag = blob[0]
    if tag == 0:
        return blob[1:]
    if tag == 1:
        return zstd.ZstdDecompressor().decompress(blob[1:])
    if tag == 2:
        return _unshuffle(zstd.ZstdDecompressor().decompress(blob[2:]), blob[1])
    raise ValueError("unknown zbest tag")


def mask_blob(mask: np.ndarray) -> bytes:
    """Boolean column, shared by every method: the smaller of (b'B') the zbest-coded bitmap
    and (b'I') the zbest-coded gaps between consecutive True indices (sparse masks: band / slab / exception masks)."""
    mask = np.asarray(mask, dtype=bool)
    a = b"B" + zbest(np.packbits(mask).tobytes())
    idx = np.flatnonzero(mask)
    gaps = np.diff(idx, prepend=-1) - 1
    dt = _dtype_for(int(gaps.max(initial=0)))
    b = b"I" + bytes([DTYPE_CODE[np.dtype(dt)]]) + zbest(gaps.astype(dt))
    return a if len(a) <= len(b) else b


def mask_load(blob: bytes, n: int) -> np.ndarray:
    if blob[:1] == b"B":
        return np.unpackbits(np.frombuffer(zload(blob[1:]), dtype=np.uint8))[:n].astype(bool)
    if blob[:1] == b"I":
        gaps = np.frombuffer(zload(blob[2:]), dtype=DTYPES[blob[1]]).astype(np.int64)
        out = np.zeros(n, dtype=bool)
        out[np.cumsum(gaps + 1) - 1] = True
        return out
    raise ValueError("unknown mask tag")


def _dtype_for(maxval: int):
    return np.uint8 if maxval < 256 else (np.uint16 if maxval < 65536 else np.uint32)


# ------------------------------------------------------------------------------------------------ quadtree structure

def _node_decided_np(root, d: np.ndarray, i: np.ndarray, j: np.ndarray, fam: Family) -> np.ndarray:
    """Family status (T2 corner test) of quadtree nodes, with exactly the leaf-geometry formulas of `Partition` (NumPy
    reference; kept for verification of the compiled kernel)."""
    from .family import rect_decided

    q = Partition(tuple(root), i.astype(np.int64), j.astype(np.int64), d.astype(np.int64), np.zeros(d.size, np.int8))
    return rect_decided(q.x0, q.y0, q.x1, q.y1, fam.th_lo, fam.th_hi, fam.b_lo, fam.b_hi)


def _node_decided(root, d: np.ndarray, i: np.ndarray, j: np.ndarray, fam: Family) -> np.ndarray:
    """Family status of quadtree nodes with the compiled kernel shared by the encoder and the decoder (fastdec)."""
    from .fastdec import fam_args, node_status_arr

    return node_status_arr(float(root[0]), float(root[1]), float(root[2]), np.ascontiguousarray(d, dtype=np.int64),
                           np.ascontiguousarray(i, dtype=np.int64), np.ascontiguousarray(j, dtype=np.int64),
                           *fam_args(fam))


def _quadtree_bits(part: Partition, fam: Family, generic: bool = False) -> tuple[bytes, np.ndarray, int]:
    """Quadtree structure, level by level. Nothing that the decoder can recompute from the
    stored root and family is stored: (i) leaf status = T2 corner test of the leaf; (ii) a node decided by the whole
    family is always a leaf (construction refines only undecided leaves and `prune` merges every group of four decided
    same-side siblings, and decidedness is monotone under shrinking), so it carries no split bit. One split bit is
    stored per family-undecided node, levels in increasing depth, nodes of a level in (ix, iy) order. Band leaves are
    ranked in (depth, ix, iy) order. Returns (packed bytes, band rank per leaf (-1 if not band), n_bits).
    `generic=True` (A7: an unpruned, family-agnostic tree may contain decided internal nodes): every node carries a split
    bit (flagged in the header)."""
    if not np.array_equal(part.status, _node_decided(part.root, part.depth, part.ix, part.iy, fam)):
        raise AssertionError("leaf status disagrees with the corner test (status is not stored)")
    leaves = {(int(d), int(i), int(j)) for d, i, j in zip(part.depth, part.ix, part.iy)}
    internal = set()
    for (d, i, j) in leaves:
        while d > 0:
            d, i, j = d - 1, i >> 1, j >> 1
            internal.add((d, i, j))
    bits: list[int] = []
    level = [(0, 0, 0)]
    while level:
        level.sort()
        arr = np.array(level, dtype=np.int64)
        st = _node_decided(part.root, arr[:, 0], arr[:, 1], arr[:, 2], fam)
        nxt = []
        for (d, i, j), s in zip(level, st.tolist()):
            if s >= 0 and not generic:
                if (d, i, j) in internal:
                    raise AssertionError("decided internal node (prune invariant broken)")
                continue
            if (d, i, j) in internal:
                bits.append(1)
                nxt += [(d + 1, 2 * i + di, 2 * j + dj) for di in (0, 1) for dj in (0, 1)]
            else:
                bits.append(0)
        level = nxt
    band = np.flatnonzero(part.status == -1)
    order = band[np.lexsort((part.iy[band], part.ix[band], part.depth[band]))]
    band_rank = np.full(part.n_leaves, -1, dtype=np.int64)
    band_rank[order] = np.arange(order.size)
    if not bits:
        return b"", band_rank, 0
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes(), band_rank, len(bits)


def _quadtree_from_bits(packed: bytes, n_bits: int, root, fam: Family, generic: bool = False) -> tuple[Partition, np.ndarray]:
    """Inverse of `_quadtree_bits` (compiled, `fastdec.decode_tree`). Leaf status comes from the
    same kernel as the encoder's, so the decoded tree and status are the encoder's."""
    from .fastdec import decode_tree

    D, I, J, S = decode_tree(packed, n_bits, root, fam, generic)
    part = Partition(tuple(root), I, J, D, S)
    band_leaves = np.flatnonzero(S == -1)  # (depth, ix, iy) order = band rank order
    return part, band_leaves


# ------------------------------------------------------------------------------------------------ IQSTORE position

_IQ_HDR = "<11d q q q B B"  # root(3) frame(4) family(4) | N, n_tree_bits, n_exc | column-encoding flag, symbol dtype


def iqstore_blobs(rep: PositionRep, generic_tree: bool = False) -> list[bytes]:
    tree, band_rank, n_bits = _quadtree_bits(rep.part, rep.family, generic_tree)
    N = rep.codes.size
    sym = np.empty(N, dtype=np.int64)
    sym[rep.codes == CODE_OMEGA0] = 0
    sym[rep.codes == CODE_OMEGA1] = 1
    sym[rep.codes == CODE_EXCEPTION] = 2
    band = rep.codes >= BAND_BASE
    sym[band] = 3 + band_rank[rep.codes[band] - BAND_BASE]
    n_band = int((band_rank >= 0).sum())
    dt1 = _dtype_for(int(sym.max(initial=0)))
    e1 = [zbest(sym.astype(dt1))]
    nonband = sym < 3
    is_exc = sym[nonband] == 2
    side = sym[nonband][~is_exc] == 1
    bdt = _dtype_for(n_band)
    # band mask (sparse) and, only when exceptions exist, the exception mask over non-band shots (r1b)
    e2 = [mask_blob(~nonband)] + ([mask_blob(is_exc)] if is_exc.any() else []) +          [zbest(np.packbits(side).tobytes()), zbest((sym[~nonband] - 3).astype(bdt))]
    if sum(len(b) + 4 for b in e1) <= sum(len(b) + 4 for b in e2):
        flag, dcode, cols = 1, DTYPE_CODE[np.dtype(dt1)], e1
    else:
        flag, dcode, cols = 2, DTYPE_CODE[np.dtype(bdt)], e2
    header = struct.pack(_IQ_HDR, *rep.part.root, *rep.frame.to_array(), *rep.family.to_array(), N, n_bits,
                         int(rep.exc_index.size), flag + (4 if generic_tree else 0), dcode)
    blobs = [header, zbest(tree)] + cols
    if rep.exc_raw.size:
        blobs.append(bytes([DTYPE_CODE[rep.exc_raw.dtype]]) + zbest(np.ascontiguousarray(rep.exc_raw)))
    return blobs


def decode_iqstore_position(blobs: list[bytes]):
    """Inverse of `iqstore_blobs`: returns (frame, family, partition, codes, exc_index, exc_raw)."""
    h = struct.unpack(_IQ_HDR, blobs[0])
    root, fr, fam = h[0:3], Frame.from_array(np.array(h[3:7])), Family(*h[7:11])
    N, n_bits, n_exc, flag, dcode = h[11], h[12], h[13], h[14], h[15]
    generic = flag >= 4
    flag -= 4 if generic else 0
    part, band_leaves = _quadtree_from_bits(zload(blobs[1]), n_bits, root, fam, generic)
    if flag == 1:
        sym = np.frombuffer(zload(blobs[2]), dtype=DTYPES[dcode]).astype(np.int64)
        rest = blobs[3:]
    else:
        nonband = ~mask_load(blobs[2], N)
        n_nb = int(nonband.sum())
        k = 3
        if n_exc:
            is_exc = mask_load(blobs[3], n_nb)
            k = 4
        else:
            is_exc = np.zeros(n_nb, dtype=bool)
        side = np.unpackbits(np.frombuffer(zload(blobs[k]), dtype=np.uint8))[: int((~is_exc).sum())].astype(bool)
        bandv = np.frombuffer(zload(blobs[k + 1]), dtype=DTYPES[dcode]).astype(np.int64)
        sym = np.empty(N, dtype=np.int64)
        nb_sym = np.empty(n_nb, dtype=np.int64)
        nb_sym[is_exc] = 2
        nb_sym[~is_exc] = side.astype(np.int64)
        sym[nonband] = nb_sym
        sym[~nonband] = 3 + bandv
        rest = blobs[k + 2:]
    codes = np.empty(N, dtype=np.int32)
    codes[sym == 0] = CODE_OMEGA0
    codes[sym == 1] = CODE_OMEGA1
    codes[sym == 2] = CODE_EXCEPTION
    bl = np.array(band_leaves, dtype=np.int64)
    m = sym >= 3
    codes[m] = BAND_BASE + bl[sym[m] - 3]
    exc_index = np.flatnonzero(sym == 2)
    exc_raw = np.zeros(0)
    if n_exc:
        exc_raw = np.frombuffer(zload(rest[0][1:]), dtype=DTYPES[rest[0][0]])
    return fr, fam, part, codes, exc_index, exc_raw


# ------------------------------------------------------------------------------------------------ container

def write_container(path: pathlib.Path, blobs: list[bytes], method: str = "", N: int = 0, m: int = 0) -> int:
    """Counted header (method, N, m, number of blobs) + length-prefixed blobs. Returns the file size on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = method.encode("utf-8")[:255]
    with open(path, "wb") as f:
        f.write(struct.pack("<B", len(name)) + name + struct.pack("<qqI", N, m, len(blobs)))
        for b in blobs:
            f.write(struct.pack("<I", len(b)))
            f.write(b)
    return path.stat().st_size


def read_file(path) -> bytes:
    """Whole file with few system calls. Every engine that reads a whole container reads through it. A first read of
    64 KiB returns a small file whole (open, read, close: no fstat); a larger file is then read again from offset 0 in
    one call of its exact size (fstat, lseek, read), so its bytes are not joined from pieces. (A large fixed read
    buffer is not used: os.read allocates the requested size, which costs milliseconds for large requests.)"""
    fd = os.open(path, _OFLAGS)
    try:
        data = os.read(fd, _FIRST)
        if len(data) < _FIRST:
            return data
        n = os.fstat(fd).st_size
        os.lseek(fd, 0, 0)
        data = os.read(fd, n)
        while len(data) < n:
            more = os.read(fd, n - len(data))
            if not more:
                break
            data += more
        return data
    finally:
        os.close(fd)


_OFLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0)
_FIRST = 65536


def read_container(path: pathlib.Path) -> tuple[str, int, int, list]:
    """(method, N, m, blobs); blobs are memoryviews into one read of the file."""
    data = read_file(path)
    mv = memoryview(data)
    ln = data[0]
    name = data[1:1 + ln].decode("utf-8")
    N, m, nb = struct.unpack_from("<qqI", data, 1 + ln)
    pos = 1 + ln + 20
    blobs = []
    for _ in range(nb):
        (L,) = struct.unpack_from("<I", data, pos)
        blobs.append(mv[pos + 4:pos + 4 + L])
        pos += 4 + L
    return name, N, m, blobs


def bits_per_shot_position(nbytes: int, N: int, m: int) -> float:
    return 8.0 * nbytes / (N * m)
