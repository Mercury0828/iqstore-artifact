"""Generate the evaluation numbers, tables and figures of the paper from the TEST outputs in runs/m2_test.

Writes paper/generated/{numbers.tex, tab_storage.tex, tab_query.tex, fig_storage.pdf, fig_query.pdf, results.json}.
Statistic: median over evaluation datasets of the per-dataset ratio; 95 % CI from a bootstrap over whole epochs
(10,000 replicates, fixed seed). Usage: python paper/gen_results.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
T = ROOT / "runs" / "m2_test"
G = ROOT / "paper" / "generated"
G.mkdir(exist_ok=True)
rng = np.random.default_rng(20261110)
FAMS = ("F-boot", "F-drift")
FAMS_PAPER = ("F-drift", "F-boot")  # drift sets lead
FAMNAME = {"F-boot": "Boot", "F-drift": "Drift"}
TAUS = ("0.01", "0.001", "0.0001")
CLS = "12345"
CLSNAME = {"1": "single", "2": "pair", "3": "3--4 pos.", "4": "mitigated", "5": "sweep"}


def boot(vals, epochs, B=10000):
    vals, epochs = np.asarray(vals, float), np.asarray(epochs)
    med = float(np.median(vals))
    ue = np.unique(epochs)
    if len(ue) < 2:
        return [med, float("nan"), float("nan")]
    groups = [vals[epochs == e] for e in ue]
    bs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(ue), len(ue))
        bs[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return [med, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


q1 = {f.stem: json.loads(f.read_text()) for f in sorted((T / "q1").glob("*.json"))}
cov = json.loads((ROOT / "paper" / "coverage.json").read_text()) if (ROOT / "paper" / "coverage.json").exists() else None
EPOCH = {n: d["epoch"] for n, d in q1.items()}
if cov:
    EPOCH.update({n: d["epoch"] for n, d in cov["datasets"].items()})

def b0_layout(name: str) -> str:
    """'block' (whole matrix as one blob) or 'columns' (one blob per position), read from the B0 container header
    exactly as eval/b0q.py B0FileSel parses it."""
    import struct
    with open(T / "q2" / "containers" / name / "B0.bin", "rb") as fh:
        ln = fh.read(1)[0]
        fh.seek(1 + ln + 20)
        (L,) = struct.unpack("<I", fh.read(4))
        hdr = fh.read(L)
    return "columns" if struct.unpack("<BqqB", hdr)[3] == 1 else "block"


R: dict = {"q1_n": len(q1)}
# ---------------- storage
st = {}
for fam in FAMS:
    for tau in TAUS:
        k = f"{fam}|{tau}"
        rows = [d for d in q1.values() if k in d]
        ep = [d["epoch"] for d in rows]
        r = {"IQ": boot([d[k]["IQSTORE"] for d in rows], ep), "IQmax": max(d[k]["IQSTORE"] for d in rows)}
        for m in ("B4q", "B4", "A7", "B2-zfp", "B2-SZ3"):
            r[m] = float(np.median([d[k][m] for d in rows]))
            r[m + "/IQ"] = boot([d[k][m] / d[k]["IQSTORE"] for d in rows], ep)
        r["B2/IQ"] = boot([min(d[k]["B2-zfp"], d[k]["B2-SZ3"]) / d[k]["IQSTORE"] for d in rows], ep)
        r["B0/IQ"] = boot([d["B0"] / d[k]["IQSTORE"] for d in rows], ep)
        r["B0"] = float(np.median([d["B0"] for d in rows]))
        r["B1"] = float(np.median([d["B1_zstd"] for d in rows]))
        st[k] = r
R["storage"] = st
# ---------------- correctness
he = [json.loads(f.read_text()) for f in sorted((T / "he").glob("*.json"))]
R["he"] = {"files": len(he), "datasets": len({h["dataset"] for h in he}), "pairs": sum(h["pairs"] for h in he),
           "statistics": sum(h["statistics"] for h in he), "violations": sum(h["n_violations"] for h in he),
           "width_violations": sum(h["n_width_bound_violations"] for h in he)}
# ---------------- queries
qt = {}
for base in FAMS:
    D = [json.loads(f.read_text()) for f in sorted((T / "q2" / "time").glob(f"*_{base}.json"))]
    ep = [EPOCH.get(d["dataset"], d["dataset"]) for d in D]
    r = {"n": len(D), "datasets": [d["dataset"] for d in D], "pairs": sum(d["pairs"] for d in D)}
    for c in CLS:
        a = [d["class_median_s_all_pairs"] for d in D]
        mm = [d["class_median_s_matched_pairs_same_mask"] for d in D]
        x = {"IQ_ms": float(np.median([v["IQSTORE"][c] for v in a]) * 1e3),
             "B0q_ms": float(np.median([v["B0q"][c] for v in a]) * 1e3),
             "B4q_eq_ms": float(np.median([v["B4q_eq"][c] for v in a]) * 1e3),
             "B0q/IQ": boot([v["B0q"][c] / v["IQSTORE"][c] for v in a], ep),
             "B0q/IQ_min": float(min(v["B0q"][c] / v["IQSTORE"][c] for v in a)),
             "B4q_eq/IQ": boot([v["B4q_eq"][c] / v["IQSTORE"][c] for v in a], ep),
             "unmatched": float(1 - sum(d["matched_pairs_per_class"][c] for d in D)
                                / sum(d["pairs_per_class"][c] for d in D))}
        ok = [(v, e) for v, e, d in zip(mm, ep, D) if d["matched_pairs_per_class"][c] > 0]
        if ok:
            x["B4q_m/IQ"] = boot([v["B4q_matched"][c] / v["IQSTORE"][c] for v, _ in ok], [e for _, e in ok])
        r[c] = x
    r["op_ms"] = float(np.median([d["class4_operator_s_median"] for d in D]) * 1e3)
    wl = {}
    for d in D:
        w = b0_layout(d["dataset"])
        rr = [d["class_median_s_all_pairs"]["B0q"][c] / d["class_median_s_all_pairs"]["IQSTORE"][c] for c in CLS]
        wl.setdefault(w, []).append((min(rr), max(rr)))
    r["by_workload"] = {w: {"datasets": len(v), "min": min(a for a, _ in v), "max": max(b for _, b in v)}
                        for w, v in wl.items()}
    r["below10"] = sum(1 for d in D for c in CLS
                       if d["class_median_s_all_pairs"]["B0q"][c] / d["class_median_s_all_pairs"]["IQSTORE"][c] < 10)
    qt[base] = r
R["query"] = qt
R["coverage"] = {k: v for k, v in cov.items() if k != "datasets"} if cov else None


# ---------------- maintenance (H-c event stream; a partial stream is read up to its last complete line)
def _ratio(rows, other):
    tot = {}
    for r in rows:
        a = tot.setdefault(r["dataset"], [0.0, 0.0])
        a[0] += r[other]["median"]; a[1] += r["iq"]["median"]
    return {k: v[0] / v[1] for k, v in tot.items() if v[1] > 0}


mt = {}
for fam in FAMS:
    f = T / "hc" / f"events_{fam}.jsonl"
    if not f.exists():
        continue
    lines = []
    for s in f.read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.loads(s))
        except json.JSONDecodeError:
            break
    ev = [r for r in lines if r.get("kind") in ("real", "synthetic") and r["split"] == "TEST"]
    groups = {"real in-family": [r for r in ev if r["kind"] == "real" and r["in_family"]],
              "real out-of-family": [r for r in ev if r["kind"] == "real" and not r["in_family"]],
              "synthetic in-family": [r for r in ev if r["kind"] == "synthetic" and r["in_family"]],
              "synthetic out-of-family": [r for r in ev if r["kind"] == "synthetic" and not r["in_family"]]}
    g = {"complete": bool(lines and lines[-1].get("footer")), "events": len(ev),
         "datasets": len({r["dataset"] for r in ev})}
    for name, rows in groups.items():
        if not rows:
            continue
        rb = _ratio(rows, "b6"); ra = _ratio(rows, "a6")
        ds = sorted(rb)
        ep = [EPOCH.get(n, n) for n in ds]
        g[name] = {"events": len(rows), "datasets": len(ds),
                   "B6/IQ": boot([rb[n] for n in ds], ep), "A6/IQ": boot([ra[n] for n in ds], ep),
                   "iq_ms": float(np.median([r["iq"]["median"] for r in rows]) * 1e3),
                   "b6_ms": float(np.median([r["b6"]["median"] for r in rows]) * 1e3),
                   "a6_ms": float(np.median([r["a6"]["median"] for r in rows]) * 1e3),
                   "recomputed": sum(r["views_recomputed"] > 0 for r in rows)}
    mt[fam] = g
R["maintenance"] = mt
(G / "results.json").write_text(json.dumps(R, indent=1))


# ---------------- tex
def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def ci(v, fmt=f1):
    return f"{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]"


m = []
s = st["F-boot|0.001"]; s2 = st["F-drift|0.001"]
m += [rf"\newcommand{{\NqOne}}{{{len(q1)}}}",
      rf"\newcommand{{\IQbitsBoot}}{{{f2(s['IQ'][0])}}}", rf"\newcommand{{\IQbitsDrift}}{{{f2(s2['IQ'][0])}}}",
      rf"\newcommand{{\IQbitsMax}}{{{f2(max(s['IQmax'], s2['IQmax']))}}}",
      rf"\newcommand{{\BzeroBits}}{{{f1(s['B0'])}}}",
      rf"\newcommand{{\RBzeroBoot}}{{{f1(s['B0/IQ'][0])}}}", rf"\newcommand{{\RBzeroDrift}}{{{f1(s2['B0/IQ'][0])}}}",
      rf"\newcommand{{\RBtwoBoot}}{{{f1(s['B2/IQ'][0])}}}", rf"\newcommand{{\RBtwoDrift}}{{{f1(s2['B2/IQ'][0])}}}",
      rf"\newcommand{{\RBtwoBootLo}}{{{f1(s['B2/IQ'][1])}}}", rf"\newcommand{{\RBtwoDriftLo}}{{{f1(s2['B2/IQ'][1])}}}",
      rf"\newcommand{{\RBzeroBootLo}}{{{f1(s['B0/IQ'][1])}}}", rf"\newcommand{{\RBzeroDriftLo}}{{{f1(s2['B0/IQ'][1])}}}",
      rf"\newcommand{{\RBfourqBoot}}{{{f2(s['B4q/IQ'][0])}}}", rf"\newcommand{{\RBfourqDrift}}{{{f2(s2['B4q/IQ'][0])}}}",
      rf"\newcommand{{\RBfourqDriftCI}}{{{ci(s2['B4q/IQ'], f2)}}}",
      rf"\newcommand{{\RBfourqDriftLow}}{{{f2(st['F-drift|0.0001']['B4q/IQ'][0])}}}",
      rf"\newcommand{{\RAsevenBoot}}{{{f2(s['A7/IQ'][0])}}}", rf"\newcommand{{\RAsevenDrift}}{{{f2(s2['A7/IQ'][0])}}}",
      rf"\newcommand{{\RBfourBoot}}{{{f2(s['B4/IQ'][0])}}}", rf"\newcommand{{\RBfourDrift}}{{{f2(s2['B4/IQ'][0])}}}",
      rf"\newcommand{{\HEdatasets}}{{{R['he']['datasets']}}}",
      rf"\newcommand{{\HEpairs}}{{{R['he']['pairs']:,}}}".replace(",", "{,}"),
      rf"\newcommand{{\HEstats}}{{{R['he']['statistics']:,}}}".replace(",", "{,}"),
      rf"\newcommand{{\HEviol}}{{{R['he']['violations'] + R['he']['width_violations']}}}"]
for base in FAMS:
    q = qt[base]; nm = FAMNAME[base]
    rs = [q[c]["B0q/IQ"][0] for c in CLS]
    los = [q[c]["B0q/IQ"][1] for c in CLS]
    m += [rf"\newcommand{{\Nq{nm}}}{{{q['n']}}}",
          rf"\newcommand{{\QminRatio{nm}}}{{{f1(min(rs))}}}", rf"\newcommand{{\QmaxRatio{nm}}}{{{f1(max(rs))}}}",
          rf"\newcommand{{\QminLo{nm}}}{{{f1(min(los))}}}",
          rf"\newcommand{{\Qpairs{nm}}}{{{q['pairs']:,}}}".replace(",", "{,}")]
m.append(rf"\newcommand{{\BoneBits}}{{{f2(st['F-boot|0.001']['B1'])}}}")
# width-unmatched shares from the complete (untimed) plan of the full workload
um = {}
for base in FAMS:
    tot = {c: 0 for c in CLS}; un = {c: 0 for c in CLS}
    for f in sorted((T / "q2" / "plan").glob(f"*_{base}.npz")):
        z = np.load(f)
        for c in CLS:
            sel = z["cls"] == int(c)
            tot[c] += int(sel.sum()); un[c] += int((z["matched"][sel] < 0).sum())
    um[base] = {c: 100 * un[c] / tot[c] for c in CLS if tot[c]}
R["unmatched_plan_pct"] = um
m += [rf"\newcommand{{\UnmatchedBootMin}}{{{min(um['F-boot'].values()):.0f}}}",
      rf"\newcommand{{\UnmatchedBootMax}}{{{max(um['F-boot'].values()):.0f}}}",
      rf"\newcommand{{\UnmatchedDriftMax}}{{{max(um['F-drift'].values()):.1f}}}"]
(G / "results.json").write_text(json.dumps(R, indent=1))
WLNAME = {"GHZ": "GHZ", "RCL": "random Clifford", "PROD": "product-state", "REP": "repetition-code"}
parts = []
for w in ("GHZ", "RCL", "PROD", "REP"):
    xs = [qt[b]["by_workload"][w] for b in FAMS if w in qt[b]["by_workload"]]
    if xs:
        parts.append(f"{f1(min(x['min'] for x in xs))}--{f1(max(x['max'] for x in xs))}$\\times$ on the "
                     f"{WLNAME[w]} datasets")
for lay, mac in (("columns", "PerCol"), ("block", "Block")):
    pc = [qt[b]["by_workload"][lay] for b in FAMS if lay in qt[b]["by_workload"]]
    if pc:
        m.append(rf"\newcommand{{\{mac}Range}}{{{f1(min(x['min'] for x in pc))}--{f1(max(x['max'] for x in pc))}}}")
        m.append(rf"\newcommand{{\{mac}N}}{{{max(x['datasets'] for x in pc)}}}")
LAY = {}
for base in FAMS:
    for n in qt[base]["datasets"]:
        LAY[n] = b0_layout(n)
R["b0_layout"] = LAY
(G / "results.json").write_text(json.dumps(R, indent=1))
blk = sorted(n for n, v in LAY.items() if v == "block")
m.append(rf"\newcommand{{\BlockList}}{{{', '.join(n.replace('_', r'\_') for n in blk)}}}")
if False:
    m.append(r"\renewcommand{\QUERYWORKLOAD}{Per dataset and class, the ratio ranges over " + ", ".join(parts) + ".}")
mx = mt.get("F-drift")
if mx and "real in-family" in mx:
    a, b, c = mx["real in-family"], mx.get("real out-of-family"), mx.get("synthetic in-family")
    m += [rf"\newcommand{{\MtEvents}}{{{mx['events']:,}}}".replace(",", "{,}"),
          rf"\newcommand{{\MtDatasets}}{{{mx['datasets']}}}",
          rf"\newcommand{{\MtInN}}{{{a['events']}}}", rf"\newcommand{{\MtInRecomp}}{{{a['recomputed']}}}", rf"\newcommand{{\MtInMed}}{{{f1(a['B6/IQ'][0])}}}", rf"\newcommand{{\MtInBsix}}{{{ci(a['B6/IQ'])}}}",
          rf"\newcommand{{\MtInAsix}}{{{ci(a['A6/IQ'])}}}",
          rf"\newcommand{{\MtInIQms}}{{{a['iq_ms']:.2f}}}", rf"\newcommand{{\MtInBsixms}}{{{a['b6_ms']:.1f}}}"]
    if b:
        m += [rf"\newcommand{{\MtOutN}}{{{b['events']}}}", rf"\newcommand{{\MtOutBsix}}{{{f1(b['B6/IQ'][0])}}}",
              rf"\newcommand{{\MtOutRecomp}}{{{b['recomputed']}}}"]
    if c:
        m += [rf"\newcommand{{\MtSynN}}{{{c['events']}}}", rf"\newcommand{{\MtSynBsix}}{{{ci(c['B6/IQ'])}}}",
              rf"\newcommand{{\MtSynAsix}}{{{f1(c['A6/IQ'][0])}}}"]
mb = mt.get("F-boot")
if mb:
    for key, mac in (("real in-family", "In"), ("real out-of-family", "Out"), ("synthetic in-family", "Syn")):
        if key in mb:
            m += [rf"\newcommand{{\MtB{mac}N}}{{{mb[key]['events']}}}",
                  rf"\newcommand{{\MtB{mac}Bsix}}{{{ci(mb[key]['B6/IQ'])}}}"]
    m.append(rf"\newcommand{{\MtBEvents}}{{{mb['events']:,}}}".replace(",", "{,}"))
if cov:
    for base in FAMS:
        c = cov.get(base)
        if c:
            m.append(rf"\newcommand{{\Cov{FAMNAME[base]}}}{{{100 * c['inside'] / c['total']:.1f}}}")
            m.append(rf"\newcommand{{\CovN{FAMNAME[base]}}}{{{c['total']:,}}}".replace(",", "{,}"))
(G / "numbers.tex").write_text("\n".join(m) + "\n")

# storage table: tau rows x families
L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
     r"sets & $\tau$ & IQSTORE & B0 & B2 & B4 & B4q & A7\\", r"\midrule"]
for fam in FAMS_PAPER:
    for tau in TAUS:
        x = st[f"{fam}|{tau}"]
        b2 = min(x["B2-zfp"], x["B2-SZ3"])
        L.append(rf"{FAMNAME[fam]} & $10^{{{int(round(np.log10(float(tau))))}}}$ & {f2(x['IQ'][0])} & {f1(x['B0'])} & "
                 rf"{f2(b2)} & {f2(x['B4'])} & {f2(x['B4q'])} & {f2(x['A7'])}\\")
    L.append(r"\addlinespace")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
(G / "tab_storage.tex").write_text("\n".join(L) + "\n")

# query table
L = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
     r"class & \makecell{IQSTORE\\(ms)} & \makecell{B0q\\(ms)} & \makecell{B0q/IQSTORE\\{}[95\,\% CI]} & \makecell{B4q/\\IQSTORE}\\", r"\midrule"]
for base in FAMS_PAPER:
    q = qt[base]
    L.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{FAMNAME[base]} sets ({q['n']} datasets)}}}}\\")
    for c in CLS:
        x = q[c]
        L.append(rf"{CLSNAME[c]} & {x['IQ_ms']:.3f} & {x['B0q_ms']:.2f} & {ci(x['B0q/IQ'])} & {f2(x['B4q_eq/IQ'][0])}\\")
    L.append(r"\addlinespace")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
(G / "tab_query.tex").write_text("\n".join(L) + "\n")

# figure: storage vs tau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
tv = [float(t) for t in TAUS]
for ax, fam in zip(axs, FAMS):
    for meth, key, sty in (("IQSTORE", "IQ", "o-"), ("B4q", "B4q", "s--"), ("A7", "A7", "^:"),
                           ("B2", None, "d-."), ("B0", "B0", "k-")):
        ys = []
        for t in TAUS:
            x = st[f"{fam}|{t}"]
            ys.append(x["IQ"][0] if key == "IQ" else (min(x["B2-zfp"], x["B2-SZ3"]) if key is None else x[key]))
        ax.plot(tv, ys, sty, label=meth, ms=3, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"width target $\tau$"); ax.set_title(f"{FAMNAME[fam]} sets", fontsize=8)
axs[0].set_ylabel("bits per shot and position")
axs[1].legend(fontsize=6, frameon=False, loc="center right")
fig.tight_layout(pad=0.3)
fig.savefig(G / "fig_storage.pdf")
exec(open(pathlib.Path(__file__).with_name("gen_extra.py"), encoding="utf-8").read())
print(json.dumps({"q1": len(q1), "time": {b: qt[b]["n"] for b in FAMS}}, indent=0))
