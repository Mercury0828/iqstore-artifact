# Extra tables, figures and numbers for the paper. Executed at the end of gen_results.py, in its
# namespace (q1, st, qt, mt, cov, LAY, boot, EPOCH, T, G, R, plt, np, json are defined there).
xm = []

# ---------------- datasets table (all 35 datasets; TEST counts from q1, totals from the collection schedule)
WL = [("GHZ", "GHZ state on 8, 12 or 16 qubits", "16", 6),
      ("RCL", "random Clifford layers, depth 2--20", "16", 20),
      ("PROD", "product state", "16", 4),
      ("REP", "distance-5 repetition code, 5--10 rounds", "25--45", 5)]
L = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"workload & circuits & $m$ & dev & eval\\", r"\midrule"]
for key, desc, mm, tot in WL:
    te = sum(1 for n in q1 if n.startswith(key))
    L.append(rf"{ {'GHZ': 'GHZ', 'RCL': 'Clifford', 'PROD': 'product', 'REP': 'repetition'}[key]} & {desc} & {mm} & {tot - te} & {te}\\")
L += [r"\midrule", rf"total & & & {35 - len(q1)} & {len(q1)}\\", r"\bottomrule", r"\end{tabular}"]
(G / "tab_datasets.tex").write_text("\n".join(L) + "\n")

# ---------------- storage against the bits alone, size breakdown, band share
k = "F-drift|0.001"
ep = [d["epoch"] for d in q1.values()]
r_b1 = boot([d[k]["IQSTORE"] / d["B1_zstd"] for d in q1.values()], ep)
xm.append(rf"\newcommand{{\RIQBone}}{{{r_b1[0]:.2f}}}")
xm.append(rf"\newcommand{{\OpMsDrift}}{{{qt['F-drift']['op_ms']:.1f}}}")
xm.append(rf"\newcommand{{\BtwoBitsDrift}}{{{min(st[k]['B2-zfp'], st[k]['B2-SZ3']):.2f}}}")
xm.append(rf"\newcommand{{\DeltaBone}}{{{np.median([d[k]['IQSTORE'] - d['B1_zstd'] for d in q1.values()]):.2f}}}")
tot = {p: sum(d[k]["stats"]["iq_bytes"][p] for d in q1.values()) for p in ("header", "tree", "columns", "exceptions")}
allb = sum(tot.values())
for p, mac in (("header", "Hdr"), ("tree", "Tree"), ("columns", "Cols"), ("exceptions", "Exc")):
    xm.append(rf"\newcommand{{\Share{mac}}}{{{100 * tot[p] / allb:.1f}}}")
band = [100 * d[k]["stats"]["band_shots"] / (d["N"] * d["m"]) for d in q1.values()]
xm.append(rf"\newcommand{{\BandPct}}{{{np.median(band):.2f}}}")
xm.append(rf"\newcommand{{\BandPctMax}}{{{max(band):.2f}}}")
kb = "F-boot|0.001"
bandb = [100 * d[kb]["stats"]["band_shots"] / (d["N"] * d["m"]) for d in q1.values()]
xm.append(rf"\newcommand{{\BandPctBoot}}{{{np.median(bandb):.2f}}}")
R["extra_storage"] = {"IQ/B1": r_b1, "bytes": tot, "band_pct_median": float(np.median(band))}

# ---------------- ingest time (H-c replay ingest events on the evaluation datasets)
ing = {}
for fam in FAMS:
    rows = []
    for s_ in (T / "hc" / f"events_{fam}.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(s_)
        except json.JSONDecodeError:
            break
        if e.get("kind") == "ingest" and e.get("split") == "TEST":
            rows.append(e)
    ing[fam] = {"n": len(rows), "median_s": float(np.median([e["build_s"] for e in rows])),
                "per_pos_s": float(np.median([e["build_s"] / e["m"] for e in rows])),
                "max_s": float(max(e["build_s"] for e in rows))}
R["ingest"] = ing
xm += [rf"\newcommand{{\IngestDrift}}{{{ing['F-drift']['median_s']:.0f}}}",
       rf"\newcommand{{\IngestPosDrift}}{{{ing['F-drift']['per_pos_s']:.1f}}}",
       rf"\newcommand{{\IngestBoot}}{{{ing['F-boot']['median_s']:.0f}}}",
       rf"\newcommand{{\IngestPosBoot}}{{{ing['F-boot']['per_pos_s']:.1f}}}"]

# ---------------- coverage: smallest per-dataset share inside the drift sets
if cov:
    sh = [v["F-drift"]["inside"] / v["F-drift"]["total"] for v in cov["datasets"].values() if v["F-drift"]["total"]]
    xm.append(rf"\newcommand{{\CovMinDrift}}{{{100 * min(sh):.1f}}}")
    nd = sum(1 for v in cov["datasets"].values() if v["F-drift"]["total"])
    xm.append(rf"\newcommand{{\CovDatasets}}{{{nd}}}")

# ---------------- per-dataset query ratios (figure)
import matplotlib.lines as mlines
plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), sharey=True)
xs_lab = ["single", "pair", "3-4", "mitig.", "sweep"]
for ax, base in zip(axs, FAMS_PAPER):
    D = [json.loads(f.read_text()) for f in sorted((T / "q2" / "time").glob(f"*_{base}.json"))]
    for i, c in enumerate(CLS):
        for d in D:
            r_ = d["class_median_s_all_pairs"]["B0q"][c] / d["class_median_s_all_pairs"]["IQSTORE"][c]
            blk = LAY[d["dataset"]] == "block"
            jit = (sum(map(ord, d["dataset"])) % 7 - 3) * 0.05
            ax.plot(i + jit, r_, "o" if blk else "^", ms=2.6, mfc="black" if blk else "none", mec="black", mew=0.6)
        med = qt[base][c]["B0q/IQ"][0]
        ax.plot([i - 0.32, i + 0.32], [med, med], "-", color="0.45", lw=1.4)
    ax.set_yscale("log"); ax.set_xticks(range(5)); ax.set_xticklabels(xs_lab, rotation=35, ha="right", fontsize=6.5)
    ax.set_title(f"{FAMNAME[base]} sets", fontsize=7.5); ax.set_xlabel("query class")
    ax.grid(axis="y", which="major", lw=0.3, color="0.85")
axs[0].set_ylabel("B0q time / IQSTORE time")
h1 = mlines.Line2D([], [], ls="none", marker="o", mfc="black", mec="black", ms=3, label="B0 one block")
h2 = mlines.Line2D([], [], ls="none", marker="^", mfc="none", mec="black", ms=3, label="B0 per position")
h3 = mlines.Line2D([], [], color="0.45", lw=1.4, label="median")
fig.legend(handles=[h1, h2, h3], loc="upper center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.55, 1.01))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
fig.savefig(G / "fig_query.pdf")
plt.close(fig)

# ---------------- maintenance time against affected shots (figure), drift sets, real changes
rows = []
for s_ in (T / "hc" / "events_F-drift.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        e = json.loads(s_)
    except json.JSONDecodeError:
        break
    if e.get("kind") == "real" and e.get("split") == "TEST":
        rows.append(e)
fig, ax = plt.subplots(figsize=(3.45, 1.95))
for inf, mk, lab in ((True, "o", "IQSTORE, change inside the sets"), (False, "s", "IQSTORE, change leaves the sets")):
    sel = [e for e in rows if bool(e["in_family"]) == inf]
    ax.plot([max(int(e["shots"]), 1) for e in sel], [e["iq"]["median"] * 1e3 for e in sel], mk, ms=2.2,
            mfc="black" if inf else "none", mec="black", mew=0.5, ls="none", label=lab)
ax.plot([max(int(e["shots"]), 1) for e in rows], [e["b6"]["median"] * 1e3 for e in rows], "x", ms=2.2,
        color="0.55", mew=0.5, ls="none", label="B6, recompute every view")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("shots whose label set changes (1 = none)"); ax.set_ylabel("time per change (ms)")
ax.legend(fontsize=6.2, frameon=False, loc="lower right")
ax.grid(which="major", lw=0.3, color="0.85")
fig.tight_layout(pad=0.3)
fig.savefig(G / "fig_maint.pdf")
plt.close(fig)
aff_in = [int(e["shots"]) for e in rows if e["in_family"]]
aff_out = [int(e["shots"]) for e in rows if not e["in_family"]]
xm += [rf"\newcommand{{\AffInMed}}{{{np.median(aff_in):,.0f}}}".replace(",", "{,}"),
       rf"\newcommand{{\AffOutMed}}{{{np.median(aff_out):,.0f}}}".replace(",", "{,}")]
R["maint_affected"] = {"in_median": float(np.median(aff_in)), "out_median": float(np.median(aff_out)),
                       "n_real_test": len(rows)}

with open(G / "numbers.tex", "a") as fh:
    fh.write("\n".join(xm) + "\n")
(G / "results.json").write_text(json.dumps(R, indent=1))
