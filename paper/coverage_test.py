"""H-d on TEST: share of later re-fits (every accepted later epoch, every position) that lie inside the family fixed at
ingest, per base family; also the dataset -> epoch map. Untimed. Output paper/coverage.json."""
import os, sys, json, pathlib
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
ROOT = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from eval.m2_test import test_datasets, fdrift_families, freeze
from eval.m2_cov import later_refs_all
fz, _ = freeze()
out = {"datasets": {}}
for entry, ep, ds in test_datasets():
    refs = later_refs_all(ds)
    fams = {"F-boot": [p.fams["F-boot"] for p in ds.positions], "F-drift": fdrift_families(ds, fz)}
    if "F-plan" in ds.positions[0].fams:
        fams["F-plan"] = [p.fams["F-plan"] for p in ds.positions]
    r = {"epoch": ds.epoch, "m": len(ds.positions), "later_epochs": sorted(refs)}
    for name, F in fams.items():
        tot = ins = 0
        ctx = {}
        for e, hs in refs.items():
            for p, f, h in zip(ds.positions, F, hs):
                c = f.contains(h)
                tot += 1; ins += int(c)
                k = str(p.context); a = ctx.setdefault(k, [0, 0]); a[0] += int(c); a[1] += 1
        r[name] = {"inside": ins, "total": tot, "by_context": ctx}
    out["datasets"][ds.name] = r
    print(ds.name, ds.epoch, {k: (v["inside"], v["total"]) for k, v in r.items() if isinstance(v, dict)}, flush=True)
for name in ("F-boot", "F-drift", "F-plan"):
    rows = [d[name] for d in out["datasets"].values() if name in d]
    if rows:
        out[name] = {"inside": sum(x["inside"] for x in rows), "total": sum(x["total"] for x in rows)}
(ROOT / "paper" / "coverage.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "datasets"})
