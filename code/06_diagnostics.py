from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", message=".*valid feature names.*")

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parent / "data" / "cache" / "aiuse_sample.parquet"
RAW = HERE.parent / "data" / "processed" / "CY09_MS_STU_PUF.parquet"
OUT = HERE.parent / "results" / "diagnostics_stats.json"
TABLES = HERE.parent / "results" / "diagnostics_tables.json"

SEED, N_FOLDS, FAY_K = 42, 5, 0.5
N_PV, N_REP = 10, 80
DOMAINS = ["READ", "SCIE", "MATH", "CPRO", "CMPS", "CMOD", "CPPK"]
CONV, COMP = ["READ", "SCIE", "MATH"], ["CPRO", "CMPS", "CMOD", "CPPK"]
SCI = ["SEPS", "SEDE", "SEID"]
BASE = ["ESCS", "HISEI", "PAREDINT", "HOMEPOS", "AGE", "GRADE",
        "FEMALE", "IMMIG_2", "IMMIG_3", "REPEAT"]
ANCHORS = ["GRADE", "REPEAT", "ESCS", "PAREDINT", "HOMEPOS"]
DIRECT = {"CONV - COMP": (CONV, COMP), "CMPS - CMOD": (["CMPS"], ["CMOD"])}
_LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)


def crossfit(X, y, w, folds):
    pred = np.empty(len(y))
    for tr, te in folds:
        m = LGBMRegressor(**_LGB)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        pred[te] = m.predict(X[te])
    return pred


def senate(raw, codes):
    return raw / np.bincount(codes, raw)[codes] * 1000.0


def wstd(x, w):
    mu = np.average(x, weights=w)
    return (x - mu) / np.sqrt(np.average((x - mu) ** 2, weights=w))


def within(x, w, codes):
    return x - (np.bincount(codes, w * x) / np.bincount(codes, w))[codes]


def design(df, with_I=True):
    miss = [c for c in df.columns if c.endswith("_miss")]
    cdum = pd.get_dummies(df["CNT"], prefix="CNT", drop_first=True).astype(np.float64)
    cols = (["I"] if with_I else []) + BASE + miss
    X = pd.concat([df[cols].astype(np.float64), cdum], axis=1).to_numpy()
    sch = df["CNTSCHID"]
    cl = np.where(sch.notna(), "S" + sch.astype("string"),
                  "NOSCH_" + df["CNT"].astype(str)).astype(str)
    return X, cl


def load(smoke):
    df = pd.read_parquet(SAMPLE)
    raw = pd.read_parquet(RAW, columns=["CNTSTUID", "CNT", "ST022Q01TA"]
                          + [f"PV{m}{s}" for m in range(1, 11) for s in SCI])
    raw = raw.drop_duplicates(subset=["CNTSTUID", "CNT"])
    df = df.merge(raw, on=["CNTSTUID", "CNT"], how="left", validate="many_to_one")
    ok = np.ones(len(df), bool)
    for d in DOMAINS:
        ok &= np.isfinite(df[f"PV1{d}"].to_numpy(float))
    df = df[ok & ~np.isclose(df["I"], 1.0)].reset_index(drop=True)
    if smoke:
        _, cl = design(df)
        u = np.unique(cl)
        keep = np.random.RandomState(0).choice(u, size=len(u) // 25, replace=False)
        df = df[np.isin(cl, keep)].reset_index(drop=True)
    return df


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def raw_weight(df, r):
    return df["W_FSTUWT" if r == 0 else f"W_FSTURWT{r}"].to_numpy(float)


def stage1(df, res, n_pv, n_rep, out):
    st = res.setdefault("stage1", {})
    X, cl = design(df)
    C = df["C"].to_numpy(float)
    codes = pd.factorize(df["CNT"])[0]
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, C, cl))
    escs_c = df["ESCS"].to_numpy(float) - df.groupby("CNT")["ESCS"].transform("mean").to_numpy()
    anchors = {a: df[a].to_numpy(float) for a in ANCHORS}

    msk = np.ones(len(df), bool)
    for s in SCI:
        msk &= np.isfinite(df[f"PV1{s}"].to_numpy(float))
    pos = np.full(len(df), -1)
    pos[msk] = np.arange(msk.sum())
    folds_s = [(pos[tr][msk[tr]], pos[te][msk[te]]) for tr, te in folds]
    Xs, codes_s, es = X[msk], codes[msk], escs_c[msk]
    print(f"stage 1: n = {len(df):,} | science-subscale subset n = {msk.sum():,}", flush=True)

    w0 = senate(raw_weight(df, 0), codes)
    w0s = senate(raw_weight(df, 0)[msk], codes_s)
    t0 = time.time()
    for r in range(n_rep + 1):
        if str(r) in st:
            continue
        t1 = time.time()
        w = senate(raw_weight(df, r), codes)
        ws = senate(raw_weight(df, r)[msk], codes_s)
        Cr = C - crossfit(X, C, w, folds)
        Cs = Cr[msk]
        rec = {"A": [float((w * Cr ** 2).sum()), float((w * Cr ** 2 * escs_c).sum()),
                     float((w * Cr ** 2 * escs_c ** 2).sum())],
               "A_sci": [float((ws * Cs ** 2).sum()), float((ws * Cs ** 2 * es).sum()),
                         float((ws * Cs ** 2 * es ** 2).sum())],
               "pv": {}}
        xa = {a: within(v, w, codes) for a, v in anchors.items()}
        for m in (range(1, n_pv + 1) if r == 0 else [1]):
            o = {}
            for d in DOMAINS:
                z = wstd(df[f"PV{m}{d}"].to_numpy(float), w0)
                yr = z - crossfit(X, z, w, folds)
                o[d] = {"n": [float((w * Cr * yr).sum()), float((w * Cr * escs_c * yr).sum())],
                        "b": {a: float((w * x * z).sum() / (w * x * x).sum()) for a, x in xa.items()}}
            for s in SCI:
                z = wstd(df[f"PV{m}{s}"].to_numpy(float)[msk], w0s)
                yr = z - crossfit(Xs, z, ws, folds_s)
                o[s] = {"n": [float((ws * Cs * yr).sum()), float((ws * Cs * es * yr).sum())]}
            rec["pv"][str(m)] = o
        st[str(r)] = rec
        save(res, out)
        print(f"  stage 1 weight {r:>2}/{n_rep}  {(time.time()-t1)/60:5.1f} min "
              f"(elapsed {(time.time()-t0)/3600:.2f} h)", flush=True)


def direct_outcome(df, m, pos, neg, w0, rows=slice(None)):
    z = {d: wstd(df[f"PV{m}{d}"].to_numpy(float)[rows], w0) for d in set(pos + neg)}
    return np.mean([z[d] for d in pos], axis=0) - np.mean([z[d] for d in neg], axis=0)


def stage2_two(df, res, n_pv, n_rep, out):
    st = res.setdefault("two", {})
    X2, cl = design(df, with_I=False)
    off, sca = df["offload"].to_numpy(float), df["scaffold"].to_numpy(float)
    codes = pd.factorize(df["CNT"])[0]
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X2, off, cl))
    w0 = senate(raw_weight(df, 0), codes)
    for r in range(n_rep + 1):
        if str(r) in st:
            continue
        t1 = time.time()
        w = senate(raw_weight(df, r), codes)
        o_r = off - crossfit(X2, off, w, folds)
        s_r = sca - crossfit(X2, sca, w, folds)
        rec = {"S": [float((w * o_r ** 2).sum()), float((w * o_r * s_r).sum()),
                     float((w * s_r ** 2).sum())], "pv": {}}
        for m in (range(1, n_pv + 1) if r == 0 else [1]):
            o = {}
            for lab, (pos, neg) in DIRECT.items():
                y = direct_outcome(df, m, pos, neg, w0)
                yr = y - crossfit(X2, y, w, folds)
                o[lab] = [float((w * o_r * yr).sum()), float((w * s_r * yr).sum())]
            rec["pv"][str(m)] = o
        st[str(r)] = rec
        save(res, out)
        print(f"  two-treatment weight {r:>2}/{n_rep}  {(time.time()-t1)/60:5.1f} min", flush=True)


def stage2_lang(df, res, n_pv, n_rep, out):
    lang = df["ST022Q01TA"].where(df["ST022Q01TA"] <= 2).to_numpy(float)
    for code in (1, 2):
        st = res.setdefault(f"lang{code}", {})
        rows = lang == code
        d2 = df[rows].reset_index(drop=True)
        X, cl = design(d2)
        X = X[:, X.std(axis=0) > 0]
        C = d2["C"].to_numpy(float)
        codes = pd.factorize(d2["CNT"])[0]
        folds = list(GroupKFold(n_splits=N_FOLDS).split(X, C, cl))
        w0 = senate(raw_weight(d2, 0), codes)
        st.setdefault("n", int(rows.sum()))
        for r in range(n_rep + 1):
            if str(r) in st:
                continue
            t1 = time.time()
            w = senate(raw_weight(d2, r), codes)
            Cr = C - crossfit(X, C, w, folds)
            rec = {"A": float((w * Cr ** 2).sum()), "pv": {}}
            for m in (range(1, n_pv + 1) if r == 0 else [1]):
                o = {}
                for lab, (pos, neg) in DIRECT.items():
                    y = direct_outcome(d2, m, pos, neg, w0)
                    o[lab] = float((w * Cr * (y - crossfit(X, y, w, folds))).sum())
                rec["pv"][str(m)] = o
            st[str(r)] = rec
            save(res, out)
            print(f"  language group {code} weight {r:>2}/{n_rep}  {(time.time()-t1)/60:5.1f} min",
                  flush=True)


def mibrr(f, n_pv, n_rep):
    pv = np.array([f(0, m) for m in range(1, n_pv + 1)])
    est = float(pv.mean())
    v_imp = (1 + 1 / n_pv) * float(pv.var(ddof=1))
    base = f(0, 1)
    v_s = float(sum((f(r, 1) - base) ** 2 for r in range(1, n_rep + 1)) / (n_rep * (1 - FAY_K) ** 2))
    se = float(np.sqrt(v_s + v_imp))
    return {"est": est, "se": se, "t": est / se if se > 0 else float("nan"),
            "pct_var_imputation": 100 * v_imp / (v_s + v_imp) if v_s + v_imp > 0 else float("nan")}


def tables(res, n_pv, n_rep):
    T = {}
    s1 = res.get("stage1", {})
    if len(s1) == n_rep + 1:
        def th(r, m, d):
            rec = s1[str(r)]
            return rec["pv"][str(m)][d]["n"][0] / rec["A_sci" if d in SCI else "A"][0]

        def lin(pos, neg, k=None):
            def f(r, m):
                kk = 1.0 if k is None else k(r, m)
                return np.mean([th(r, m, d) for d in pos]) - kk * np.mean([th(r, m, d) for d in neg])
            return f

        def b(r, m, a, ds):
            return np.mean([s1[str(r)]["pv"][str(m)][d]["b"][a] for d in ds])

        def kappa(a, pos, neg):
            return lambda r, m: b(r, m, a, pos) / b(r, m, a, neg)

        def run(f):
            return mibrr(f, n_pv, n_rep)

        T["contrasts"] = {
            "CONV - COMP (four-scale mean)": run(lin(CONV, COMP)),
            "CONV - CMPS (overall scale)": run(lin(CONV, ["CMPS"])),
            "CONV - mean(CMOD, CPRO)": run(lin(CONV, ["CMOD", "CPRO"])),
            "CONV - CPPK": run(lin(CONV, ["CPPK"])),
        }
        T["negative_controls"] = {
            "CMPS - CMOD": run(lin(["CMPS"], ["CMOD"])),
            "CPPK - CMOD": run(lin(["CPPK"], ["CMOD"])),
            "CMOD - CPRO": run(lin(["CMOD"], ["CPRO"])),
            "SEPS - SEDE": run(lin(["SEPS"], ["SEDE"])),
            "SEPS - SEID": run(lin(["SEPS"], ["SEID"])),
            "SEDE - SEID": run(lin(["SEDE"], ["SEID"])),
        }
        T["levels"] = {d: run(lambda r, m, d=d: th(r, m, d)) for d in DOMAINS + SCI}
        T["kappa_zeroing_primary"] = run(lambda r, m: np.mean([th(r, m, d) for d in CONV])
                                         / np.mean([th(r, m, d) for d in COMP]))
        anc = {}
        for a in ANCHORS:
            k = kappa(a, CONV, COMP)
            k_fixed = float(np.mean([k(0, m) for m in range(1, n_pv + 1)]))
            k2 = kappa(a, ["CMOD"], ["CPRO"])
            anc[a] = {
                "slope_CONV": run(lambda r, m, a=a: b(r, m, a, CONV)),
                "slope_COMP": run(lambda r, m, a=a: b(r, m, a, COMP)),
                "slope_on_D": run(lambda r, m, a=a: b(r, m, a, CONV) - b(r, m, a, COMP)),
                "kappa": run(k),
                "recalibrated_primary": run(lin(CONV, COMP, k)),
                "recalibrated_primary_kappa_fixed": run(lin(CONV, COMP, lambda r, m: k_fixed)),
                "kappa_CMOD_CPRO": run(k2),
                "recalibrated_CMOD_CPRO": run(lin(["CMOD"], ["CPRO"], k2)),
                "slopes_by_scale": {d: run(lambda r, m, a=a, d=d: b(r, m, a, [d])) for d in DOMAINS},
            }
        T["anchors"] = anc

        def blp(pos, neg, j, k=None):
            def f(r, m):
                rec = s1[str(r)]
                A = rec["A_sci" if pos[0] in SCI else "A"]
                kk = 1.0 if k is None else k(r, m)
                n = (np.mean([rec["pv"][str(m)][d]["n"] for d in pos], axis=0)
                     - kk * np.mean([rec["pv"][str(m)][d]["n"] for d in neg], axis=0))
                return np.linalg.solve([[A[0], A[1]], [A[1], A[2]]], n)[j]
            return f
        T["escs_moderation"] = {
            lab: {"theta0": run(blp(p, q, 0)), "escs_slope": run(blp(p, q, 1))}
            for lab, (p, q) in DIRECT.items()}
        NC = {"CMPS - CMOD": (["CMPS"], ["CMOD"]), "CPPK - CMOD": (["CPPK"], ["CMOD"]),
              "CMOD - CPRO": (["CMOD"], ["CPRO"]), "SEPS - SEDE": (["SEPS"], ["SEDE"]),
              "SEPS - SEID": (["SEPS"], ["SEID"]), "SEDE - SEID": (["SEDE"], ["SEID"])}
        zero = lambda r, m: 0.0
        T["escs_slope_controls"] = {lab: run(blp(p, q, 1)) for lab, (p, q) in NC.items()}
        T["escs_slope_composites"] = {"CONV": run(blp(CONV, COMP, 1, zero)),
                                      "COMP": run(blp(COMP, CONV, 1, zero))}
        T["escs_slope_recalibrated"] = {a: run(blp(CONV, COMP, 1, kappa(a, CONV, COMP)))
                                        for a in ANCHORS}

        def gap(kp=None, kc=None):
            return lambda r, m: lin(CONV, COMP, kp)(r, m) - lin(["CMOD"], ["CPRO"], kc)(r, m)
        T["primary_minus_largest_control"] = {
            "uncalibrated": run(gap()),
            **{a: run(gap(kappa(a, CONV, COMP), kappa(a, ["CMOD"], ["CPRO"]))) for a in ANCHORS}}

    two = res.get("two", {})
    if len(two) == n_rep + 1:
        def coef(lab, j):
            def f(r, m):
                S = two[str(r)]["S"]
                return np.linalg.solve([[S[0], S[1]], [S[1], S[2]]], two[str(r)]["pv"][str(m)][lab])[j]
            return f
        T["two_treatment"] = {
            lab: {"offloading": mibrr(coef(lab, 0), n_pv, n_rep),
                  "scaffolding": mibrr(coef(lab, 1), n_pv, n_rep),
                  "half_difference": mibrr(lambda r, m, lab=lab: (coef(lab, 0)(r, m) - coef(lab, 1)(r, m)) / 2,
                                           n_pv, n_rep)}
            for lab in DIRECT}

    l1, l2 = res.get("lang1", {}), res.get("lang2", {})
    if len(l1) == n_rep + 2 and len(l2) == n_rep + 2:
        def g(st, lab):
            return lambda r, m: st[str(r)]["pv"][str(m)][lab] / st[str(r)]["A"]
        T["language_at_home"] = {
            lab: {"test_language": {"n": l1["n"], **mibrr(g(l1, lab), n_pv, n_rep)},
                  "other_language": {"n": l2["n"], **mibrr(g(l2, lab), n_pv, n_rep)},
                  "difference": mibrr(lambda r, m, lab=lab: g(l1, lab)(r, m) - g(l2, lab)(r, m),
                                      n_pv, n_rep)}
            for lab in DIRECT}
    return T


def show(T):
    def line(lab, v, extra=""):
        print(f"   {lab:38s} {v['est']:+.4f}  (SE {v['se']:.4f}, t {v['t']:+6.1f}, "
              f"imputation {v['pct_var_imputation']:4.1f}%) {extra}")
    for sec in ("contrasts", "negative_controls"):
        if sec in T:
            print(f"\n{sec.upper()}")
            for k, v in T[sec].items():
                line(k, v)
    if "anchors" in T:
        print("\nANCHORS (kappa re-estimated under every weight and plausible value)")
        line("kappa that zeroes the primary", T["kappa_zeroing_primary"])
        for a, v in T["anchors"].items():
            print(f"  {a}")
            for k in ("slope_CONV", "slope_COMP", "slope_on_D", "kappa", "recalibrated_primary",
                      "recalibrated_primary_kappa_fixed", "kappa_CMOD_CPRO", "recalibrated_CMOD_CPRO"):
                line(k, v[k])
    for sec in ("escs_slope_controls", "escs_slope_composites", "escs_slope_recalibrated",
                "primary_minus_largest_control"):
        if sec in T:
            print(f"\n{sec.upper()}")
            for k, v in T[sec].items():
                line(k, v)
    for sec in ("escs_moderation", "two_treatment", "language_at_home"):
        if sec in T:
            print(f"\n{sec.upper()}")
            for lab, d in T[sec].items():
                for k, v in d.items():
                    line(f"{lab} | {k}", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    n_pv, n_rep = (2, 2) if a.smoke else (N_PV, N_REP)
    out = a.out or (OUT.with_name("diagnostics_smoke_stats.json") if a.smoke else OUT)
    if a.smoke:
        _LGB["n_estimators"] = 30
    res = json.loads(out.read_text()) if out.exists() else {}

    if not a.tables:
        df = load(a.smoke)
        print(f"estimation sample: {len(df):,} students, {df['CNT'].nunique()} systems", flush=True)
        if a.stage in (None, 1):
            stage1(df, res, n_pv, n_rep, out)
        if a.stage in (None, 2):
            stage2_two(df, res, n_pv, n_rep, out)
            stage2_lang(df, res, n_pv, n_rep, out)

    T = tables(res, n_pv, n_rep)
    show(T)
    tpath = out.with_name(out.name.replace("_stats", "_tables"))
    save(T, tpath)
    print(f"\n-> {out}\n-> {tpath}")


if __name__ == "__main__":
    main()
