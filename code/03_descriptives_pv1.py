from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", message=".*valid feature names.*")

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parent / "data" / "cache" / "aiuse_sample.parquet"
OUT = HERE.parent / "results" / "descriptives_pv1.json"

SEED, N_FOLDS = 42, 5
DOMAINS = ["READ", "SCIE", "MATH", "CMOD", "CPPK", "CMPS", "CPRO"]
CONV, COMP = ["READ", "SCIE", "MATH"], ["CMOD", "CPPK", "CMPS", "CPRO"]
CONTROLS = ["I", "ESCS", "HISEI", "PAREDINT", "HOMEPOS", "AGE", "GRADE",
            "FEMALE", "IMMIG_2", "IMMIG_3", "REPEAT"]
ANCHORS = ["GRADE", "REPEAT", "ESCS", "HOMEPOS", "PAREDINT"]
_LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)


def crossfit(X, y, w, folds):
    p = np.empty(len(y))
    for tr, te in folds:
        m = LGBMRegressor(**_LGB); m.fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = m.predict(X[te])
    return p


def cluster_se(t, eps, w, cl):
    s = pd.DataFrame({"c": cl, "s": w * t * eps})
    return float(np.sqrt((s.groupby("c")["s"].sum() ** 2).sum()) / (w * t ** 2).sum())


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    ok = np.ones(len(df), bool)
    for d in DOMAINS:
        ok &= np.isfinite(df[f"PV1{d}"].to_numpy(float))
    full7 = df[ok]
    df = full7[~np.isclose(full7["I"], 1.0)].reset_index(drop=True)
    cnt = df["CNT"].to_numpy()
    raw = df["W_FSTUWT"].to_numpy(float)
    w = raw / pd.Series(raw).groupby(cnt).transform("sum").to_numpy() * 1000.0
    out = {"n": int(len(df))}

    def wstd(x):
        mu = np.average(x, weights=w)
        return (x - mu) / np.sqrt(np.average((x - mu) ** 2, weights=w))

    def within(x):
        s = pd.DataFrame({"c": cnt, "xw": x * w, "w": w}).groupby("c")[["xw", "w"]].transform("sum")
        return x - (s["xw"] / s["w"]).to_numpy()

    off, sca = df["offload"].to_numpy(float), df["scaffold"].to_numpy(float)
    C, I = df["C"].to_numpy(float), df["I"].to_numpy(float)
    never = np.isclose(full7["I"], 1.0)
    bal7 = (full7["C_cat"] == "balanced")
    Cw = within(C)
    desc = {
        "corr_off_sca": float(np.corrcoef(off, sca)[0, 1]),
        "corr_C_I": float(np.corrcoef(C, I)[0, 1]),
        "var_off": float(off.var()), "var_sca": float(sca.var()),
        "C_mean": float(C.mean()), "C_sd": float(C.std()),
        "within_system_share_C": float((w * Cw ** 2).sum() / (w * (C - np.average(C, weights=w)) ** 2).sum()),
        "never_users_n": int(never.sum()), "never_users_pct": float(never.mean() * 100),
        "never_share_of_balanced_pct": float((never & bal7).sum() / bal7.sum() * 100),
        "pct_C_zero_users": float(np.isclose(C, 0).mean() * 100),
    }
    out["descriptives"] = desc
    print("A. DESCRIPTIVES ON THE ESTIMATION SAMPLE (n = %d)" % len(df))
    for k, v in desc.items():
        print(f"   {k:30s} {v:.4f}" if isinstance(v, float) else f"   {k:30s} {v}")

    Z = {d: wstd(df[f"PV1{d}"].to_numpy(float)) for d in DOMAINS}
    Zw = {d: within(Z[d]) for d in DOMAINS}
    prof, kappa = {}, {}
    print("\nB. COEFFICIENT PROFILES (within-system slope of each standardised domain on the anchor)")
    print(f"   {'anchor':10s}" + "".join(f"{d:>8s}" for d in DOMAINS) + "   CONV    COMP   kappa")
    for a in ANCHORS:
        x = within(df[a].to_numpy(float))
        b = {d: float((w * x * Zw[d]).sum() / (w * x * x).sum()) for d in DOMAINS}
        bc, bp = np.mean([b[d] for d in CONV]), np.mean([b[d] for d in COMP])
        prof[a], kappa[a] = b, float(bc / bp)
        print(f"   {a:10s}" + "".join(f"{b[d]:8.3f}" for d in DOMAINS)
              + f"  {bc:6.3f}  {bp:6.3f}  {bc/bp:6.3f}")
    out["anchor_profiles"], out["kappa"] = prof, kappa

    miss = [c for c in df.columns if c.endswith("_miss")]
    cdum = pd.get_dummies(df["CNT"], prefix="CNT", drop_first=True).astype(np.float64)
    X = pd.concat([df[CONTROLS + miss].astype(np.float64), cdum], axis=1).to_numpy()
    sch = df["CNTSCHID"]
    cl = np.where(sch.notna(), "S" + sch.astype("string"),
                  "NOSCH_" + df["CNT"].astype(str)).astype(str)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, C, cl))

    C_res = C - crossfit(X, C, w, folds)
    yc = np.mean([Z[d] for d in CONV], axis=0)
    yp = np.mean([Z[d] for d in COMP], axis=0)
    yc_res = yc - crossfit(X, yc, w, folds)
    yp_res = yp - crossfit(X, yp, w, folds)
    den = (w * C_res ** 2).sum()

    def r2(y, res):
        return float(1 - np.average(res ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2, weights=w))

    out["nuisance_r2"] = {"treatment_C": r2(C, C_res), "CONV": r2(yc, yc_res), "COMP": r2(yp, yp_res)}
    d_res = yc_res - yp_res
    out["nuisance_r2"]["D_CONV_minus_COMP"] = r2(yc - yp, d_res)
    print("\nD. CROSS-FITTED NUISANCE FIT IN THIS APPLICATION (R^2)")
    for k, v in out["nuisance_r2"].items():
        print(f"   {k:22s} {v:.3f}")

    b_conv = float((w * C_res * yc_res).sum() / den)
    b_comp = float((w * C_res * yp_res).sum() / den)
    print(f"\nC. CALIBRATED CONTRAST  D*(kappa) = CONV - kappa * COMP   "
          f"(level coefficients: CONV {b_conv:+.4f}, COMP {b_comp:+.4f})")
    print(f"   {'calibration':26s}{'kappa':>7s}{'tau*':>10s}{'SE':>9s}{'t':>7s}{'% of raw':>10s}")
    cal = {}
    grid = [("none (as in manuscript)", 1.0)] + [(f"anchor: {a}", kappa[a]) for a in ANCHORS]
    raw_tau = None
    for lab, k in grid:
        res = yc_res - k * yp_res
        th = float((w * C_res * res).sum() / den)
        se = cluster_se(C_res, res - th * C_res, w, cl)
        raw_tau = th if raw_tau is None else raw_tau
        cal[lab] = {"kappa": k, "tau": th, "se": se, "pct_of_raw": th / raw_tau * 100}
        print(f"   {lab:26s}{k:7.3f}{th:+10.4f}{se:9.4f}{th/se:7.1f}{th/raw_tau*100:9.0f}%")
    out["calibrated"] = cal
    out["kappa_that_zeroes_contrast"] = b_conv / b_comp
    print(f"\n   kappa at which the contrast is exactly zero: {b_conv/b_comp:.3f}")

    print("\n   Association of whole-profile anchors with D = CONV - COMP (should be 0 under equal loading):")
    Dw = within(yc - yp)
    nce = {}
    for a in ANCHORS:
        x = within(df[a].to_numpy(float))
        b = float((w * x * Dw).sum() / (w * x * x).sum())
        e = Dw - b * x
        se = cluster_se(x, e, w, cl)
        nce[a] = {"slope": b, "se": se}
        print(f"     {a:10s} {b:+.4f}  (SE {se:.4f}, t {b/se:+.1f})")
    out["anchor_on_D"] = nce

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
