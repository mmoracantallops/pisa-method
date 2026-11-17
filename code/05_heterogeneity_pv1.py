from __future__ import annotations

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
OUT = HERE.parent / "results" / "heterogeneity_pv1.json"

SEED, N_FOLDS = 42, 5
DOMAINS = ["READ", "SCIE", "MATH", "CPRO", "CMPS", "CMOD", "CPPK"]
CONV, COMP = ["READ", "SCIE", "MATH"], ["CPRO", "CMPS", "CMOD", "CPPK"]
CONTROLS = ["I", "ESCS", "HISEI", "PAREDINT", "HOMEPOS", "AGE", "GRADE",
            "FEMALE", "IMMIG_2", "IMMIG_3", "REPEAT"]
CONTRASTS = [("CONV - COMP", CONV, COMP), ("READ - CPRO", ["READ"], ["CPRO"]),
             ("CMPS - CMOD", ["CMPS"], ["CMOD"])]
_LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)


def crossfit(X, y, w, g):
    p = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        m = LGBMRegressor(**_LGB); m.fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = m.predict(X[te])
    return p


def wls_cluster(Z, y, w, cl):
    ZW = Z * w[:, None]
    XtX_inv = np.linalg.pinv(Z.T @ ZW)
    beta = XtX_inv @ (ZW.T @ y)
    u = y - Z @ beta
    s = pd.DataFrame(ZW * u[:, None])
    s["c"] = cl
    G = s.groupby("c").sum().to_numpy()
    meat = G.T @ G
    V = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def wstd(x, w):
    mu = np.average(x, weights=w)
    return (x - mu) / np.sqrt(np.average((x - mu) ** 2, weights=w))


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    ok = np.ones(len(df), bool)
    for d in DOMAINS:
        ok &= np.isfinite(df[f"PV1{d}"].to_numpy(float))
    df = df[ok & ~np.isclose(df["I"], 1.0)].reset_index(drop=True)
    print(f"AI users with all seven domains: {len(df):,}, {df['CNT'].nunique()} countries")

    miss = [c for c in df.columns if c.endswith("_miss")]
    cnt = pd.get_dummies(df["CNT"], prefix="CNT", drop_first=True).astype(np.float64)
    X = pd.concat([df[CONTROLS + miss].astype(np.float64), cnt], axis=1).to_numpy()
    C = df["C"].to_numpy(float)
    _raw = df["W_FSTUWT"].to_numpy(float)
    w = _raw / pd.Series(_raw).groupby(df["CNT"].to_numpy()).transform("sum").to_numpy() * 1000.0
    sch = df["CNTSCHID"]
    cl = np.where(sch.notna(), "S" + sch.astype("string"),
                  "NOSCH_" + df["CNT"].astype(str)).astype(str)

    escs = df["ESCS"].to_numpy(float)
    escs_c = escs - df.groupby("CNT")["ESCS"].transform("mean").to_numpy()
    qt = pd.qcut(escs_c, 5, labels=False, duplicates="drop")

    print("\ncross-fitting treatment model ...")
    t0 = time.perf_counter()
    C_res = C - crossfit(X, C, w, cl)
    print(f"  {time.perf_counter()-t0:.0f}s")

    z = {d: wstd(df[f"PV1{d}"].to_numpy(float), w) for d in DOMAINS}
    out = {}
    for lab, pos, neg in CONTRASTS:
        y = np.mean([z[d] for d in pos], axis=0) - np.mean([z[d] for d in neg], axis=0)
        y_res = y - crossfit(X, y, w, cl)

        Z = np.column_stack([C_res, C_res * escs_c])
        b, se = wls_cluster(Z, y_res, w, cl)
        blp = {"theta0": round(float(b[0]), 5), "theta0_se": round(float(se[0]), 5),
               "theta_escs": round(float(b[1]), 5), "theta_escs_se": round(float(se[1]), 5),
               "t_escs": round(float(b[1] / se[1]), 2)}

        quints = {}
        for q in range(5):
            m = qt == q
            bq, seq = wls_cluster(C_res[m][:, None], y_res[m], w[m], cl[m])
            quints[f"Q{q+1}"] = {"n": int(m.sum()), "theta": round(float(bq[0]), 5),
                                 "se": round(float(seq[0]), 5)}
        out[lab] = {"blp": blp, "quintiles": quints}

        print(f"\n{lab}")
        print(f"  BLP: theta0 = {blp['theta0']:+.4f} ({blp['theta0_se']:.4f}) | "
              f"x ESCS = {blp['theta_escs']:+.5f} ({blp['theta_escs_se']:.5f}, "
              f"t {blp['t_escs']:+.1f})")
        print("  by within-country ESCS quintile (Q1 = most disadvantaged):")
        for k, v in quints.items():
            print(f"    {k} n={v['n']:>7,}  {v['theta']:+.4f} ({v['se']:.4f})")

    OUT.write_text(json.dumps({"n": int(len(df)), "results": out}, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
