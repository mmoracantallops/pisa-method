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
OUT = HERE.parent / "results" / "robustness_pv1.json"

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


def crossfit(X, y, w, folds):
    p = np.empty(len(y))
    for tr, te in folds:
        m = LGBMRegressor(**_LGB); m.fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = m.predict(X[te])
    return p


def wls_cluster(Z, y, w, cl):
    ZW = Z * w[:, None]
    inv = np.linalg.pinv(Z.T @ ZW)
    beta = inv @ (ZW.T @ y)
    u = y - Z @ beta
    s = pd.DataFrame(ZW * u[:, None]); s["c"] = cl
    G = s.groupby("c").sum().to_numpy()
    V = inv @ (G.T @ G) @ inv
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
    print(f"AI users: {len(df):,} | {df['CNT'].nunique()} countries")

    miss = [c for c in df.columns if c.endswith("_miss")]
    cnt = pd.get_dummies(df["CNT"], prefix="CNT", drop_first=True).astype(np.float64)
    X = pd.concat([df[CONTROLS + miss].astype(np.float64), cnt], axis=1).to_numpy()
    C = df["C"].to_numpy(float)
    _cnt = df["CNT"].to_numpy()
    _raw = df["W_FSTUWT"].to_numpy(float)
    w = _raw / pd.Series(_raw).groupby(_cnt).transform("sum").to_numpy() * 1000.0
    sch = df["CNTSCHID"]
    cl = np.where(sch.notna(), "S" + sch.astype("string"),
                  "NOSCH_" + df["CNT"].astype(str)).astype(str)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, C, cl))
    out = {"n": int(len(df))}

    print("\n[1] continuous positivity on the final specification")
    C_hat = crossfit(X, C, w, folds)
    resid = C - C_hat
    marg = float(C.std())
    rows = []
    for nb in (10, 20, 50):
        b = pd.qcut(C_hat, nb, labels=False, duplicates="drop")
        sd = pd.Series(resid).groupby(b).std()
        rows.append({"bins": nb, "min_sd": round(float(sd.min()), 4),
                     "p05_sd": round(float(sd.quantile(0.05)), 4),
                     "median_sd": round(float(sd.median()), 4),
                     "ratio_min": round(float(sd.min() / marg), 4),
                     "frac_below_25pct": round(float((sd < 0.25 * marg).mean()), 4)})
        print(f"   {nb:>3} bins: min SD {sd.min():.3f} | median {sd.median():.3f} | "
              f"ratio {sd.min()/marg:.3f} | bins<25% of marginal: "
              f"{(sd < 0.25*marg).mean()*100:.0f}%")
    out["positivity"] = {"marginal_sd": round(marg, 4), "by_binning": rows}

    print("\n[2] categorical specification (reference = balanced)")
    off = (df["C_cat"] == "offload_tilted").astype(float).to_numpy()
    sca = (df["C_cat"] == "scaffold_tilted").astype(float).to_numpy()
    print(f"   shares: offload {off.mean()*100:.1f}% | balanced "
          f"{(1-off.mean()-sca.mean())*100:.1f}% | scaffold {sca.mean()*100:.1f}%")
    t0 = time.perf_counter()
    off_res = off - crossfit(X, off, w, folds)
    sca_res = sca - crossfit(X, sca, w, folds)
    print(f"   treatment models {time.perf_counter()-t0:.0f}s")

    z = {d: wstd(df[f"PV1{d}"].to_numpy(float), w) for d in DOMAINS}
    cat = {}
    for lab, pos, neg in CONTRASTS:
        y = np.mean([z[d] for d in pos], axis=0) - np.mean([z[d] for d in neg], axis=0)
        y_res = y - crossfit(X, y, w, folds)
        Z = np.column_stack([off_res, sca_res])
        b, se = wls_cluster(Z, y_res, w, cl)
        cat[lab] = {"offload_vs_balanced": round(float(b[0]), 5),
                    "offload_se": round(float(se[0]), 5),
                    "scaffold_vs_balanced": round(float(b[1]), 5),
                    "scaffold_se": round(float(se[1]), 5),
                    "implied_spread": round(float(b[0] - b[1]), 5)}
        c_ = cat[lab]
        print(f"   {lab:14s} offload {c_['offload_vs_balanced']:+.4f} "
              f"({c_['offload_se']:.4f}) | scaffold "
              f"{c_['scaffold_vs_balanced']:+.4f} ({c_['scaffold_se']:.4f}) | "
              f"spread {c_['implied_spread']:+.4f}")
    out["categorical"] = cat

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
