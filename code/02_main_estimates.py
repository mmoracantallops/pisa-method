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
OUT = HERE.parent / "results" / "main_estimates.json"

SEED, N_FOLDS = 42, 5
N_PV, N_REP = 10, 80
FAY_K = 0.5
BRR_DIVISOR = N_REP * (1 - FAY_K) ** 2
BRR_PVS = [1, 5, 9]

DOMAINS = ["READ", "SCIE", "MATH", "CPRO", "CMPS", "CMOD", "CPPK"]
CONV, COMP = ["READ", "SCIE", "MATH"], ["CPRO", "CMPS", "CMOD", "CPPK"]
CONTROLS = ["I", "ESCS", "HISEI", "PAREDINT", "HOMEPOS", "AGE", "GRADE",
            "FEMALE", "IMMIG_2", "IMMIG_3", "REPEAT"]

PRIMARY = "CONV - COMP"
CONTRASTS = [
    ("CONV - COMP", CONV, COMP, "primary"),
    ("READ - CPRO", ["READ"], ["CPRO"], "sharpest single-domain"),
    ("CMPS - CMOD", ["CMPS"], ["CMOD"], "clean negative control"),
    ("CPPK - CMOD", ["CPPK"], ["CMOD"], "clean negative control"),
    ("READ - SCIE", ["READ"], ["SCIE"], "gradient probe"),
    ("SCIE - MATH", ["SCIE"], ["MATH"], "gradient probe"),
]
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


def senate(raw: np.ndarray, country_codes: np.ndarray) -> np.ndarray:
    s = pd.Series(raw).groupby(country_codes).transform("sum").to_numpy()
    return raw / s * 1000.0


def wstd(x, w):
    mu = np.average(x, weights=w)
    return (x - mu) / np.sqrt(np.average((x - mu) ** 2, weights=w))


def theta(C_res, y, X, w, folds):
    y_res = y - crossfit(X, y, w, folds)
    return float((w * C_res * y_res).sum() / (w * C_res ** 2).sum())


def outcome(df, m, pos, neg, w):
    z = {d: wstd(df[f"PV{m}{d}"].to_numpy(float), w) for d in set(pos + neg)}
    y = np.mean([z[d] for d in pos], axis=0)
    if neg:
        y = y - np.mean([z[d] for d in neg], axis=0)
    return y


def load_existing() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(obj: dict) -> None:
    OUT.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-levels", action="store_true",
                    help="skip the 7 level estimates (not identified; context only)")
    args = ap.parse_args()

    df = pd.read_parquet(SAMPLE)
    ok = np.ones(len(df), bool)
    for d in DOMAINS:
        ok &= np.isfinite(df[f"PV1{d}"].to_numpy(float))
    df = df[ok & ~np.isclose(df["I"], 1.0)].reset_index(drop=True)
    n = len(df)
    print(f"AI users, all seven domains: {n:,} | {df['CNT'].nunique()} countries")

    miss = [c for c in df.columns if c.endswith("_miss")]
    cnt_d = pd.get_dummies(df["CNT"], prefix="CNT", drop_first=True).astype(np.float64)
    X = pd.concat([df[CONTROLS + miss].astype(np.float64), cnt_d], axis=1).to_numpy()
    C = df["C"].to_numpy(float)
    country = df["CNT"].to_numpy()
    sch = df["CNTSCHID"]
    cluster = np.where(sch.notna(), "S" + sch.astype("string"),
                       "NOSCH_" + df["CNT"].astype(str)).astype(str)

    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, C, cluster))

    print("senate-normalising all weights on the estimation sample ...")
    W = [senate(df["W_FSTUWT"].to_numpy(float), country)]
    for r in range(1, N_REP + 1):
        W.append(senate(df[f"W_FSTURWT{r}"].to_numpy(float), country))
    chk0 = pd.Series(W[0]).groupby(country).sum()
    print(f"  final weight per-system sums: min {chk0.min():.2f} max {chk0.max():.2f}")
    chk = pd.Series(W[1]).groupby(country).sum()
    print(f"  replicate 1 per-country sums: min {chk.min():.1f} max {chk.max():.1f} "
          "(constant => OK)")

    results = load_existing()
    results.setdefault("meta", {
        "n": int(n), "countries": int(df["CNT"].nunique()),
        "n_pv": N_PV, "n_rep": N_REP, "fay_k": FAY_K,
        "brr_divisor": BRR_DIVISOR, "estimand": "AI users (I>1), composition C",
    })
    save(results)

    t0 = time.perf_counter()
    print(f"\n[1/3] treatment models E[C|X] for {len(W)} weights (shared across contrasts)")
    C_res = []
    for i, w in enumerate(W):
        C_res.append(C - crossfit(X, C, w, folds))
        if i % 10 == 0:
            el = time.perf_counter() - t0
            print(f"   weight {i:>3}/{len(W)-1}  {el/60:.1f} min elapsed", flush=True)
    print(f"   done in {(time.perf_counter()-t0)/60:.1f} min")

    def run_one(label, pos, neg, note, brr_pvs):
        if label in results.get("contrasts", {}):
            print(f"   [skip] {label} already done")
            return
        t1 = time.perf_counter()
        thetas = []
        for m in range(1, N_PV + 1):
            y = outcome(df, m, pos, neg, W[0])
            thetas.append(theta(C_res[0], y, X, W[0], folds))
        th0 = float(np.mean(thetas))
        v_imp = (1 + 1 / N_PV) * float(np.var(thetas, ddof=1))

        samp = {}
        for m in brr_pvs:
            y = outcome(df, m, pos, neg, W[0])
            base = theta(C_res[0], y, X, W[0], folds)
            reps = [theta(C_res[r], y, X, W[r], folds) for r in range(1, N_REP + 1)]
            samp[f"PV{m}"] = float(np.sum((np.array(reps) - base) ** 2) / BRR_DIVISOR)
        v_samp = float(np.mean(list(samp.values())))

        se = float(np.sqrt(v_samp + v_imp))
        rec = {"note": note, "theta": round(th0, 5), "se": round(se, 5),
               "t": round(th0 / se, 2),
               "ci95": [round(th0 - 1.96 * se, 5), round(th0 + 1.96 * se, 5)],
               "v_sampling": v_samp, "v_imputation": v_imp,
               "pct_var_imputation": round(100 * v_imp / (v_samp + v_imp), 1),
               "brr_by_pv": {k: round(v, 10) for k, v in samp.items()},
               "pv_estimates": [round(x, 5) for x in thetas],
               "minutes": round((time.perf_counter() - t1) / 60, 1)}
        results.setdefault("contrasts", {})[label] = rec
        save(results)
        print(f"   {label:14s} {th0:+.4f}  se {se:.4f}  t {th0/se:+6.1f}   "
              f"[{rec['minutes']:.0f} min]  {note}", flush=True)

    print(f"\n[2/3] contrasts ({len(CONTRASTS)})")
    for label, pos, neg, note in CONTRASTS:
        run_one(label, pos, neg, note,
                BRR_PVS if label == PRIMARY else [1])

    if not args.no_levels:
        print(f"\n[3/3] level estimates ({len(DOMAINS)}) — NOT identified, context only")
        for d in DOMAINS:
            run_one(f"LEVEL {d}", [d], [], "level (not identified)", [1])
    else:
        print("\n[3/3] level estimates skipped (--no-levels)")

    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
