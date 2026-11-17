from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROCESSED = HERE.parent / "data" / "processed"
STU = PROCESSED / "CY09_MS_STU_PUF.parquet"
CACHE = Path(os.environ.get("AIUSE_CACHE", HERE.parent / "data" / "cache"))
CACHE.mkdir(parents=True, exist_ok=True)
OUT = CACHE / "aiuse_sample.parquet"

SEED = 42

OFFLOAD = ["ST438Q01DA", "ST438Q03DA"]
SCAFFOLD = ["ST438Q02DA", "ST438Q04DA"]
AI_ITEMS = OFFLOAD + SCAFFOLD

DOMAINS = ["READ", "SCIE", "MATH", "CPRO", "CMPS", "CMOD", "CPPK"]
N_PV = 10

CONTINUOUS = ["ESCS", "HISEI", "PAREDINT", "HOMEPOS", "AGE", "GRADE"]
CATEGORICAL = ["IMMIG", "ST004D01T", "REPEAT"]

MISSING_CODES = {
    "AI": 5,
    "IMMIG": 3,
    "ST004D01T": 2,
    "REPEAT": 1,
}


def main() -> None:
    brr = [f"W_FSTURWT{i}" for i in range(1, 81)]
    pvs = [f"PV{i}{d}" for d in DOMAINS for i in range(1, N_PV + 1)]
    cols = (["CNT", "CNTSCHID", "CNTSTUID", "W_FSTUWT"] + AI_ITEMS
            + CONTINUOUS + CATEGORICAL + pvs + brr)

    print(f"reading {STU.name} ({STU.stat().st_size/1e9:.2f} GB) ...")
    df = pd.read_parquet(STU, columns=cols)
    print(f"  raw: {len(df):,} rows, {df['CNT'].nunique()} countries")

    for c in AI_ITEMS:
        df[c] = df[c].where(df[c] <= MISSING_CODES["AI"])
    df["offload"] = df[OFFLOAD].mean(axis=1)
    df["scaffold"] = df[SCAFFOLD].mean(axis=1)
    df["I"] = (df["offload"] + df["scaffold"]) / 2.0
    df["C"] = df["offload"] - df["scaffold"]

    df["C_cat"] = np.select(
        [df["C"] > 0, df["C"] < 0], ["offload_tilted", "scaffold_tilted"],
        default="balanced",
    )
    df.loc[df["C"].isna(), "C_cat"] = np.nan

    for c in CATEGORICAL:
        df[c] = df[c].where(df[c] <= MISSING_CODES[c] + 1)

    gender = df["ST004D01T"]
    immig = df["IMMIG"]
    df["FEMALE"] = (gender == 1).astype("float").where(gender.notna())
    df["IMMIG_2"] = (immig == 2).astype("float").where(immig.notna())
    df["IMMIG_3"] = (immig == 3).astype("float").where(immig.notna())

    for c in CONTINUOUS + ["FEMALE", "IMMIG_2", "IMMIG_3", "REPEAT"]:
        df[f"{c}_miss"] = df[c].isna().astype("float")

    n0 = len(df)
    keep = df[AI_ITEMS].notna().all(axis=1)
    print(f"  all four AI items present: {keep.sum():,}  (dropped {n0-keep.sum():,})")
    keep &= df[[f"PV1{d}" for d in DOMAINS]].notna().any(axis=1)
    keep &= df["W_FSTUWT"].notna()
    df = df[keep].copy()
    print(f"  analytic sample: {len(df):,} rows, {df['CNT'].nunique()} countries, "
          f"{df['CNTSCHID'].nunique():,} schools")

    g = df.groupby("CNT")["W_FSTUWT"]
    df["W_SENATE"] = df["W_FSTUWT"] / g.transform("sum") * 1000.0

    for c in CONTINUOUS:
        df[c] = df[c].fillna(df[c].median())
    for c in ["FEMALE", "IMMIG_2", "IMMIG_3", "REPEAT"]:
        df[c] = df[c].fillna(df[c].mean())

    keep_cols = (["CNT", "CNTSCHID", "CNTSTUID", "W_FSTUWT", "W_SENATE",
                  "offload", "scaffold", "I", "C", "C_cat"]
                 + CONTINUOUS + ["FEMALE", "IMMIG_2", "IMMIG_3", "REPEAT"]
                 + [c for c in df.columns if c.endswith("_miss")]
                 + pvs + brr)
    out = df[keep_cols].copy()
    out.to_parquet(OUT, index=False)

    print(f"\ncached -> {OUT}  ({out.shape[0]:,} x {out.shape[1]})")
    print(f"\ncorr(C, I) = {out[['C','I']].corr().iloc[0,1]:.4f}   "
          "(near-zero by construction — this is what the design rests on)")
    print("\ncomposition categories:")
    print((out["C_cat"].value_counts(normalize=True) * 100).round(1).to_string())

    print("\nwithin- vs between-country variance of the treatment:")
    for v in ["offload", "scaffold", "I", "C"]:
        s = out[[v, "CNT"]].dropna()
        grand = s[v].mean()
        between = ((s.groupby("CNT")[v].transform("mean") - grand) ** 2).mean()
        total = ((s[v] - grand) ** 2).mean()
        print(f"  {v:9s} within-country {100*(1-between/total):5.1f}%")

    meta = {
        "n": int(len(out)),
        "countries": int(out["CNT"].nunique()),
        "schools": int(out["CNTSCHID"].nunique()),
        "corr_C_I": round(float(out[["C", "I"]].corr().iloc[0, 1]), 4),
        "C_cat_pct": {k: round(v * 100, 2) for k, v in
                      out["C_cat"].value_counts(normalize=True).items()},
    }
    (CACHE / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmeta -> {CACHE/'sample_meta.json'}")


if __name__ == "__main__":
    main()
