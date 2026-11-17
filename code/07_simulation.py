from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "simulation.json"

N, REPS, SEED = 20_000, 2_000, 20261116
N_POP = 2_000_000
RHO, GAMMA, SIGMA, PHI, LAM_A = 0.40, 0.12, 0.88, 0.35, 0.80
NA, NB = 3, 4

SCENARIOS = [
    ("1 Equal loading, no effect",                    dict(kappa=1.0, kappa_z=1.0)),
    ("2 Unequal loading, no effect",                  dict(kappa=1.2, kappa_z=1.2)),
    ("3 Unequal loading, effect on A only",           dict(kappa=1.2, kappa_z=1.2, dA=-0.015)),
    ("4 As 2, anchor understates the ratio",          dict(kappa=1.2, kappa_z=1.1)),
    ("5 As 2, anchor overstates the ratio",           dict(kappa=1.2, kappa_z=1.3)),
    ("6 Unequal loading, common effect on all scales", dict(kappa=1.2, kappa_z=1.2, dA=-0.05, dB=-0.05)),
    ("7 Unequal loading, effect through general factor", dict(kappa=1.2, kappa_z=1.2, eta=-0.05)),
    ("8 Equal loading, family-specific confounder",   dict(kappa=1.0, kappa_z=1.0, g=0.05)),
]


def params(kappa, kappa_z, dA=0.0, dB=0.0, eta=0.0, g=0.0):
    lam = np.r_[np.full(NA, LAM_A), np.full(NB, LAM_A / kappa)]
    pi = np.r_[np.full(NA, RHO * (LAM_A / kappa) * (kappa_z - kappa)), np.zeros(NB)]
    delta = np.r_[np.full(NA, dA), np.full(NB, dB)] + eta * lam
    sd_eps = np.sqrt(1 - lam ** 2 - PHI ** 2)
    return dict(lam=lam, pi=pi, delta=delta, sd_eps=sd_eps, g=g, kappa=kappa)


def draw(n, p, rng):
    alpha, fA, fB, nu, eta = rng.standard_normal((5, n))
    Z = RHO * alpha + np.sqrt(1 - RHO ** 2) * nu
    C = -GAMMA * alpha - p["g"] * fA + SIGMA * eta
    Y = np.empty((n, NA + NB))
    for d in range(NA + NB):
        Y[:, d] = (p["lam"][d] * alpha + PHI * (fA if d < NA else fB) + p["pi"][d] * Z
                   + p["delta"][d] * C + p["sd_eps"][d] * rng.standard_normal(n))
    return Z, C, Y


def slope(x, y):
    sxx = x @ x
    b = (x @ y) / sxx
    infl = x * (y - b * x) / sxx * len(x)
    return b, np.sqrt((infl ** 2).sum()) / len(x), infl


def estimate(Z, C, Y):
    Ys = (Y - Y.mean(0)) / Y.std(0)
    yA, yB = Ys[:, :NA].mean(1), Ys[:, NA:].mean(1)
    Zc = Z - Z.mean()
    res = lambda v: (v - v.mean()) - Zc * (Zc @ (v - v.mean())) / (Zc @ Zc)
    Ct, rA, rB = res(C), res(yA), res(yB)

    thA, _, ifA = slope(Ct, rA)
    thB, _, ifB = slope(Ct, rB)
    naive, se_naive, _ = slope(Ct, rA - rB)
    ctrl, se_ctrl, _ = slope(Ct, res(Ys[:, NA] - Ys[:, NA + 1]))

    bA, _, jA = slope(Zc, yA)
    bB, _, jB = slope(Zc, yB)
    anc, se_anc, _ = slope(Zc, yA - yB)
    k = bA / bB
    cal, se_fixed, _ = slope(Ct, rA - k * rB)
    infl = ifA - k * ifB - thB * (jA / bB - bA * jB / bB ** 2)
    se_full = np.sqrt((infl ** 2).sum()) / len(infl)
    return dict(naive=naive, se_naive=se_naive, ctrl=ctrl, se_ctrl=se_ctrl, anc=anc,
                se_anc=se_anc, kappa_hat=k, cal=cal, se_fixed=se_fixed, se_full=se_full,
                thA=thA, thB=thB)


def main():
    rng = np.random.default_rng(SEED)
    out = {"design": dict(n=N, reps=REPS, rho=RHO, gamma=GAMMA, sigma=SIGMA, phi=PHI,
                          lambda_A=LAM_A, n_population_draw=N_POP)}
    hdr = (f"{'scenario':50s}{'tau':>8s}{'naive':>8s}{'rej':>6s}{'ctrl':>6s}{'anch':>6s}"
           f"{'kap_Z':>7s}{'target':>8s}{'recal':>8s}{'sd':>7s}{'cov_fix':>8s}{'cov_est':>8s}")
    print(hdr)
    for name, kw in SCENARIOS:
        p = params(**kw)
        Zp, Cp, Yp = draw(N_POP, p, rng)
        sd = Yp.std(0)
        dstd = p["delta"] / sd
        tau = dstd[:NA].mean() - dstd[NA:].mean()
        kap_conf = (p["lam"] / sd)[:NA].mean() / (p["lam"] / sd)[NA:].mean()
        pop = estimate(Zp, Cp, Yp)
        target = dstd[:NA].mean() - pop["kappa_hat"] * dstd[NA:].mean()
        del Zp, Cp, Yp

        R = {k: np.empty(REPS) for k in pop}
        for i in range(REPS):
            e = estimate(*draw(N, p, rng))
            for k, v in e.items():
                R[k][i] = v
        rej = lambda est, se: float(np.mean(np.abs(R[est] / R[se]) > 1.96))
        cov = lambda se: float(np.mean(np.abs(R["cal"] - target) <= 1.96 * R[se]))
        row = dict(
            true_differential_effect=float(tau), confounder_kappa_std=float(kap_conf),
            anchor_kappa_population=float(pop["kappa_hat"]),
            naive_mean=float(R["naive"].mean()), naive_sd=float(R["naive"].std()),
            naive_bias=float(R["naive"].mean() - tau), naive_reject=rej("naive", "se_naive"),
            control_mean=float(R["ctrl"].mean()), control_reject=rej("ctrl", "se_ctrl"),
            anchor_test_reject=rej("anc", "se_anc"),
            kappa_hat_mean=float(R["kappa_hat"].mean()), kappa_hat_sd=float(R["kappa_hat"].std()),
            recal_target=float(target), recal_population_value=float(pop["cal"]),
            recal_mean=float(R["cal"].mean()), recal_sd=float(R["cal"].std()),
            recal_mean_se_kappa_fixed=float(R["se_fixed"].mean()),
            recal_mean_se_kappa_estimated=float(R["se_full"].mean()),
            recal_coverage_kappa_fixed=cov("se_fixed"), recal_coverage_kappa_estimated=cov("se_full"),
            recal_reject=float(np.mean(np.abs(R["cal"] / R["se_full"]) > 1.96)),
            level_coef_A=float(R["thA"].mean()), level_coef_B=float(R["thB"].mean()),
        )
        out[name] = row
        print(f"{name:50s}{tau:+8.4f}{row['naive_mean']:+8.4f}{row['naive_reject']:6.2f}"
              f"{row['control_reject']:6.2f}{row['anchor_test_reject']:6.2f}"
              f"{row['kappa_hat_mean']:7.3f}{target:+8.4f}{row['recal_mean']:+8.4f}"
              f"{row['recal_sd']:7.4f}{row['recal_coverage_kappa_fixed']:8.3f}"
              f"{row['recal_coverage_kappa_estimated']:8.3f}", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
