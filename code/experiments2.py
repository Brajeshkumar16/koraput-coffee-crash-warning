"""Additional experiments for the revision (Koraput panel only).

E1  Reviewer-requested origin-referenced crash labels, tested against their own nulls (index and model),
    with filled block-years excluded and with block-wise detrending.
E2  Detectability (power): synthetic panels shaped like the Koraput panel, with genuine persistence
    (AR(1), phi) or a genuine origin-time predictor (share of variance R2), run through the same test.
E3  Biennial bearing: lag-2 autocorrelation and sign alternation of season-to-season changes.
E4  Equivalence of the index and the full model: two one-sided tests with a 0.05 AUC margin.
E5  Persistence actually present: lag-1 autocorrelation with a block-cluster bootstrap interval.

Writes outputs/tables/X01..X07. Usage: python experiments2.py [--quick]
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import ewcore as ew
import reanalysis as ra

QUICK = "--quick" in sys.argv
OUT = ra.OUT
N_NULL = 200 if QUICK else 1000
N_SIM = 50 if QUICK else 300
N_MODEL_NULL = 20 if QUICK else 100
T0 = time.time()
YEARS = np.arange(2006, 2026)
BASE_J = np.where((YEARS >= 2014) & (YEARS <= 2024))[0]  # base seasons whose target is 2015-2025


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def fast_auc(y, s):
    """ROC-AUC via ranks (ties averaged); y binary."""
    y = np.asarray(y, bool)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def index_and_labels(M, theta=0.20):
    """For a complete blocks x 20 matrix: index, current-referenced and origin-referenced labels over the test pool."""
    clim = np.cumsum(M, axis=1) / np.arange(1, M.shape[1] + 1)
    j = BASE_J
    idx = ((M[:, j] - clim[:, j]) / clim[:, j]).ravel()
    cur = (M[:, j + 1] <= (1 - theta) * M[:, j]).ravel()
    ref = (M[:, j + 1] <= (1 - theta) * clim[:, j]).ravel()
    return idx, cur, ref


def null_aucs(M, which, n, rng, score_sign=1.0):
    out = np.empty(n)
    for k in range(n):
        idx, cur, ref = index_and_labels(rng.permuted(M, axis=1))
        out[k] = fast_auc(cur if which == "cur" else ref, score_sign * idx)
    return out


def main():
    raw = ew.load_panel()
    M = raw.pivot(index="Block", columns="Year", values="Yield per Ha").loc[:, YEARS].to_numpy(float)
    rng = np.random.default_rng(42)

    # sanity: vectorised version reproduces the published numbers
    idx, cur, ref = index_and_labels(M)
    assert abs(fast_auc(cur, idx) - 0.7779) < 1e-3 and abs(fast_auc(ref, idx) - 0.366) < 2e-3

    # ---------------- E1: origin-referenced labels against their own nulls
    yearly = ew.load_yearly_means()
    real = set(zip(yearly.Block, yearly.Year))
    obs_panel = raw[[(b, y) in real for b, y in zip(raw.Block, raw.Year)]].reset_index(drop=True)
    rows = []

    def two_sided(obs, null):
        c = np.nanmean(null)
        return float(np.mean(np.abs(null - c) >= abs(obs - c) - 1e-12))

    for name, panel in (("all 200 block-years", raw), ("18 filled block-years excluded", obs_panel)):
        for lab in ("cur", "clim", "below"):
            d = ra.index_samples(panel, label=lab)
            obs = ra.auc_or_nan(d.label, d["index"])
            nl = ra.index_null(panel, "shuffle", N_NULL, seed=21, label=lab)
            rows.append({"data": name, "label": {"cur": "relative to current season (published)", "clim": "relative to expanding mean at t",
                                                  "below": "below expanding mean at t"}[lab],
                         "score": "index", "n": len(d), "prevalence": d.label.mean(), "observed_auc": obs, "null_mean": np.nanmean(nl),
                         "null_q025": np.nanquantile(nl, .025), "null_q975": np.nanquantile(nl, .975), "p_two_sided": two_sided(obs, nl),
                         "null_reps": int(np.sum(~np.isnan(nl)))})
    log("E1 index rows done")
    # detrended variant: remove each block's linear trend in log yield (fitted on all years; descriptive only)
    L = np.log(M)
    tt = YEARS - YEARS.mean()
    slopes = np.array([np.polyfit(tt, L[i], 1)[0] for i in range(len(L))])
    Md = np.exp(L - slopes[:, None] * tt[None, :])
    for lab in ("cur", "clim"):
        idx_d, cur_d, ref_d = index_and_labels(Md)
        obs = fast_auc(cur_d if lab == "cur" else ref_d, idx_d)
        nl = null_aucs(Md, lab, N_NULL, np.random.default_rng(22))
        rows.append({"data": "all 200, block-wise linear trend removed (descriptive)", "label": lab, "score": "index", "n": len(idx_d),
                     "prevalence": (cur_d if lab == "cur" else ref_d).mean(), "observed_auc": obs, "null_mean": nl.mean(),
                     "null_q025": np.quantile(nl, .025), "null_q975": np.quantile(nl, .975), "p_two_sided": two_sided(obs, nl), "null_reps": N_NULL})
    # model under the origin-referenced labels
    s = ew.build_samples(raw)
    for ycol, lab in (("y_crash_clim", "relative to expanding mean at t"), ("y_below", "below expanding mean at t")):
        p, _ = ew.rolling_logreg(s, ycol)
        obs = roc_auc_score(p.actual, p.prob)
        log(f"E1 model null {ycol}")
        nl = ra.model_null(raw, N_MODEL_NULL, 31 if ycol == "y_crash_clim" else 32, ycol=ycol)
        rows.append({"data": "all 200 block-years", "label": lab, "score": "logistic model, tuned per fold", "n": len(p),
                     "prevalence": p.actual.mean(), "observed_auc": obs, "null_mean": np.nanmean(nl), "null_q025": np.nanquantile(nl, .025),
                     "null_q975": np.nanquantile(nl, .975), "p_two_sided": two_sided(obs, nl), "null_reps": int(np.sum(~np.isnan(nl)))})
    pd.DataFrame(rows).to_csv(OUT / "X01_origin_referenced_labels.csv", index=False)
    log("wrote X01")

    # ---------------- E5: persistence present, with block-cluster bootstrap
    D = np.log(M) - np.log(M).mean(1, keepdims=True)
    def ac1(D_):
        return np.corrcoef(D_[:, :-1].ravel(), D_[:, 1:].ravel())[0, 1]
    def ac2(D_):
        return np.corrcoef(D_[:, :-2].ravel(), D_[:, 2:].ravel())[0, 1]
    bs1, bs2 = [], []
    for _ in range(2000):
        pick = rng.integers(0, len(D), len(D))
        bs1.append(ac1(D[pick]))
        bs2.append(ac2(D[pick]))
    pers = pd.DataFrame([{"quantity": "lag-1 autocorrelation of block-demeaned log yield", "estimate": ac1(D),
                          "ci_lo": np.quantile(bs1, .025), "ci_hi": np.quantile(bs1, .975)},
                         {"quantity": "lag-2 autocorrelation of block-demeaned log yield", "estimate": ac2(D),
                          "ci_lo": np.quantile(bs2, .025), "ci_hi": np.quantile(bs2, .975)}])

    # ---------------- E3: biennial bearing
    dif = np.diff(np.log(M), axis=1)
    alt = (np.sign(dif[:, 1:]) * np.sign(dif[:, :-1]) < 0)
    alt_null = []
    for _ in range(2000):
        P = rng.permuted(np.log(M), axis=1)
        dd = np.diff(P, axis=1)
        alt_null.append(np.mean(np.sign(dd[:, 1:]) * np.sign(dd[:, :-1]) < 0))
    bien = pd.concat([pers, pd.DataFrame([
        {"quantity": "share of consecutive season-to-season changes that alternate in sign", "estimate": alt.mean(),
         "ci_lo": np.quantile(alt_null, .025), "ci_hi": np.quantile(alt_null, .975),
         "note": "interval = 95% range under shuffled seasons (expected about 2/3 for independent values)"},
        {"quantity": "blocks whose alternation share exceeds 0.8", "estimate": int((alt.mean(1) > 0.8).sum()), "ci_lo": np.nan, "ci_hi": np.nan,
         "note": "strong biennial bearing would push alternation towards 1"}])], ignore_index=True)
    bien.to_csv(OUT / "X02_persistence_and_biennial.csv", index=False)
    log("wrote X02")

    # ---------------- E4: equivalence (TOST, margin 0.05) using block-cluster bootstrap 90% intervals
    pf = pd.read_csv(OUT / "pred_logreg_perfold.csv", index_col=0)
    pool = s[s.target_year.isin(ew.TEST_YEARS)]
    d = pf.join(pool[[ew.INDEX]])
    groups = {b: g for b, g in d.groupby("Block")}
    diffs = []
    for _ in range(4000):
        pick = rng.choice(list(groups), len(groups))
        g = pd.concat([groups[b] for b in pick])
        if g.actual.nunique() == 2:
            diffs.append(fast_auc(g.actual, g[ew.INDEX]) - fast_auc(g.actual, g.prob))
    diffs = np.array(diffs)
    lo90, hi90 = np.quantile(diffs, [.05, .95])
    est = fast_auc(d.actual, d[ew.INDEX]) - fast_auc(d.actual, d.prob)
    pd.DataFrame([{"comparison": "index minus logistic model tuned per fold", "estimate": est, "ci90_lo": lo90, "ci90_hi": hi90,
                   "margin": 0.05, "equivalent_within_margin": bool(lo90 > -0.05 and hi90 < 0.05),
                   "smallest_margin_supported": max(abs(lo90), abs(hi90)), "resampling": "blocks (10 clusters), 4000 draws"}]).to_csv(
        OUT / "X03_equivalence.csv", index=False)
    log("wrote X03")

    # ---------------- E2: detectability
    lm, ls = np.log(M).mean(1), np.log(M).std(1)

    def sim_ar1(phi, r):
        u = np.empty_like(M)
        u[:, 0] = r.standard_normal(len(M))
        for t in range(1, M.shape[1]):
            u[:, t] = phi * u[:, t - 1] + np.sqrt(1 - phi ** 2) * r.standard_normal(len(M))
        return np.exp(lm[:, None] + ls[:, None] * u)

    rows = []
    sim_rng = np.random.default_rng(101)
    for phi in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        rej_ref, rej_cur, obs_list = [], [], []
        for _ in range(N_SIM):
            S = sim_ar1(phi, sim_rng)
            idx_s, cur_s, ref_s = index_and_labels(S)
            o_ref = fast_auc(ref_s, -idx_s)          # persistence: a high current level predicts no fall below the mean
            n_ref = null_aucs(S, "clim", 200, sim_rng, score_sign=-1.0)
            o_cur = fast_auc(cur_s, idx_s)
            n_cur = null_aucs(S, "cur", 200, sim_rng)
            rej_ref.append(o_ref > np.quantile(n_ref, 0.95))
            rej_cur.append(o_cur > np.quantile(n_cur, 0.95))
            obs_list.append(o_cur)
        rows.append({"design": "AR(1) persistence", "parameter": "phi", "value": phi, "sims": N_SIM,
                     "power_origin_referenced_label": np.mean(rej_ref), "power_published_label_above_null": np.mean(rej_cur),
                     "mean_auc_published_label": np.mean(obs_list)})
        log(f"E2 phi={phi}")
    # a genuine predictor known at the origin: log y(t+1) - mu = sqrt(R2) z(t) + sqrt(1-R2) e
    for r2 in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        rej = []
        for _ in range(N_SIM):
            Z = sim_rng.standard_normal(M.shape)
            E = sim_rng.standard_normal(M.shape)
            U = np.empty_like(M)
            U[:, 0] = E[:, 0]
            U[:, 1:] = -np.sqrt(r2) * Z[:, :-1] + np.sqrt(1 - r2) * E[:, 1:]   # a high z at t precedes a low next season
            S = np.exp(lm[:, None] + ls[:, None] * U)
            clim = np.cumsum(S, axis=1) / np.arange(1, S.shape[1] + 1)
            j = BASE_J
            ref_s = (S[:, j + 1] <= 0.8 * clim[:, j]).ravel()
            score = Z[:, j].ravel()
            o = fast_auc(ref_s, score)
            nl = np.array([fast_auc(ref_s, sim_rng.permuted(Z, axis=1)[:, j].ravel()) for _ in range(200)])
            rej.append(o > np.quantile(nl, 0.95))
        rows.append({"design": "predictor known at the origin", "parameter": "R2 of next-season log yield", "value": r2, "sims": N_SIM,
                     "power_origin_referenced_label": np.mean(rej), "power_published_label_above_null": np.nan, "mean_auc_published_label": np.nan})
        log(f"E2 R2={r2}")
    pd.DataFrame(rows).to_csv(OUT / "X04_detectability.csv", index=False)
    log("wrote X04, done")


if __name__ == "__main__":
    main()
