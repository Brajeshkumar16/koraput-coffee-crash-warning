"""Re-analysis for the revision of paper 2 (Discover Sustainability, MSID d99a7a07).

Every number in the revised manuscript and the response to reviewers comes from the tables this
script writes to outputs/tables/. Seed 42 throughout; run twice to confirm identical output.
Usage (paper-3 venv):  python reanalysis.py [--quick]
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

import ewcore as ew

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
N_INDEX_NULL = 200 if QUICK else 2000
N_MODEL_NULL = 20 if QUICK else 200
N_MODEL_NULL_SMALL = 10 if QUICK else 100
B_BOOT = 500 if QUICK else 2000
JOBS = 8
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    log(f"wrote {name}")


# ----------------------------------------------------------------------------- fast index evaluation
def index_samples(panel, theta=0.20, horizon=1, label="cur", first_base=2008, test_from=2015, test_to=2025):
    """Index and label for every (block, t) where t-2..t and t+h are present, from a long panel.
    Mirrors build_samples: expanding mean over all present years <= t."""
    rows = []
    for b, g in panel.groupby("Block"):
        g = g.sort_values("Year")
        yrs, y = g.Year.to_numpy(), g["Yield per Ha"].to_numpy(float)
        present = dict(zip(yrs, y))
        for i, t in enumerate(yrs):
            if t < first_base or any((t - k) not in present for k in (1, 2)) or (t + horizon) not in present:
                continue
            if not (test_from <= t + horizon <= test_to):
                continue
            hist = y[yrs <= t]
            clim = hist.mean()
            yt, yn = present[t], present[t + horizon]
            idx = (yt - clim) / clim
            if label == "cur":
                lab = yn <= (1 - theta) * yt
            elif label == "clim":
                lab = yn <= (1 - theta) * clim
            else:
                lab = yn < clim
            rows.append((b, t, t + horizon, idx, yt, int(lab)))
    return pd.DataFrame(rows, columns=["Block", "base_year", "target_year", "index", "Yield_t", "label"])


def auc_or_nan(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def shuffle_panel(panel, rng):
    """Permute whole rows (all variables of a block-year together) among the years of each block."""
    parts = []
    for b, g in panel.groupby("Block"):
        g = g.copy()
        perm = rng.permutation(len(g))
        cols = [c for c in g.columns if c not in ("Block", "Year")]
        g[cols] = g[cols].to_numpy()[perm]
        parts.append(g)
    return pd.concat(parts).sort_values(["Block", "Year"]).reset_index(drop=True)


def lognormal_panel(panel, rng):
    """Independent lognormal yields with each block's log-mean and log-SD (other columns kept)."""
    p = panel.copy()
    for b, g in p.groupby("Block"):
        ly = np.log(g["Yield per Ha"].to_numpy(float))
        p.loc[g.index, "Yield per Ha"] = np.exp(ly.mean() + ly.std() * rng.standard_normal(len(g)))
    return p


def index_null(panel, kind, n, seed=42, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q = shuffle_panel(panel, rng) if kind == "shuffle" else lognormal_panel(panel, rng)
        d = index_samples(q, **kw)
        out.append(auc_or_nan(d.label, d["index"]))
    return np.array(out)


def null_row(analysis, observed, null, kind, n_obs, prevalence, note=""):
    null = null[~np.isnan(null)]
    return {"analysis": analysis, "observed_auc": observed, "null": kind, "null_reps": len(null),
            "null_mean": null.mean(), "null_q025": np.quantile(null, .025), "null_q975": np.quantile(null, .975),
            "share_null_ge_observed": float(np.mean(null >= observed - 1e-12)), "n": n_obs, "prevalence": prevalence, "note": note}


# ----------------------------------------------------------------------------- model-based nulls
def model_auc_on(panel, ycol="y_crash", tycol="target_year", years=ew.TEST_YEARS, protocol="rolling"):
    s = ew.build_samples(panel)
    if ycol == "y_crash_h2":
        s = s[s.y_crash_h2.notna()].copy()
        s["y_crash_h2"] = s.y_crash_h2.astype(int)
    if protocol == "rolling":
        p, _ = ew.rolling_logreg(s, ycol, tycol, years)
    elif protocol == "lobo_all":
        p = lobo(s, ycol, earlier_only=False)
    else:
        p = lobo(s, ycol, earlier_only=True)
    return auc_or_nan(p.actual, p.prob)


def model_null(panel, n, seed, **kw):
    seeds = np.random.SeedSequence(seed).spawn(n)
    res = Parallel(n_jobs=JOBS)(delayed(_one_model_null)(panel, np.random.default_rng(s), kw) for s in seeds)
    return np.array(res)


def _one_model_null(panel, rng, kw):
    return model_auc_on(shuffle_panel(panel, rng), **kw)


def lobo(s, ycol="y_crash", earlier_only=False):
    """Leave-one-block-out with the per-fold tuned logistic model.
    earlier_only=False: published design (train on other blocks, all seasons).
    earlier_only=True: train on other blocks with target season <= Y-1 only, test block at season Y."""
    parts = []
    for b in sorted(s.Block.unique()):
        if not earlier_only:
            tr, te = s[s.Block != b], s[s.Block == b]
            C, p = ew.tune_logreg(tr, ycol, "target_year")
            m = ew.make_logreg(C, p).fit(tr[ew.FEATURES].to_numpy(), tr[ycol].to_numpy())
            parts.append(pd.DataFrame({"prob": m.predict_proba(te[ew.FEATURES].to_numpy())[:, 1], "actual": te[ycol].to_numpy(),
                                       "Block": b, "target_year": te.target_year.to_numpy()}))
        else:
            for Y in ew.TEST_YEARS:
                tr, te = s[(s.Block != b) & (s.target_year <= Y - 1)], s[(s.Block == b) & (s.target_year == Y)]
                if te.empty or tr[ycol].nunique() < 2:
                    continue
                C, p = ew.tune_logreg(tr, ycol, "target_year")
                m = ew.make_logreg(C, p).fit(tr[ew.FEATURES].to_numpy(), tr[ycol].to_numpy())
                parts.append(pd.DataFrame({"prob": m.predict_proba(te[ew.FEATURES].to_numpy())[:, 1], "actual": te[ycol].to_numpy(),
                                           "Block": b, "target_year": Y}))
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- cluster bootstrap
def cluster_boot(df, cols, cluster, B=2000, seed=42):
    """Percentile CIs for AUC of each score column and for their difference, resampling clusters."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    draws = []
    for _ in range(B):
        pick = rng.choice(keys, size=len(keys), replace=True)
        d = pd.concat([groups[k] for k in pick])
        if d.actual.nunique() < 2:
            continue
        draws.append([roc_auc_score(d.actual, d[c]) for c in cols])
    draws = np.array(draws)
    return draws


# ----------------------------------------------------------------------------- main
def main():
    raw = ew.load_panel()
    src_yearly = ew.load_yearly_means()
    real = set(zip(src_yearly.Block, src_yearly.Year))
    raw["filled"] = [(b, y) not in real for b, y in zip(raw.Block, raw.Year)]
    panel = raw.drop(columns="filled")
    panel_obs = raw[~raw.filled].drop(columns="filled").reset_index(drop=True)
    s = ew.build_samples(panel)
    pool = s[s.target_year.isin(ew.TEST_YEARS)].copy()
    log(f"samples {len(s)}, test pool {len(pool)}")

    # --- headline reproduction and the corrected model
    pf, chosen = ew.rolling_logreg(s)
    import collections
    order = {cp: i for i, cp in enumerate(ew.LOGREG_GRID)}
    modal = sorted(collections.Counter(chosen).items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]
    pm, _ = ew.rolling_logreg(s, params=modal)
    pf.to_csv(OUT / "pred_logreg_perfold.csv")
    pm.to_csv(OUT / "pred_logreg_modal_published.csv")
    v2 = pd.read_csv(ew.DATA / "original_submission" / "T10_bootstrap_delong.csv")
    headline = pd.DataFrame([
        {"quantity": "test predictions (target seasons 2015-2025)", "value": len(pool)},
        {"quantity": "usable samples (base years 2008-2024)", "value": len(s)},
        {"quantity": "crash prevalence, all usable samples", "value": s.y_crash.mean()},
        {"quantity": "crash prevalence, test predictions", "value": pool.y_crash.mean()},
        {"quantity": "index ROC-AUC", "value": roc_auc_score(pool.y_crash, pool[ew.INDEX])},
        {"quantity": "index PR-AUC", "value": average_precision_score(pool.y_crash, pool[ew.INDEX])},
        {"quantity": "published model (modal settings across folds) ROC-AUC", "value": roc_auc_score(pm.actual, pm.prob)},
        {"quantity": "published model PR-AUC", "value": average_precision_score(pm.actual, pm.prob)},
        {"quantity": "corrected model (tuned inside each fold) ROC-AUC", "value": roc_auc_score(pf.actual, pf.prob)},
        {"quantity": "corrected model PR-AUC", "value": average_precision_score(pf.actual, pf.prob)},
        {"quantity": "published calibrated model ROC-AUC (Figure 3, DeLong comparison)", "value": float(v2.loc[0, "Value"])},
        {"quantity": "published DeLong p (index vs calibrated model)", "value": 0.1115},
        {"quantity": f"modal settings used in the published run: C={modal[0]}, penalty={modal[1]}", "value": np.nan},
    ])
    save(headline, "T01_headline_reproduction")

    # --- Reviewer 2's seven checks
    ip = index_samples(panel)
    M = panel.pivot(index="Block", columns="Year", values="Yield per Ha")
    dm = M.sub(M.mean(1), axis=0).to_numpy()
    dd = np.diff(M.to_numpy(), axis=1)
    perm = index_null(panel, "shuffle", N_INDEX_NULL)
    wn = index_null(panel, "lognormal", N_INDEX_NULL, seed=7)
    clim_lab = index_samples(panel, label="clim")
    sem = panel[panel.Block == "Semiliguda"].set_index("Year").loc[[2024, 2025]]
    r2 = pd.DataFrame([
        {"check": "Current-season yield alone, ROC-AUC", "reviewer2": 0.828, "ours": auc_or_nan(ip.label, ip.Yield_t)},
        {"check": "Within-block year permutation, mean AUC", "reviewer2": 0.810, "ours": perm.mean(),
         "ours_range": f"{np.quantile(perm, .025):.3f}-{np.quantile(perm, .975):.3f}", "reps": len(perm)},
        {"check": "Share of permutations >= observed 0.778", "reviewer2": 0.82, "ours": float(np.mean(perm >= auc_or_nan(ip.label, ip["index"])))},
        {"check": "Independent lognormal series, mean AUC", "reviewer2": 0.816, "ours": wn.mean(),
         "ours_range": f"{np.quantile(wn, .025):.3f}-{np.quantile(wn, .975):.3f}", "reps": len(wn)},
        {"check": "Lag-1 autocorrelation, block-demeaned yield", "reviewer2": 0.045, "ours": np.corrcoef(dm[:, :-1].ravel(), dm[:, 1:].ravel())[0, 1]},
        {"check": "Lag-1 autocorrelation, first differences", "reviewer2": -0.36, "ours": np.corrcoef(dd[:, :-1].ravel(), dd[:, 1:].ravel())[0, 1]},
        {"check": "Index vs expanding-mean crash label, AUC", "reviewer2": 0.37, "ours": auc_or_nan(clim_lab.label, clim_lab["index"])},
        {"check": "Prevalence under the expanding-mean label", "reviewer2": 0.49, "ours": clim_lab.label.mean()},
        {"check": "Semiliguda 2024 and 2025 identical in every variable", "reviewer2": 1, "ours": float((sem.iloc[0] == sem.iloc[1]).all())},
    ])
    save(r2, "T02_reviewer2_checks")

    # --- nulls beside every AUC
    rows = []
    obs = auc_or_nan(ip.label, ip["index"])
    rows.append(null_row("index, one season ahead", obs, perm, "within-block shuffle", len(ip), ip.label.mean()))
    rows.append(null_row("index, one season ahead", obs, wn, "independent lognormal", len(ip), ip.label.mean()))
    log("model null, one season ahead")
    mn = model_null(panel, N_MODEL_NULL, 11)
    rows.append(null_row("logistic model (tuned per fold), one season ahead", roc_auc_score(pf.actual, pf.prob), mn,
                         "within-block shuffle", len(pf), pf.actual.mean()))
    ih2 = index_samples(panel, horizon=2, test_from=2017)
    rows.append(null_row("index, two seasons ahead (crash at t+2 relative to Y_t)", auc_or_nan(ih2.label, ih2["index"]),
                         index_null(panel, "shuffle", N_INDEX_NULL, horizon=2, test_from=2017), "within-block shuffle", len(ih2), ih2.label.mean()))
    s2 = s[s.y_crash_h2.notna()].copy()
    s2["y_crash_h2"] = s2.y_crash_h2.astype(int)
    p2, _ = ew.rolling_logreg(s2, "y_crash_h2", "target_year_h2", ew.H2_YEARS)
    log("model null, two seasons ahead")
    rows.append(null_row("logistic model, two seasons ahead", roc_auc_score(p2.actual, p2.prob),
                         model_null(panel, N_MODEL_NULL_SMALL, 12, ycol="y_crash_h2", tycol="target_year_h2", years=ew.H2_YEARS),
                         "within-block shuffle", len(p2), p2.actual.mean()))
    la = lobo(s, earlier_only=False)
    le = lobo(s, earlier_only=True)
    la.to_csv(OUT / "pred_lobo_allseasons.csv", index=False)
    le.to_csv(OUT / "pred_lobo_earlier.csv", index=False)
    log("model null, leave-one-block-out")
    rows.append(null_row("leave-one-block-out, all seasons in training (published design)", roc_auc_score(la.actual, la.prob),
                         model_null(panel, N_MODEL_NULL_SMALL, 13, protocol="lobo_all"), "within-block shuffle", len(la), la.actual.mean(),
                         "uses seasons later than the test season"))
    rows.append(null_row("leave-one-block-out, earlier seasons only", roc_auc_score(le.actual, le.prob),
                         model_null(panel, N_MODEL_NULL_SMALL, 14, protocol="lobo_earlier"), "within-block shuffle", len(le), le.actual.mean()))
    iall = index_samples(panel, test_from=2009)
    rows.append(null_row("index, all 170 samples (what leave-one-block-out reduces to for a rule without training)",
                         auc_or_nan(iall.label, iall["index"]), index_null(panel, "shuffle", N_INDEX_NULL, test_from=2009),
                         "within-block shuffle", len(iall), iall.label.mean()))
    # 2024-25 holdout
    trh, teh = s[s.target_year <= 2023], s[s.target_year.isin([2024, 2025])]
    c_h, pen = ew.tune_logreg(trh, "y_crash", "target_year")
    mh = ew.make_logreg(c_h, pen).fit(trh[ew.FEATURES].to_numpy(), trh.y_crash.to_numpy())
    ph = mh.predict_proba(teh[ew.FEATURES].to_numpy())[:, 1]
    mhp = ew.make_logreg(*modal).fit(trh[ew.FEATURES].to_numpy(), trh.y_crash.to_numpy())
    php = mhp.predict_proba(teh[ew.FEATURES].to_numpy())[:, 1]
    ihold = index_samples(panel, test_from=2024)
    hold = pd.DataFrame([{"analysis": "2024-25 holdout, published model settings", "n": len(teh), "crashes": int(teh.y_crash.sum()),
                          "auc": roc_auc_score(teh.y_crash, php)},
                         {"analysis": "2024-25 holdout, tuned on seasons <= 2023", "n": len(teh), "crashes": int(teh.y_crash.sum()),
                          "auc": roc_auc_score(teh.y_crash, ph)},
                         {"analysis": "2024-25 holdout, index", "n": len(ihold), "crashes": int(ihold.label.sum()),
                          "auc": auc_or_nan(ihold.label, ihold["index"])}])
    rng = np.random.default_rng(42)
    for i, sc in enumerate([php, ph, ihold["index"].to_numpy()]):
        yv = teh.y_crash.to_numpy() if i < 2 else ihold.label.to_numpy()
        bs = []
        for _ in range(B_BOOT):
            k = rng.integers(0, len(yv), len(yv))
            if 0 < yv[k].sum() < len(k):
                bs.append(roc_auc_score(yv[k], sc[k]))
        hold.loc[i, "ci_lo"], hold.loc[i, "ci_hi"] = np.quantile(bs, .025), np.quantile(bs, .975)
    save(hold, "T05_holdout_2024_25")
    hn = index_null(panel, "shuffle", N_INDEX_NULL, test_from=2024)
    rows.append(null_row("index, 2024-25 holdout", auc_or_nan(ihold.label, ihold["index"]), hn, "within-block shuffle", len(ihold), ihold.label.mean()))
    save(pd.DataFrame(rows), "T03_nulls")

    # --- alternative crash definitions
    rows = []
    for lab, ycol, name in (("cur", "y_crash", "published: Y_t+1 <= 0.8 x Y_t"), ("clim", "y_crash_clim", "Y_t+1 <= 0.8 x expanding mean at t"),
                            ("below", "y_below", "below-normal season: Y_t+1 < expanding mean at t")):
        d = index_samples(panel, label=lab)
        nl = index_null(panel, "shuffle", N_INDEX_NULL, label=lab)
        pl, _ = ew.rolling_logreg(s, ycol)
        rows.append({"definition": name, "n_test": len(d), "prevalence_test": d.label.mean(), "prevalence_all": s[ycol].mean(),
                     "index_auc": auc_or_nan(d.label, d["index"]), "index_null_mean": np.nanmean(nl),
                     "index_null_q025": np.nanquantile(nl, .025), "index_null_q975": np.nanquantile(nl, .975),
                     "model_auc": roc_auc_score(pl.actual, pl.prob)})
    save(pd.DataFrame(rows), "T04_alternative_definitions")

    # --- threshold sensitivity
    rows = []
    for th in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = index_samples(panel, theta=th)
        nl = index_null(panel, "shuffle", N_INDEX_NULL, theta=th)
        sa = ew.add_labels(s, th)
        rows.append({"theta": th, "prevalence_all": sa.y_crash.mean(), "prevalence_test": d.label.mean(),
                     "index_auc": auc_or_nan(d.label, d["index"]), "null_mean": np.nanmean(nl), "null_q975": np.nanquantile(nl, .975),
                     "share_null_ge_observed": float(np.mean(nl >= auc_or_nan(d.label, d["index"]) - 1e-12))})
    save(pd.DataFrame(rows), "T06_threshold_sensitivity")

    # --- filled block-years excluded
    so = ew.build_samples(panel_obs)
    po = so[so.target_year.isin(ew.TEST_YEARS)]
    pfo, _ = ew.rolling_logreg(so)
    io = index_samples(panel_obs)
    nlo = index_null(panel_obs, "shuffle", N_INDEX_NULL)
    save(pd.DataFrame([{"analysis": "18 filled block-years excluded", "usable_samples": len(so), "test_predictions": len(po),
                        "prevalence_test": po.y_crash.mean(), "index_auc": auc_or_nan(io.label, io["index"]),
                        "index_null_mean": nlo.mean(), "index_null_q975": np.quantile(nlo, .975),
                        "model_auc": roc_auc_score(pfo.actual, pfo.prob), "model_pr_auc": average_precision_score(pfo.actual, pfo.prob)}]),
         "T07_filled_excluded")

    # --- cluster bootstrap and the AUC difference
    d = pf.merge(pool[[ew.INDEX]], left_index=True, right_index=True)
    d = d.rename(columns={ew.INDEX: "idx"}).merge(pm[["prob"]].rename(columns={"prob": "prob_pub"}), left_index=True, right_index=True)
    rows = []
    for cl in ("Block", "target_year", None):
        if cl is None:
            dd_ = d.reset_index(drop=True).copy()
            dd_["row"] = np.arange(len(dd_))
            draws = cluster_boot(dd_, ["idx", "prob", "prob_pub"], "row", B_BOOT)
            name = "independent rows (published method)"
        else:
            draws = cluster_boot(d, ["idx", "prob", "prob_pub"], cl, B_BOOT)
            name = "blocks (10 clusters)" if cl == "Block" else "seasons (11 clusters)"
        for j, lab in enumerate(["index", "model tuned per fold", "published model"]):
            rows.append({"resampling_unit": name, "quantity": f"{lab} ROC-AUC", "estimate": roc_auc_score(d.actual, d[["idx", "prob", "prob_pub"][j]]),
                         "ci_lo": np.quantile(draws[:, j], .025), "ci_hi": np.quantile(draws[:, j], .975), "valid_draws": len(draws)})
        diff = draws[:, 0] - draws[:, 1]
        rows.append({"resampling_unit": name, "quantity": "AUC difference, index minus model tuned per fold",
                     "estimate": roc_auc_score(d.actual, d.idx) - roc_auc_score(d.actual, d.prob),
                     "ci_lo": np.quantile(diff, .025), "ci_hi": np.quantile(diff, .975), "valid_draws": len(draws)})
    save(pd.DataFrame(rows), "T08_cluster_bootstrap")

    # --- per-season AUC
    rows = []
    for Y, g in d.groupby("target_year"):
        rows.append({"target_season": Y, "n": len(g), "crashes": int(g.actual.sum()), "index_auc": auc_or_nan(g.actual, g.idx),
                     "model_auc": auc_or_nan(g.actual, g.prob)})
    save(pd.DataFrame(rows), "T09_per_season")

    # --- risk tiers with Wilson intervals
    rows = []
    pub = pd.read_csv(ew.DATA / "original_submission" / "T13c_tiered_score.csv")
    for r in pub.itertuples():
        k = int(round(r.observed_crash_rate * r.n))
        lo, hi = proportion_confint(k, r.n, method="wilson")
        rows.append({"probabilities": "published (calibrated model)", "tier": r.Tier, "n": r.n, "crashes": k, "mean_predicted": r.mean_prob,
                     "observed_rate": k / r.n, "wilson_lo": lo, "wilson_hi": hi})
    tiers = pd.cut(d.prob, [-np.inf, 0.33, 0.66, np.inf], labels=["Low", "Medium", "High"])
    for t in ["Low", "Medium", "High"]:
        g = d[tiers == t]
        if len(g):
            lo, hi = proportion_confint(int(g.actual.sum()), len(g), method="wilson")
            rows.append({"probabilities": "uncalibrated model tuned per fold", "tier": t, "n": len(g), "crashes": int(g.actual.sum()),
                         "mean_predicted": g.prob.mean(), "observed_rate": g.actual.mean(), "wilson_lo": lo, "wilson_hi": hi})
    save(pd.DataFrame(rows), "T10_risk_tiers")

    # --- decision analysis on uncalibrated probabilities, several cost ratios
    rows = []
    prev = d.actual.mean()
    for pt in np.round(np.arange(0.05, 0.601, 0.05), 2):
        yhat = d.prob >= pt
        tp, fp = int((yhat & (d.actual == 1)).sum()), int((yhat & (d.actual == 0)).sum())
        w = pt / (1 - pt)
        rows.append({"threshold": pt, "net_benefit_model": tp / len(d) - fp / len(d) * w, "net_benefit_warn_all": prev - (1 - prev) * w})
    nb = pd.DataFrame(rows)
    for ratio in (1, 2, 5, 10):
        nb[f"best_threshold_cost_{ratio}to1"] = np.nan
    cr = []
    for ratio in (1, 2, 5, 10):
        best = min(((((d.prob >= t) == 0) & (d.actual == 1)).sum() * ratio + (((d.prob >= t) == 1) & (d.actual == 0)).sum(), t)
                   for t in np.round(np.arange(0.05, 0.951, 0.05), 2))
        cr.append({"cost_ratio_missed_to_false_alarm": f"{ratio}:1", "best_threshold": best[1], "expected_cost_per_sample": best[0] / len(d)})
    save(nb.drop(columns=[c for c in nb.columns if c.startswith("best_")]), "T11a_net_benefit_uncalibrated")
    save(pd.DataFrame(cr), "T11b_cost_ratios")

    # --- regression benchmark, climatology prespecified
    rp = ew.rolling_regression(s)
    mae_p = mean_absolute_error(pool.y_yield, pool.bl_pers_yield)
    rows = []
    rmse_clim = np.sqrt(mean_squared_error(pool.y_yield, pool.bl_clim_yield))
    for name, pr in list(rp.items()) + [("Baseline: climatology", pool.assign(pred=pool.bl_clim_yield, actual=pool.y_yield)),
                                        ("Baseline: persistence", pool.assign(pred=pool.bl_pers_yield, actual=pool.y_yield)),
                                        ("Baseline: per-block linear trend", pool.assign(pred=pool.bl_trend_yield, actual=pool.y_yield))]:
        rmse = np.sqrt(mean_squared_error(pr.actual, pr.pred))
        # block-cluster bootstrap of skill vs climatology
        j = pr[["pred", "actual"]].join(pool[["bl_clim_yield", "Block"]])
        rng = np.random.default_rng(42)
        blocks = j.Block.unique()
        sk = []
        for _ in range(B_BOOT):
            pick = rng.choice(blocks, len(blocks))
            jj = pd.concat([j[j.Block == b] for b in pick])
            sk.append(100 * (1 - np.sqrt(np.mean((jj.actual - jj.pred) ** 2)) / np.sqrt(np.mean((jj.actual - jj.bl_clim_yield) ** 2))))
        rows.append({"model": name, "RMSE": rmse, "MAE": mean_absolute_error(pr.actual, pr.pred), "R2": r2_score(pr.actual, pr.pred),
                     "rMAE_vs_persistence": mean_absolute_error(pr.actual, pr.pred) / mae_p,
                     "skill_vs_climatology_pct": 100 * (1 - rmse / rmse_clim),
                     "skill_ci_lo": np.quantile(sk, .025), "skill_ci_hi": np.quantile(sk, .975)})
    save(pd.DataFrame(rows), "T12_regression_benchmark")

    # --- variance decomposition (two-way fixed effects, no replication)
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    for label, ss in (("all samples, test pool", pool), ("filled excluded, test pool", po)):
        v = ss.assign(y_change=ss.y_yield - ss.Yield_t)
        for ycol in ("y_yield", "y_change"):
            m = smf.ols(f"{ycol} ~ C(Block) + C(target_year)", data=v).fit()
            a = anova_lm(m, typ=2)
            tot = a.sum_sq.sum()
            rows.append({"sample": label, "target": ycol, "n": len(v), "block_pct": 100 * a.sum_sq["C(Block)"] / tot,
                         "season_pct": 100 * a.sum_sq["C(target_year)"] / tot, "residual_pct": 100 * a.sum_sq["Residual"] / tot})
    save(pd.DataFrame(rows), "T13_variance_decomposition")

    # --- conformal intervals: widths and per-block coverage with n
    rows, per = [], []
    for alpha in (0.10, 0.20):
        recs = []
        for Y in ew.TEST_YEARS:
            tr, te = s[s.target_year <= Y - 1], s[s.target_year == Y]
            cy = int(tr.target_year.max())
            cal, prop = tr[tr.target_year == cy], tr[tr.target_year < cy]
            mdl = ew.make_reg_models()["Lasso"].fit(prop[ew.FEATURES].to_numpy(), prop.y_yield.to_numpy())
            res = np.sort(np.abs(cal.y_yield.to_numpy() - mdl.predict(cal[ew.FEATURES].to_numpy())))
            k = min(int(np.ceil((len(res) + 1) * (1 - alpha))), len(res))
            q = res[k - 1]
            pr = mdl.predict(te[ew.FEATURES].to_numpy())
            recs.append(pd.DataFrame({"Block": te.Block.to_numpy(), "covered": np.abs(te.y_yield.to_numpy() - pr) <= q, "width": 2 * q,
                                      "n_cal": len(res)}))
        c = pd.concat(recs)
        rows.append({"nominal": 1 - alpha, "empirical_coverage": c.covered.mean(), "mean_width_t_ha": c.width.mean(),
                     "median_yield_t_ha": pool.y_yield.median(), "calibration_set": "previous target season only (10 obs per fold)"})
        if alpha == 0.10:
            for b, g in c.groupby("Block"):
                per.append({"block": b, "n": len(g), "coverage_90": g.covered.mean(), "mean_width": g.width.mean()})
    save(pd.DataFrame(rows), "T14a_conformal")
    save(pd.DataFrame(per), "T14b_conformal_per_block")
    log("done")


if __name__ == "__main__":
    main()
