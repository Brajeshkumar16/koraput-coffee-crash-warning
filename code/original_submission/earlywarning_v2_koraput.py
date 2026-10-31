# -*- coding: utf-8 -*-
"""
V2: One-year-ahead (early-warning) prediction of coffee performance in Koraput district.
Strict SUPERSET of earlywarning_koraput.py (v1). Leakage-free rolling-origin evaluation.
Seed = 42 everywhere; fully deterministic.

Reads (READ ONLY):  D:/kotaput coffee/final_clustered_data.csv
Writes:             D:/kotaput coffee/earlywarning_v2_outputs/{tables,figures}/...
(v1 outputs in earlywarning_outputs/ are NOT touched.)

New vs v1:
  - 11 new leakage-safe features (climatology deviation, volatility, growth,
    decline-run, spatial common-shock YoY), all covered by the corruption test.
  - T9  tuned + calibrated classification (headline)
  - T10 bootstrap CIs + DeLong test vs naive single-feature baseline
  - T11 generalization stress tests (LOBO spatial transfer, last-2-years holdout)
  - T12 multi-horizon (h=1 vs h=2) early warning
  - T13 decision-curve / net-benefit + cost analysis + tiered score
  - T14 variance decomposition (predictability ceiling)
  - T15 per-block conformal coverage
  - T16 robustness (seed stability, extended theta sensitivity)
  - New figures F7..F13
"""

import os
import math
import warnings
from collections import Counter
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             f1_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             precision_score, recall_score, confusion_matrix,
                             roc_curve, precision_recall_curve, brier_score_loss)
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import norm

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------------- paths
BASE = "D:/kotaput coffee"
DATA_PATH = os.path.join(BASE, "final_clustered_data.csv")
OUT = os.path.join(BASE, "earlywarning_v2_outputs")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------- plotting style (colorblind-safe Okabe-Ito)
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_VERM   = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY   = "#7F7F7F"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.frameon": False,
})

# ----------------------------------------------------------------------------- load data
raw = pd.read_csv(DATA_PATH)
raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")] + ["Cluster"], errors="ignore")
raw["Year"] = raw["Year"].astype(int)
BLOCKS = sorted(raw["Block"].unique())
YEARS = sorted(raw["Year"].unique())
assert len(raw) == 200 and len(BLOCKS) == 10, "unexpected data shape"
assert not raw.duplicated(subset=["Block", "Year"]).any()

# fast lookup: (block, year) -> row dict
_lut = {(r["Block"], r["Year"]): r for _, r in raw.iterrows()}

def val(block, year, col):
    return float(_lut[(block, year)][col])

# ----------------------------------------------------------------------------- feature engineering (leakage-free)
V1_FEATURES = ["Yield_t", "Production_t", "Export_t", "Area_t", "Farmers_t",
               "ExportRatio_t", "Productivity_t",
               "Yield_lag1", "Yield_lag2", "Production_lag1", "Export_lag1",
               "Yield_yoy", "Yield_yoy_pct", "Yield_roll3_mean", "Yield_roll3_slope",
               "Farmer_to_Area", "Block_hist_mean", "Spatial_LOO_mean", "Year"]

NEW_FEATURES = ["Yield_dev_clim", "Yield_dev_clim_pct", "Yield_z_clim",
                "Yield_roll3_std", "Yield_cv3", "ExportRatio_yoy",
                "Area_growth", "Farmers_growth", "Decline_run",
                "Down_years_3", "Spatial_LOO_yoy"]

FEATURES = V1_FEATURES + NEW_FEATURES

GROUP_A = ["Yield_t", "Production_t", "Export_t", "Area_t", "Farmers_t",
           "ExportRatio_t", "Productivity_t", "Farmer_to_Area", "Year"]
GROUP_B = GROUP_A + ["Yield_lag1", "Yield_lag2", "Production_lag1", "Export_lag1",
                     "Yield_yoy", "Yield_yoy_pct", "Yield_roll3_mean", "Yield_roll3_slope"]
GROUP_C = GROUP_B + ["Block_hist_mean", "Spatial_LOO_mean"] + NEW_FEATURES  # = full set
assert sorted(GROUP_C) == sorted(FEATURES)

def build_features_one(df, block, t):
    """All engineered features for sample (block, t), computed ONLY from rows with Year <= t."""
    h = df[(df["Year"] <= t)]                       # explicit temporal restriction
    hb = h[h["Block"] == block].sort_values("Year")

    def v(year, col):
        s = hb.loc[hb["Year"] == year, col]
        return float(s.iloc[0]) if len(s) else np.nan

    f = {}
    f["Yield_t"]        = v(t, "Yield per Ha")
    f["Production_t"]   = v(t, "Production MT")
    f["Export_t"]       = v(t, "Export MT")
    f["Area_t"]         = v(t, "Area Ha")
    f["Farmers_t"]      = v(t, "Farmers")
    f["ExportRatio_t"]  = v(t, "Export Ratio")
    f["Productivity_t"] = v(t, "Productivity per Farmer")
    f["Yield_lag1"]      = v(t - 1, "Yield per Ha")
    f["Yield_lag2"]      = v(t - 2, "Yield per Ha")
    f["Production_lag1"] = v(t - 1, "Production MT")
    f["Export_lag1"]     = v(t - 1, "Export MT")
    f["Yield_yoy"]     = f["Yield_t"] - f["Yield_lag1"]
    f["Yield_yoy_pct"] = (f["Yield_t"] - f["Yield_lag1"]) / f["Yield_lag1"]
    win = np.array([f["Yield_lag2"], f["Yield_lag1"], f["Yield_t"]])
    f["Yield_roll3_mean"] = float(np.mean(win))
    f["Yield_roll3_slope"] = float(np.polyfit(np.array([t - 2, t - 1, t], dtype=float), win, 1)[0])
    f["Farmer_to_Area"] = f["Farmers_t"] / f["Area_t"]
    f["Block_hist_mean"] = float(hb["Yield per Ha"].mean())                     # expanding mean over <= t
    oth = h[(h["Year"] == t) & (h["Block"] != block)]
    f["Spatial_LOO_mean"] = float(oth["Yield per Ha"].mean())                   # other blocks, year t only
    f["Year"] = float(t)

    # ---------------- NEW v2 features (all from hb / h, i.e. Year <= t only) ----------------
    yb = hb["Yield per Ha"].to_numpy(dtype=float)                               # block's yields 2006..t
    bh_mean = f["Block_hist_mean"]
    bh_std = float(np.std(yb, ddof=0)) if len(yb) >= 2 else 0.0                 # population std, expanding
    f["Yield_dev_clim"]     = f["Yield_t"] - bh_mean
    f["Yield_dev_clim_pct"] = (f["Yield_t"] - bh_mean) / bh_mean
    f["Yield_z_clim"]       = (f["Yield_t"] - bh_mean) / bh_std if bh_std > 0 else 0.0
    f["Yield_roll3_std"]    = float(np.std(win, ddof=0))                        # std of {t-2, t-1, t}
    f["Yield_cv3"]          = f["Yield_roll3_std"] / f["Yield_roll3_mean"]
    f["ExportRatio_yoy"]    = v(t, "Export Ratio") - v(t - 1, "Export Ratio")
    f["Area_growth"]        = (v(t, "Area Ha") - v(t - 1, "Area Ha")) / v(t - 1, "Area Ha")
    f["Farmers_growth"]     = (v(t, "Farmers") - v(t - 1, "Farmers")) / v(t - 1, "Farmers")
    diffs = np.diff(yb)                                                          # YoY transitions ending at t
    run = 0
    for d in diffs[::-1]:
        if d < 0:
            run += 1
        else:
            break
    f["Decline_run"] = float(run)                                                # capped at available history
    last3 = diffs[-3:] if len(diffs) else np.array([])
    f["Down_years_3"] = float((last3 < 0).sum())                                 # capped at available transitions
    oth_t  = h[(h["Year"] == t)     & (h["Block"] != block)].set_index("Block")["Yield per Ha"]
    oth_t1 = h[(h["Year"] == t - 1) & (h["Block"] != block)].set_index("Block")["Yield per Ha"]
    f["Spatial_LOO_yoy"] = float((oth_t - oth_t1).mean())                        # other blocks' mean YoY, years t & t-1
    return f

def baseline_preds_one(df, block, t, col):
    """Persistence / climatology / per-block linear-trend baselines for target col at t+1, using Year<=t only."""
    hb = df[(df["Block"] == block) & (df["Year"] <= t)].sort_values("Year")
    pers = float(hb.loc[hb["Year"] == t, col].iloc[0])
    clim = float(hb[col].mean())
    yrs = hb["Year"].to_numpy(dtype=float)
    ys = hb[col].to_numpy(dtype=float)
    b1, b0 = np.polyfit(yrs, ys, 1)
    trend = float(b1 * (t + 1) + b0)
    return pers, clim, trend

rows = []
for b in BLOCKS:
    for t in range(2008, 2025):                      # base years 2008..2024 -> targets 2009..2025
        f = build_features_one(raw, b, t)
        if any(np.isnan(list(f.values()))):
            continue
        f["Block"] = b
        f["base_year"] = t
        f["target_year"] = t + 1
        f["y_yield"] = val(b, t + 1, "Yield per Ha")
        f["y_prod"] = val(b, t + 1, "Production MT")
        p, c, tr = baseline_preds_one(raw, b, t, "Yield per Ha")
        f["bl_pers_yield"], f["bl_clim_yield"], f["bl_trend_yield"] = p, c, tr
        p, c, tr = baseline_preds_one(raw, b, t, "Production MT")
        f["bl_pers_prod"], f["bl_clim_prod"], f["bl_trend_prod"] = p, c, tr
        rows.append(f)

samples = pd.DataFrame(rows).reset_index(drop=True)
N = len(samples)

def crash_labels(theta):
    return (samples["y_yield"] <= (1.0 - theta) * samples["Yield_t"]).astype(int)

samples["y_crash"] = crash_labels(0.20)

# horizon-2 targets: crash at t+2 relative to Yield_t (base years with t+2 <= 2025)
samples["y_yield_h2"] = np.nan
samples["target_year_h2"] = np.nan
m_h2 = samples["base_year"] <= 2023
samples.loc[m_h2, "y_yield_h2"] = [val(b, t + 2, "Yield per Ha")
                                   for b, t in zip(samples.loc[m_h2, "Block"],
                                                   samples.loc[m_h2, "base_year"])]
samples.loc[m_h2, "target_year_h2"] = samples.loc[m_h2, "base_year"] + 2
samples["y_crash_h2"] = np.where(m_h2,
                                 (samples["y_yield_h2"] <= 0.8 * samples["Yield_t"]).astype(float),
                                 np.nan)
samples_h2 = samples[m_h2].copy()
samples_h2["target_year_h2"] = samples_h2["target_year_h2"].astype(int)
samples_h2["y_crash_h2"] = samples_h2["y_crash_h2"].astype(int)

# ----------------------------------------------------------------------------- LEAKAGE UNIT TEST (extended to EVERY new feature)
def leakage_check():
    rng = np.random.default_rng(SEED)
    idx = list(rng.choice(len(samples), size=3, replace=False))
    idx.append(0)                                    # deterministic early sample (t=2008, capped history)
    for i in idx:
        s = samples.iloc[i]
        b, t = s["Block"], int(s["base_year"])
        # (1) hand recomputation from the raw dataframe restricted to Year <= t
        hist = raw[raw["Year"] <= t].copy()
        hb = hist[hist["Block"] == b].sort_values("Year")
        gv = lambda yr, c: float(hb.loc[hb["Year"] == yr, c].iloc[0])
        manual = {
            "Yield_t": gv(t, "Yield per Ha"), "Production_t": gv(t, "Production MT"),
            "Export_t": gv(t, "Export MT"), "Area_t": gv(t, "Area Ha"),
            "Farmers_t": gv(t, "Farmers"), "ExportRatio_t": gv(t, "Export Ratio"),
            "Productivity_t": gv(t, "Productivity per Farmer"),
            "Yield_lag1": gv(t - 1, "Yield per Ha"), "Yield_lag2": gv(t - 2, "Yield per Ha"),
            "Production_lag1": gv(t - 1, "Production MT"), "Export_lag1": gv(t - 1, "Export MT"),
        }
        manual["Yield_yoy"] = manual["Yield_t"] - manual["Yield_lag1"]
        manual["Yield_yoy_pct"] = manual["Yield_yoy"] / manual["Yield_lag1"]
        w = np.array([manual["Yield_lag2"], manual["Yield_lag1"], manual["Yield_t"]])
        manual["Yield_roll3_mean"] = w.mean()
        manual["Yield_roll3_slope"] = np.polyfit([t - 2.0, t - 1.0, float(t)], w, 1)[0]
        manual["Farmer_to_Area"] = manual["Farmers_t"] / manual["Area_t"]
        manual["Block_hist_mean"] = float(hb["Yield per Ha"].mean())
        manual["Spatial_LOO_mean"] = float(
            hist[(hist["Year"] == t) & (hist["Block"] != b)]["Yield per Ha"].mean())
        manual["Year"] = float(t)
        # ---- new v2 features, hand-recomputed on Year <= t only ----
        yb = hb["Yield per Ha"].to_numpy(dtype=float)
        bh_mean = float(yb.mean())
        bh_std = float(np.std(yb, ddof=0)) if len(yb) >= 2 else 0.0
        manual["Yield_dev_clim"] = manual["Yield_t"] - bh_mean
        manual["Yield_dev_clim_pct"] = (manual["Yield_t"] - bh_mean) / bh_mean
        manual["Yield_z_clim"] = (manual["Yield_t"] - bh_mean) / bh_std if bh_std > 0 else 0.0
        manual["Yield_roll3_std"] = float(np.std(w, ddof=0))
        manual["Yield_cv3"] = manual["Yield_roll3_std"] / manual["Yield_roll3_mean"]
        manual["ExportRatio_yoy"] = gv(t, "Export Ratio") - gv(t - 1, "Export Ratio")
        manual["Area_growth"] = (gv(t, "Area Ha") - gv(t - 1, "Area Ha")) / gv(t - 1, "Area Ha")
        manual["Farmers_growth"] = (gv(t, "Farmers") - gv(t - 1, "Farmers")) / gv(t - 1, "Farmers")
        diffs = np.diff(yb)
        run = 0
        for d in diffs[::-1]:
            if d < 0:
                run += 1
            else:
                break
        manual["Decline_run"] = float(run)
        last3 = diffs[-3:] if len(diffs) else np.array([])
        manual["Down_years_3"] = float((last3 < 0).sum())
        ot = hist[(hist["Year"] == t) & (hist["Block"] != b)].set_index("Block")["Yield per Ha"]
        o1 = hist[(hist["Year"] == t - 1) & (hist["Block"] != b)].set_index("Block")["Yield per Ha"]
        manual["Spatial_LOO_yoy"] = float((ot - o1).mean())
        assert sorted(manual.keys()) == sorted(FEATURES), "manual recomputation must cover every feature"
        for k, mv in manual.items():
            assert abs(mv - float(s[k])) < 1e-6, f"mismatch {k} for ({b},{t}): {mv} vs {s[k]}"
        # (2) corruption test: garbage in every row with Year > t must leave EVERY feature unchanged
        corrupt = raw.copy()
        num_cols = [c for c in corrupt.columns if c not in ("Year", "Block")]
        corrupt.loc[corrupt["Year"] > t, num_cols] = -9.999e9
        f2 = build_features_one(corrupt, b, t)
        for k in FEATURES:
            assert abs(float(f2[k]) - float(s[k])) < 1e-6, f"feature {k} reads Year>t for ({b},{t})"
    print(f"LEAKAGE CHECK: PASS  ({len(FEATURES)} features incl. {len(NEW_FEATURES)} new; "
          f"corruption + hand-recompute on {len(idx)} samples)")

leakage_check()

# ----------------------------------------------------------------------------- rolling-origin machinery
TEST_YEARS = list(range(2015, 2026))                 # 11 folds, target years 2015..2025

def make_reg_models():
    return {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=SEED))]),
        "Lasso": Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.1, random_state=SEED, max_iter=50000))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=None,
                                              min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=400,
                                                min_samples_leaf=5, l2_regularization=1.0,
                                                early_stopping=False, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                random_state=SEED, n_jobs=-1, verbosity=0),
    }

def rolling_origin_regression(feat_cols, target_col, model_factory):
    preds = {name: [] for name in model_factory().keys()}
    for Y in TEST_YEARS:
        tr = samples[samples["target_year"] <= Y - 1]
        te = samples[samples["target_year"] == Y]
        Xtr, ytr = tr[feat_cols].to_numpy(), tr[target_col].to_numpy()
        Xte = te[feat_cols].to_numpy()
        for name, mdl in model_factory().items():
            mdl.fit(Xtr, ytr)
            p = mdl.predict(Xte)
            preds[name].append(pd.DataFrame({
                "pred": p, "actual": te[target_col].to_numpy(),
                "target_year": te["target_year"].to_numpy(), "Block": te["Block"].to_numpy(),
            }, index=te.index))
    return {k: pd.concat(v) for k, v in preds.items()}

def reg_metrics(actual, pred, mae_pers):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    mase = mae / mae_pers
    return rmse, mae, r2, mase

test_mask = samples["target_year"].isin(TEST_YEARS)
pool = samples[test_mask]

def baseline_table(target_col, tag):
    out = {}
    y = pool[target_col].to_numpy()
    mae_pers = float(mean_absolute_error(y, pool[f"bl_pers_{tag}"]))
    for bname, col in [("Persistence", f"bl_pers_{tag}"), ("Climatology", f"bl_clim_{tag}"),
                       ("LinearTrend", f"bl_trend_{tag}")]:
        rmse, mae, r2, mase = reg_metrics(y, pool[col].to_numpy(), mae_pers)
        out[bname] = dict(RMSE=rmse, MAE=mae, R2=r2, MASE=mase)
    return out, mae_pers

bl_yield, mae_pers_yield = baseline_table("y_yield", "yield")
bl_prod, mae_pers_prod = baseline_table("y_prod", "prod")
best_bl_yield_name = min(bl_yield, key=lambda k: bl_yield[k]["RMSE"])
best_bl_yield_rmse = bl_yield[best_bl_yield_name]["RMSE"]
best_bl_prod_name = min(bl_prod, key=lambda k: bl_prod[k]["RMSE"])
best_bl_prod_rmse = bl_prod[best_bl_prod_name]["RMSE"]

# ---- T2: regression on Yield (full features)
reg_preds_yield = rolling_origin_regression(FEATURES, "y_yield", make_reg_models)
t2_rows = []
for name, df in reg_preds_yield.items():
    rmse, mae, r2, mase = reg_metrics(df["actual"], df["pred"], mae_pers_yield)
    skill = 100.0 * (1.0 - rmse / best_bl_yield_rmse)
    t2_rows.append(dict(Model=name, RMSE=rmse, MAE=mae, R2=r2, MASE=mase, Skill_pct=skill))
for bname, m in bl_yield.items():
    skill = 100.0 * (1.0 - m["RMSE"] / best_bl_yield_rmse)
    t2_rows.append(dict(Model=f"Baseline: {bname}", RMSE=m["RMSE"], MAE=m["MAE"],
                        R2=m["R2"], MASE=m["MASE"], Skill_pct=skill))
T2 = pd.DataFrame(t2_rows)
best_reg_name = T2[~T2["Model"].str.startswith("Baseline")].sort_values("RMSE")["Model"].iloc[0]
best_reg_pred = reg_preds_yield[best_reg_name]
best_reg_rmse = float(T2.loc[T2["Model"] == best_reg_name, "RMSE"].iloc[0])

# ---- T2b: regression on Production MT
reg_preds_prod = rolling_origin_regression(FEATURES, "y_prod", make_reg_models)
t2b_rows = []
for name, df in reg_preds_prod.items():
    rmse, mae, r2, mase = reg_metrics(df["actual"], df["pred"], mae_pers_prod)
    skill = 100.0 * (1.0 - rmse / best_bl_prod_rmse)
    t2b_rows.append(dict(Model=name, RMSE=rmse, MAE=mae, R2=r2, MASE=mase, Skill_pct=skill))
for bname, m in bl_prod.items():
    skill = 100.0 * (1.0 - m["RMSE"] / best_bl_prod_rmse)
    t2b_rows.append(dict(Model=f"Baseline: {bname}", RMSE=m["RMSE"], MAE=m["MAE"],
                         R2=m["R2"], MASE=m["MASE"], Skill_pct=skill))
T2b = pd.DataFrame(t2b_rows)
best_prod_name = T2b[~T2b["Model"].str.startswith("Baseline")].sort_values("RMSE")["Model"].iloc[0]

# ---- T3
t3_rows = []
for bname, m in bl_yield.items():
    t3_rows.append(dict(Baseline=bname, RMSE=m["RMSE"], MAE=m["MAE"], MASE=m["MASE"],
                        BestModel_skill_vs_this_pct=100.0 * (1.0 - best_reg_rmse / m["RMSE"])))
T3 = pd.DataFrame(t3_rows)

# ----------------------------------------------------------------------------- classification building blocks
def make_logreg_pipe(C=1.0, penalty="l2", seed=SEED):
    solver = "saga" if penalty == "l1" else "lbfgs"
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=C, penalty=penalty, solver=solver,
                                              class_weight="balanced", max_iter=20000,
                                              random_state=seed))])

def make_rf_clf(seed=SEED):
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                  class_weight="balanced", random_state=seed, n_jobs=-1)

def make_hgb_clf(seed=SEED):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=400,
                                          min_samples_leaf=5, l2_regularization=1.0,
                                          class_weight="balanced", early_stopping=False,
                                          random_state=seed)

def make_xgb_clf(spw, seed=SEED):
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                         scale_pos_weight=spw, random_state=seed, n_jobs=-1,
                         verbosity=0, eval_metric="logloss")

def make_clf_models(spw):
    """v1's default (untuned) classifier zoo, unchanged."""
    return {
        "LogisticRegression": Pipeline([("sc", StandardScaler()),
                                        ("m", LogisticRegression(class_weight="balanced",
                                                                 max_iter=2000, random_state=SEED))]),
        "RandomForest": make_rf_clf(SEED),
        "HistGB": make_hgb_clf(SEED),
        "XGBoost": make_xgb_clf(spw, SEED),
    }

def rolling_origin_classification(ycol):
    names = list(make_clf_models(1.0).keys())
    preds = {n: [] for n in names}
    for Y in TEST_YEARS:
        tr = samples[samples["target_year"] <= Y - 1]
        te = samples[samples["target_year"] == Y]
        Xtr, ytr = tr[FEATURES].to_numpy(), tr[ycol].to_numpy()
        Xte = te[FEATURES].to_numpy()
        npos = int(ytr.sum())
        spw = (len(ytr) - npos) / max(npos, 1)
        for name, mdl in make_clf_models(spw).items():
            mdl.fit(Xtr, ytr)
            prob = mdl.predict_proba(Xte)[:, 1]
            preds[name].append(pd.DataFrame({
                "prob": prob, "actual": te[ycol].to_numpy(),
                "target_year": te["target_year"].to_numpy(), "Block": te["Block"].to_numpy(),
            }, index=te.index))
    return {k: pd.concat(v) for k, v in preds.items()}

def clf_metrics(df):
    y, p = df["actual"].to_numpy(), df["prob"].to_numpy()
    yhat = (p >= 0.5).astype(int)
    return dict(
        F1=f1_score(y, yhat, zero_division=0),
        BalAcc=balanced_accuracy_score(y, yhat),
        ROC_AUC=roc_auc_score(y, p),
        PR_AUC=average_precision_score(y, p),
        MCC=matthews_corrcoef(y, yhat),
        Precision=precision_score(y, yhat, zero_division=0),
        Recall=recall_score(y, yhat, zero_division=0),
    )

# ---- T5: v1-style (untuned, uncalibrated) classification
clf_preds = rolling_origin_classification("y_crash")
t5_rows = [dict(Model=n, **clf_metrics(df)) for n, df in clf_preds.items()]
T5 = pd.DataFrame(t5_rows)
best_clf_name = T5.sort_values(["F1", "PR_AUC"], ascending=False)["Model"].iloc[0]
best_clf_pred = clf_preds[best_clf_name]
prevalence_all = float(samples["y_crash"].mean())
prevalence_test = float(pool["y_crash"].mean())

# ---- T5b: v1-style theta sensitivity
theta_rows = []
for th in (0.15, 0.20, 0.25):
    ycol = f"y_crash_{int(th*100)}"
    samples[ycol] = crash_labels(th)
    parts = []
    for Y in TEST_YEARS:
        tr = samples[samples["target_year"] <= Y - 1]
        te = samples[samples["target_year"] == Y]
        ytr = tr[ycol].to_numpy()
        npos = int(ytr.sum())
        spw = (len(ytr) - npos) / max(npos, 1)
        mdl = make_clf_models(spw)[best_clf_name]
        mdl.fit(tr[FEATURES].to_numpy(), ytr)
        prob = mdl.predict_proba(te[FEATURES].to_numpy())[:, 1]
        parts.append(pd.DataFrame({"prob": prob, "actual": te[ycol].to_numpy()}))
    dfp = pd.concat(parts)
    m = clf_metrics(dfp)
    theta_rows.append(dict(theta=th, prevalence_all=float(samples[ycol].mean()),
                           prevalence_test=float(samples.loc[test_mask, ycol].mean()),
                           F1=m["F1"], ROC_AUC=m["ROC_AUC"], PR_AUC=m["PR_AUC"]))
T5_theta = pd.DataFrame(theta_rows)

# ----------------------------------------------------------------------------- conformal intervals (best regression model)
def conformal_run(alpha):
    recs = []
    for Y in TEST_YEARS:
        tr = samples[samples["target_year"] <= Y - 1]
        te = samples[samples["target_year"] == Y]
        cal_year = int(tr["target_year"].max())
        cal = tr[tr["target_year"] == cal_year]
        prop = tr[tr["target_year"] < cal_year]
        if len(cal) < 5 or len(prop) < 20:
            rng = np.random.default_rng(SEED)
            idx = rng.permutation(len(tr))
            ncal = max(int(round(0.2 * len(tr))), 5)
            cal, prop = tr.iloc[idx[:ncal]], tr.iloc[idx[ncal:]]
        mdl = make_reg_models()[best_reg_name]
        mdl.fit(prop[FEATURES].to_numpy(), prop["y_yield"].to_numpy())
        res = np.abs(cal["y_yield"].to_numpy() - mdl.predict(cal[FEATURES].to_numpy()))
        n = len(res)
        k = min(int(math.ceil((n + 1) * (1 - alpha))), n)
        q = float(np.sort(res)[k - 1])
        p = mdl.predict(te[FEATURES].to_numpy())
        recs.append(pd.DataFrame({
            "pred": p, "lo": p - q, "hi": p + q, "actual": te["y_yield"].to_numpy(),
            "target_year": te["target_year"].to_numpy(), "Block": te["Block"].to_numpy(),
        }, index=te.index))
    df = pd.concat(recs)
    cov = float(((df["actual"] >= df["lo"]) & (df["actual"] <= df["hi"])).mean())
    width = float((df["hi"] - df["lo"]).mean())
    return df, cov, width

conf90_df, cov90, width90 = conformal_run(0.10)
conf80_df, cov80, width80 = conformal_run(0.20)
T6 = pd.DataFrame([
    dict(Nominal_coverage=0.90, Empirical_coverage=cov90, Mean_interval_width=width90),
    dict(Nominal_coverage=0.80, Empirical_coverage=cov80, Mean_interval_width=width80),
])

# ----------------------------------------------------------------------------- ablation A/B/C
ab_defs = [("A: current-year only", GROUP_A),
           ("B: A + lags/trend/momentum", GROUP_B),
           ("C: B + block-history + spatial + v2 (full)", GROUP_C)]
t4_rows = []
for label, cols in ab_defs:
    dfp = rolling_origin_regression(cols, "y_yield", make_reg_models)[best_reg_name]
    rmse, mae, r2, mase = reg_metrics(dfp["actual"], dfp["pred"], mae_pers_yield)
    skill = 100.0 * (1.0 - rmse / best_bl_yield_rmse)
    t4_rows.append(dict(FeatureSet=label, n_features=len(cols), RMSE=rmse, MAE=mae, R2=r2, Skill_pct=skill))
T4 = pd.DataFrame(t4_rows)

# ----------------------------------------------------------------------------- interpretability (train on target_year <= 2024)
tr_int = samples[samples["target_year"] <= 2024]
X_int = tr_int[FEATURES].to_numpy()

def importance_for(model, X, y, is_tree):
    model.fit(X, y)
    pi = permutation_importance(model, X, y, n_repeats=30, random_state=SEED, n_jobs=-1)
    perm = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    shap_s, shap_note = None, ""
    try:
        import shap
        est = model.named_steps["m"] if isinstance(model, Pipeline) else model
        if is_tree:
            ex = shap.TreeExplainer(est)
            sv = ex.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            Xs = model.named_steps["sc"].transform(X)
            ex = shap.LinearExplainer(est, Xs)
            sv = ex.shap_values(Xs)
        shap_s = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    except Exception as e:
        shap_note = f"SHAP failed ({type(e).__name__}: {e}); permutation importance only."
    return perm, shap_s, shap_note

TREE_NAMES = {"RandomForest", "HistGB", "XGBoost"}
reg_model_int = make_reg_models()[best_reg_name]
perm_reg, shap_reg, note_reg = importance_for(reg_model_int, X_int, tr_int["y_yield"].to_numpy(),
                                              best_reg_name in TREE_NAMES)
ytr_c = tr_int["y_crash"].to_numpy()
spw_int = (len(ytr_c) - ytr_c.sum()) / max(ytr_c.sum(), 1)
clf_model_int = make_clf_models(spw_int)[best_clf_name]
perm_clf, shap_clf, note_clf = importance_for(clf_model_int, X_int, ytr_c,
                                              best_clf_name in TREE_NAMES)
rank_reg = shap_reg if shap_reg is not None else perm_reg
rank_clf = shap_clf if shap_clf is not None else perm_clf

# ----------------------------------------------------------------------------- leakage contrast: random 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
kp = np.zeros(N)
Xall, yall = samples[FEATURES].to_numpy(), samples["y_yield"].to_numpy()
for tr_i, te_i in kf.split(Xall):
    m = make_reg_models()[best_reg_name]
    m.fit(Xall[tr_i], yall[tr_i])
    kp[te_i] = m.predict(Xall[te_i])
kf_rmse = float(np.sqrt(mean_squared_error(yall, kp)))
kf_r2 = float(r2_score(yall, kp))
ro_r2 = float(T2.loc[T2["Model"] == best_reg_name, "R2"].iloc[0])
T8 = pd.DataFrame([
    dict(Protocol="Rolling-origin (leakage-free)", R2=ro_r2, RMSE=best_reg_rmse),
    dict(Protocol="Random 5-fold KFold (naive)", R2=kf_r2, RMSE=kf_rmse),
])

# ----------------------------------------------------------------------------- breakdowns (T7)
bp = best_reg_pred.copy()
bp["abs_err"] = (bp["actual"] - bp["pred"]).abs()
bp["sq_err"] = (bp["actual"] - bp["pred"]) ** 2
per_year = bp.groupby("target_year").agg(RMSE=("sq_err", lambda s: float(np.sqrt(s.mean()))),
                                         MAE=("abs_err", "mean")).reset_index()
per_block = bp.groupby("Block").agg(RMSE=("sq_err", lambda s: float(np.sqrt(s.mean()))),
                                    MAE=("abs_err", "mean")).reset_index()
pp = pool.copy()
pp["sq_err_pers"] = (pp["y_yield"] - pp["bl_pers_yield"]) ** 2
per_year_pers = pp.groupby("target_year")["sq_err_pers"].mean().pow(0.5).reset_index(name="RMSE_pers")
cr_rows = []
for Y, g in best_clf_pred.groupby("target_year"):
    npos = int(g["actual"].sum())
    rec = recall_score(g["actual"], (g["prob"] >= 0.5).astype(int), zero_division=0) if npos else np.nan
    cr_rows.append(dict(target_year=int(Y), n_crashes=npos,
                        crash_recall=(rec if npos else np.nan)))
per_year_crash = pd.DataFrame(cr_rows)
T7_year = per_year.merge(per_year_pers, on="target_year").merge(per_year_crash, on="target_year")
T7_block = per_block

# ----------------------------------------------------------------------------- T1 feature dictionary (v1 + new)
_desc = {
    "Yield_t": ("Yield per Ha in base year t", "current-year value (year t)"),
    "Production_t": ("Production MT in year t", "current-year value"),
    "Export_t": ("Export MT in year t", "current-year value"),
    "Area_t": ("Area Ha in year t", "current-year value"),
    "Farmers_t": ("Number of farmers in year t", "current-year value"),
    "ExportRatio_t": ("Export Ratio in year t", "current-year value"),
    "Productivity_t": ("Productivity per Farmer in year t", "current-year value"),
    "Yield_lag1": ("Yield per Ha in year t-1", "lag, uses year t-1 only"),
    "Yield_lag2": ("Yield per Ha in year t-2", "lag, uses year t-2 only"),
    "Production_lag1": ("Production MT in year t-1", "lag"),
    "Export_lag1": ("Export MT in year t-1", "lag"),
    "Yield_yoy": ("Yield_t - Yield_{t-1} (absolute YoY change)", "uses years t, t-1"),
    "Yield_yoy_pct": ("(Yield_t - Yield_{t-1}) / Yield_{t-1}", "uses years t, t-1"),
    "Yield_roll3_mean": ("Mean Yield over years {t-2, t-1, t}", "3-year window ending at t"),
    "Yield_roll3_slope": ("OLS slope of Yield over years {t-2, t-1, t}", "3-year window ending at t"),
    "Farmer_to_Area": ("Farmers_t / Area_t", "current-year ratio"),
    "Block_hist_mean": ("Block's expanding mean Yield over 2006..t (fixed-effect proxy)",
                        "expanding window, never beyond t"),
    "Spatial_LOO_mean": ("Leave-one-out mean Yield of the other 9 blocks in year t",
                         "same-year spatial context, year t only, excludes own block"),
    "Year": ("Base year t as integer trend", "temporal index"),
    # ---- new v2 features ----
    "Yield_dev_clim": ("Yield_t - Block_hist_mean (deviation from own expanding climatology; "
                       "mean-reversion signal)", "expanding window 2006..t only"),
    "Yield_dev_clim_pct": ("(Yield_t - Block_hist_mean) / Block_hist_mean (relative deviation "
                           "from own climatology)", "expanding window 2006..t only"),
    "Yield_z_clim": ("(Yield_t - Block_hist_mean) / Block_hist_std; expanding population std over "
                     "2006..t; 0.0 if std==0 or <2 obs", "expanding window 2006..t only"),
    "Yield_roll3_std": ("Population std of Yield over years {t-2, t-1, t} (recent volatility)",
                        "3-year window ending at t"),
    "Yield_cv3": ("Yield_roll3_std / Yield_roll3_mean (recent coefficient of variation)",
                  "3-year window ending at t"),
    "ExportRatio_yoy": ("ExportRatio_t - ExportRatio_{t-1}", "uses years t, t-1"),
    "Area_growth": ("(Area_t - Area_{t-1}) / Area_{t-1}", "uses years t, t-1"),
    "Farmers_growth": ("(Farmers_t - Farmers_{t-1}) / Farmers_{t-1}", "uses years t, t-1"),
    "Decline_run": ("Consecutive strict YoY yield declines ending at t (walk back from t; "
                    "capped at available history 2006..t)", "backward walk within 2006..t only"),
    "Down_years_3": ("Count of yield declines among the last 3 (or fewer, if history is shorter) "
                     "year-to-year transitions ending at t", "backward window within 2006..t only"),
    "Spatial_LOO_yoy": ("Mean over the OTHER 9 blocks of (their Yield_t - their Yield_{t-1}); "
                        "common-shock proxy", "years t and t-1 only, excludes own block"),
}
assert sorted(_desc.keys()) == sorted(FEATURES)
T1 = pd.DataFrame([dict(Feature=k, Description=v[0], Leakage_note=v[1]) for k, v in _desc.items()])

# =============================================================================
# ===================== NEW V2 ANALYSES =======================================
# =============================================================================

# --------------------------------------------------------------- prefit calibration helper
def calib_prefit(fitted_model, Xcal, ycal, method):
    """Calibrate an already-fitted model on a held-out calibration set."""
    try:
        from sklearn.frozen import FrozenEstimator
        cc = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    except ImportError:
        cc = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    cc.fit(Xcal, ycal)
    return cc

# --------------------------------------------------------------- LogReg inner tuning (training years only)
LOGREG_GRID = [(C, p) for p in ("l1", "l2") for C in (0.03, 0.1, 0.3, 1.0, 3.0)]

def tune_logreg(train_df, ycol="y_crash", tycol="target_year"):
    """Inner rolling-origin tuning on the last (up to) 3 training target years.
    Never touches any year outside train_df."""
    yrs = sorted(train_df[tycol].unique())
    inner_val = [y for y in yrs if y >= yrs[-1] - 2]
    best, best_score = (1.0, "l2"), -np.inf
    for C, p in LOGREG_GRID:
        scores = []
        for Yi in inner_val:
            itr = train_df[train_df[tycol] <= Yi - 1]
            iva = train_df[train_df[tycol] == Yi]
            if itr[ycol].nunique() < 2 or iva[ycol].nunique() < 2 or len(itr) < 20:
                continue
            mdl = make_logreg_pipe(C, p, SEED)
            mdl.fit(itr[FEATURES].to_numpy(), itr[ycol].to_numpy())
            pr = mdl.predict_proba(iva[FEATURES].to_numpy())[:, 1]
            scores.append(average_precision_score(iva[ycol].to_numpy(), pr))
        if scores and float(np.mean(scores)) > best_score:
            best_score = float(np.mean(scores))
            best = (C, p)
    return best

# --------------------------------------------------------------- T9: tuned + calibrated classification
T9_BASE = ["LogisticRegression", "RandomForest", "HistGB", "XGBoost"]
T9_NAMES = T9_BASE + ["SoftVote(LR+HGB+XGB)"]
t9_store = {n: [] for n in T9_NAMES}
chosen_lr_params = []
calib_notes = []

for Y in TEST_YEARS:
    tr = samples[samples["target_year"] <= Y - 1]
    te = samples[samples["target_year"] == Y]
    prop = tr[tr["target_year"] <= Y - 2]
    cal = tr[tr["target_year"] == Y - 1]
    if cal["y_crash"].nunique() < 2:                 # single-class calibration year -> widen
        cal = tr[tr["target_year"] >= Y - 2]
        prop = tr[tr["target_year"] <= Y - 3]
        calib_notes.append(f"Y={Y}: calibration year single-class; widened calib to years {Y-2},{Y-1}")
    can_calibrate = cal["y_crash"].nunique() == 2 and prop["y_crash"].nunique() == 2 and len(prop) >= 20
    if not can_calibrate:
        calib_notes.append(f"Y={Y}: calibration not possible; using raw probabilities for this fold")
    method = "isotonic" if len(cal) >= 15 else "sigmoid"
    C_star, p_star = tune_logreg(tr)
    chosen_lr_params.append((C_star, p_star))
    yprop = prop["y_crash"].to_numpy()
    npos = int(yprop.sum())
    spw = (len(yprop) - npos) / max(npos, 1)
    models = {
        "LogisticRegression": make_logreg_pipe(C_star, p_star, SEED),
        "RandomForest": make_rf_clf(SEED),
        "HistGB": make_hgb_clf(SEED),
        "XGBoost": make_xgb_clf(spw, SEED),
    }
    Xprop, Xcal, Xte = (prop[FEATURES].to_numpy(), cal[FEATURES].to_numpy(), te[FEATURES].to_numpy())
    raw_probs, cal_probs = {}, {}
    for n, mdl in models.items():
        mdl.fit(Xprop, yprop)
        raw_probs[n] = mdl.predict_proba(Xte)[:, 1]
        if can_calibrate:
            cc = calib_prefit(mdl, Xcal, cal["y_crash"].to_numpy(), method)
            cal_probs[n] = cc.predict_proba(Xte)[:, 1]
        else:
            cal_probs[n] = raw_probs[n]
    raw_probs["SoftVote(LR+HGB+XGB)"] = np.mean(
        [raw_probs["LogisticRegression"], raw_probs["HistGB"], raw_probs["XGBoost"]], axis=0)
    cal_probs["SoftVote(LR+HGB+XGB)"] = np.mean(
        [cal_probs["LogisticRegression"], cal_probs["HistGB"], cal_probs["XGBoost"]], axis=0)
    for n in T9_NAMES:
        t9_store[n].append(pd.DataFrame({
            "prob": cal_probs[n], "prob_raw": raw_probs[n],
            "actual": te["y_crash"].to_numpy(),
            "target_year": te["target_year"].to_numpy(), "Block": te["Block"].to_numpy(),
        }, index=te.index))

t9_pool = {n: pd.concat(v) for n, v in t9_store.items()}
CALIB_METHOD_USED = "sigmoid (calibration folds have n<15)" if all(
    len(samples[samples["target_year"] == Y - 1]) < 15 for Y in TEST_YEARS) else "mixed"

def recall_at_prec(y, p, prec_floor=0.60):
    prec, rec, _ = precision_recall_curve(y, p)
    mask = prec >= prec_floor
    return float(rec[mask].max()) if mask.any() else 0.0

t9_rows = []
for n in T9_NAMES:
    d = t9_pool[n]
    y, p = d["actual"].to_numpy(), d["prob"].to_numpy()
    m = clf_metrics(d)
    t9_rows.append(dict(Model=n, **m,
                        Recall_at_Prec60=recall_at_prec(y, p, 0.60),
                        Brier=float(brier_score_loss(y, p))))
T9 = pd.DataFrame(t9_rows).sort_values(["PR_AUC", "F1"], ascending=False).reset_index(drop=True)
best9_name = T9["Model"].iloc[0]
best9 = t9_pool[best9_name]
best9_y = best9["actual"].to_numpy()
best9_p = best9["prob"].to_numpy()
best9_praw = best9["prob_raw"].to_numpy()
best9_brier_cal = float(brier_score_loss(best9_y, best9_p))
best9_brier_raw = float(brier_score_loss(best9_y, best9_praw))
best9_metrics = clf_metrics(best9)

# modal tuned LogReg hyperparams (used to instantiate the best model in stress tests)
_cnt = Counter(chosen_lr_params)
_order = {cp: i for i, cp in enumerate(LOGREG_GRID)}
modal_lr = sorted(_cnt.items(), key=lambda kv: (-kv[1], _order[kv[0]]))[0][0]

class SoftVote3:
    """Deterministic soft-voting of three fitted models (mean of P(crash))."""
    def __init__(self, models):
        self.models = models
    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self
    def predict_proba(self, X):
        p1 = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1.0 - p1, p1])

def best_factory(spw, seed=SEED):
    """Fresh instance of the T9-best model (uncalibrated), tuned LogReg = modal params."""
    if best9_name == "LogisticRegression":
        return make_logreg_pipe(modal_lr[0], modal_lr[1], seed)
    if best9_name == "RandomForest":
        return make_rf_clf(seed)
    if best9_name == "HistGB":
        return make_hgb_clf(seed)
    if best9_name == "XGBoost":
        return make_xgb_clf(spw, seed)
    return SoftVote3([make_logreg_pipe(modal_lr[0], modal_lr[1], seed),
                      make_hgb_clf(seed), make_xgb_clf(spw, seed)])

def clf_metrics4(y, p):
    yhat = (p >= 0.5).astype(int)
    roc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    pr = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return dict(ROC_AUC=roc, PR_AUC=pr,
                Recall=recall_score(y, yhat, zero_division=0),
                F1=f1_score(y, yhat, zero_division=0))

def rolling_best(df, ycol, tycol, test_years, seed=SEED):
    """Uncalibrated rolling-origin run of the best T9 model (raw probabilities)."""
    parts = []
    for Y in test_years:
        tr = df[df[tycol] <= Y - 1]
        te = df[df[tycol] == Y]
        if len(te) == 0 or tr[ycol].nunique() < 2:
            continue
        ytr = tr[ycol].to_numpy()
        npos = int(ytr.sum())
        spw = (len(ytr) - npos) / max(npos, 1)
        mdl = best_factory(spw, seed)
        mdl.fit(tr[FEATURES].to_numpy(), ytr)
        pr = mdl.predict_proba(te[FEATURES].to_numpy())[:, 1]
        parts.append(pd.DataFrame({"prob": pr, "actual": te[ycol].to_numpy(),
                                   "target_year": te[tycol].to_numpy(),
                                   "Block": te["Block"].to_numpy()}, index=te.index))
    return pd.concat(parts)

# --------------------------------------------------------------- T10: bootstrap CIs + DeLong vs naive baseline
def bootstrap_ci(y, p, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, prs = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, p[idx]))
        prs.append(average_precision_score(yy, p[idx]))
    return (np.percentile(aucs, [2.5, 97.5]), np.percentile(prs, [2.5, 97.5]),
            len(aucs))

def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    Nn = len(x)
    T = np.zeros(Nn, dtype=float)
    i = 0
    while i < Nn:
        j = i
        while j < Nn and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(Nn)
    out[J] = T
    return out

def delong_roc_test(y_true, p1, p2):
    """Fast DeLong test for two correlated ROC AUCs. Returns (auc1, auc2, diff, p)."""
    y = np.asarray(y_true).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    m, nn = len(pos), len(neg)
    preds = np.vstack([np.concatenate([p[pos], p[neg]]) for p in (p1, p2)])
    k = preds.shape[0]
    tx = np.array([_midrank(preds[r, :m]) for r in range(k)])
    ty = np.array([_midrank(preds[r, m:]) for r in range(k)])
    tz = np.array([_midrank(preds[r, :]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * nn)
    v01 = (tz[:, :m] - tx) / nn
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    S = sx / m + sy / nn
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    d = float(aucs[0] - aucs[1])
    if var <= 1e-12:                                  # fallback: paired bootstrap
        rng = np.random.default_rng(SEED)
        n = len(y)
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            diffs.append(roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p2[idx]))
        diffs = np.array(diffs)
        pval = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        return float(aucs[0]), float(aucs[1]), d, pval
    z = d / math.sqrt(var)
    pval = float(2 * (1 - norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), d, pval

auc_ci, pr_ci, n_boot_valid = bootstrap_ci(best9_y, best9_p, B=2000, seed=SEED)
naive_score = samples.loc[best9.index, "Yield_dev_clim_pct"].to_numpy()   # higher dev above climatology -> higher risk
naive_auc = float(roc_auc_score(best9_y, naive_score))
dl_auc1, dl_auc2, dl_diff, dl_p = delong_roc_test(best9_y, best9_p, naive_score)
T10 = pd.DataFrame([
    dict(Quantity=f"{best9_name} ROC_AUC", Value=float(roc_auc_score(best9_y, best9_p)),
         CI95_lo=auc_ci[0], CI95_hi=auc_ci[1], Note=f"bootstrap B=2000 ({n_boot_valid} valid resamples)"),
    dict(Quantity=f"{best9_name} PR_AUC", Value=float(average_precision_score(best9_y, best9_p)),
         CI95_lo=pr_ci[0], CI95_hi=pr_ci[1], Note="bootstrap B=2000"),
    dict(Quantity="Naive baseline ROC_AUC (score = Yield_dev_clim_pct)", Value=naive_auc,
         CI95_lo=np.nan, CI95_hi=np.nan, Note="single-feature ranking, no model"),
    dict(Quantity="AUC difference (model - naive)", Value=dl_diff,
         CI95_lo=np.nan, CI95_hi=np.nan, Note=f"DeLong p = {dl_p:.4f}"),
])

# --------------------------------------------------------------- T11: generalization stress tests
# (a) Leave-one-block-out  -- SPATIAL TRANSFER, not one-year-ahead forecasting
lobo_parts = []
for b in BLOCKS:
    tr = samples[samples["Block"] != b]
    te = samples[samples["Block"] == b]
    ytr = tr["y_crash"].to_numpy()
    npos = int(ytr.sum())
    spw = (len(ytr) - npos) / max(npos, 1)
    mdl = best_factory(spw, SEED)
    mdl.fit(tr[FEATURES].to_numpy(), ytr)
    pr = mdl.predict_proba(te[FEATURES].to_numpy())[:, 1]
    lobo_parts.append(pd.DataFrame({"prob": pr, "actual": te["y_crash"].to_numpy(),
                                    "Block": te["Block"].to_numpy()}, index=te.index))
lobo = pd.concat(lobo_parts)
lobo_m = clf_metrics4(lobo["actual"].to_numpy(), lobo["prob"].to_numpy())
lobo_block_auc = {}
for b, g in lobo.groupby("Block"):
    yy = g["actual"].to_numpy()
    lobo_block_auc[b] = float(roc_auc_score(yy, g["prob"])) if len(np.unique(yy)) == 2 else np.nan

# (b) Leave-last-2-years-out (hard temporal holdout)
tr_h = samples[samples["target_year"] <= 2023]
te_h = samples[samples["target_year"].isin([2024, 2025])]
ytr_h = tr_h["y_crash"].to_numpy()
spw_h = (len(ytr_h) - int(ytr_h.sum())) / max(int(ytr_h.sum()), 1)
mdl_h = best_factory(spw_h, SEED)
mdl_h.fit(tr_h[FEATURES].to_numpy(), ytr_h)
pr_h = mdl_h.predict_proba(te_h[FEATURES].to_numpy())[:, 1]
hold_m = clf_metrics4(te_h["y_crash"].to_numpy(), pr_h)

ref_m = dict(ROC_AUC=best9_metrics["ROC_AUC"], PR_AUC=best9_metrics["PR_AUC"],
             Recall=best9_metrics["Recall"], F1=best9_metrics["F1"])
T11 = pd.DataFrame([
    dict(Protocol="Rolling-origin one-year-ahead (T9 best, calibrated) [reference]",
         n_test=len(best9), **ref_m),
    dict(Protocol="Leave-one-block-out (SPATIAL transfer, uncalibrated; NOT one-year-ahead)",
         n_test=len(lobo), **lobo_m),
    dict(Protocol="Leave-last-2-years-out (hard temporal holdout 2024-2025, uncalibrated)",
         n_test=len(te_h), **hold_m),
])

# --------------------------------------------------------------- T12: multi-horizon (h=1 vs h=2)
# Same uncalibrated best-model rolling-origin machinery for both horizons (apples-to-apples).
h1_df = rolling_best(samples, "y_crash", "target_year", TEST_YEARS, SEED)
h2_years = list(range(2017, 2026))
h2_df = rolling_best(samples_h2, "y_crash_h2", "target_year_h2", h2_years, SEED)
h1_m = clf_metrics4(h1_df["actual"].to_numpy(), h1_df["prob"].to_numpy())
h2_m = clf_metrics4(h2_df["actual"].to_numpy(), h2_df["prob"].to_numpy())
prev_h1 = float(h1_df["actual"].mean())
prev_h2 = float(h2_df["actual"].mean())
T12 = pd.DataFrame([
    dict(Horizon="h=1 (crash at t+1)", test_years=f"{TEST_YEARS[0]}-{TEST_YEARS[-1]}",
         n_test=len(h1_df), prevalence=prev_h1, **h1_m),
    dict(Horizon="h=2 (crash at t+2 vs Yield_t)", test_years=f"{h2_years[0]}-{h2_years[-1]}",
         n_test=len(h2_df), prevalence=prev_h2, **h2_m),
])

# --------------------------------------------------------------- T13: decision curve + cost analysis + tiers
n_pool9 = len(best9_y)
prev9 = float(best9_y.mean())
nb_rows = []
for pt in np.arange(0.05, 0.601, 0.05):
    pt = round(float(pt), 2)
    yhat = (best9_p >= pt).astype(int)
    tp = int(((yhat == 1) & (best9_y == 1)).sum())
    fp = int(((yhat == 1) & (best9_y == 0)).sum())
    w = pt / (1 - pt)
    nb_model = tp / n_pool9 - fp / n_pool9 * w
    nb_all = prev9 - (1 - prev9) * w
    nb_rows.append(dict(pt=pt, NB_model=nb_model, NB_all=nb_all, NB_none=0.0))
T13_nb = pd.DataFrame(nb_rows)

# cost analysis: C_miss : C_falsealarm = 5 : 1
cost_rows = []
for thr in np.arange(0.10, 0.9001, 0.05):
    thr = round(float(thr), 2)
    yhat = (best9_p >= thr).astype(int)
    fn = int(((yhat == 0) & (best9_y == 1)).sum())
    fp = int(((yhat == 1) & (best9_y == 0)).sum())
    cost = (5.0 * fn + 1.0 * fp) / n_pool9
    cost_rows.append(dict(threshold=thr, exp_cost_per_sample=cost,
                          Precision=precision_score(best9_y, yhat, zero_division=0),
                          Recall=recall_score(best9_y, yhat, zero_division=0),
                          F1=f1_score(best9_y, yhat, zero_division=0)))
T13_cost = pd.DataFrame(cost_rows)
opt_i = int(T13_cost["exp_cost_per_sample"].idxmin())         # first minimum (lowest threshold on ties)
opt_thr = float(T13_cost.loc[opt_i, "threshold"])

# tiered early-warning score
tiers = pd.cut(best9_p, bins=[-np.inf, 0.33, 0.66, np.inf], labels=["Low", "Medium", "High"])
tier_rows = []
for tname in ["Low", "Medium", "High"]:
    mask = (tiers == tname)
    nn = int(mask.sum())
    tier_rows.append(dict(Tier=tname, n=nn,
                          mean_prob=float(best9_p[mask].mean()) if nn else np.nan,
                          observed_crash_rate=float(best9_y[mask].mean()) if nn else np.nan))
T13_tier = pd.DataFrame(tier_rows)

# --------------------------------------------------------------- T14: variance decomposition (predictability ceiling)
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

vd = pool[["Block", "target_year", "y_yield", "Yield_t"]].copy()   # TEST POOL (target years 2015-2025)
vd["y_change"] = vd["y_yield"] - vd["Yield_t"]

def variance_decomp(df, ycol):
    m = smf.ols(f"{ycol} ~ C(Block) + C(target_year)", data=df).fit()
    a = anova_lm(m, typ=2)                     # Type-II SS (balanced 10x11 panel -> = sequential)
    ss = a["sum_sq"]
    tot = float(ss.sum())
    return dict(Block_pct=100 * float(ss["C(Block)"]) / tot,
                Year_pct=100 * float(ss["C(target_year)"]) / tot,
                Residual_pct=100 * float(ss["Residual"]) / tot,
                R2=float(m.rsquared))

vd_level = variance_decomp(vd, "y_yield")
vd_change = variance_decomp(vd, "y_change")
T14 = pd.DataFrame([
    dict(Target="Next-year yield level (y_yield)", **vd_level),
    dict(Target="YoY change (y_yield - Yield_t)", **vd_change),
])

# --------------------------------------------------------------- T15: per-block conformal coverage (90%)
t15_rows = []
for b, g in conf90_df.groupby("Block"):
    cov = float(((g["actual"] >= g["lo"]) & (g["actual"] <= g["hi"])).mean())
    wid = float((g["hi"] - g["lo"]).mean())
    t15_rows.append(dict(Block=b, n=len(g), Empirical_coverage_90=cov, Mean_width=wid))
T15 = pd.DataFrame(t15_rows)

# --------------------------------------------------------------- T16: robustness
# (a) seed stability of best classifier (uncalibrated rolling-origin)
seed_rows = []
for sd in (1, 7, 42, 123, 2024):
    d = rolling_best(samples, "y_crash", "target_year", TEST_YEARS, sd)
    m = clf_metrics4(d["actual"].to_numpy(), d["prob"].to_numpy())
    seed_rows.append(dict(seed=sd, ROC_AUC=m["ROC_AUC"], PR_AUC=m["PR_AUC"]))
T16_seed = pd.DataFrame(seed_rows)
seed_summary = dict(ROC_AUC_mean=float(T16_seed["ROC_AUC"].mean()),
                    ROC_AUC_sd=float(T16_seed["ROC_AUC"].std(ddof=1)),
                    PR_AUC_mean=float(T16_seed["PR_AUC"].mean()),
                    PR_AUC_sd=float(T16_seed["PR_AUC"].std(ddof=1)))

# (b) extended theta sensitivity with the best T9 model (uncalibrated rolling-origin)
t16b_rows = []
for th in (0.10, 0.15, 0.20, 0.25, 0.30):
    ycol = f"y_crash_{int(round(th*100))}"
    if ycol not in samples.columns:
        samples[ycol] = crash_labels(th)
    d = rolling_best(samples, ycol, "target_year", TEST_YEARS, SEED)
    m = clf_metrics4(d["actual"].to_numpy(), d["prob"].to_numpy())
    t16b_rows.append(dict(theta=th,
                          prevalence_all=float(samples[ycol].mean()),
                          prevalence_test=float(samples.loc[test_mask, ycol].mean()),
                          F1=m["F1"], ROC_AUC=m["ROC_AUC"], PR_AUC=m["PR_AUC"]))
T16_theta = pd.DataFrame(t16b_rows)

# =============================================================================
# ------------------------------------------------------------------ write CSVs
def rnd(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    return out

tables = {
    "T1_feature_dictionary": T1,
    "T2_regression_rolling_origin": rnd(T2),
    "T2b_production_regression": rnd(T2b),
    "T3_baselines_and_skill": rnd(T3),
    "T4_feature_ablation": rnd(T4),
    "T5_classification_earlywarning": rnd(T5),
    "T5b_theta_sensitivity": rnd(T5_theta),
    "T6_conformal": rnd(T6),
    "T7_breakdown_per_year": rnd(T7_year),
    "T7_breakdown_per_block": rnd(T7_block),
    "T8_leakage_contrast": rnd(T8),
    "T9_tuned_calibrated_classification": rnd(T9),
    "T10_bootstrap_delong": rnd(T10),
    "T11_generalization_stress": rnd(T11),
    "T12_multi_horizon": rnd(T12),
    "T13a_net_benefit": rnd(T13_nb),
    "T13b_cost_analysis": rnd(T13_cost),
    "T13c_tiered_score": rnd(T13_tier),
    "T14_variance_decomposition": rnd(T14),
    "T15_per_block_conformal": rnd(T15),
    "T16a_seed_stability": rnd(T16_seed),
    "T16b_theta_extended": rnd(T16_theta),
}
for name, df in tables.items():
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)

pd.DataFrame({"feature": rank_reg.index, "importance": rank_reg.values}).to_csv(
    os.path.join(TAB, "importance_regression.csv"), index=False)
pd.DataFrame({"feature": rank_clf.index, "importance": rank_clf.values}).to_csv(
    os.path.join(TAB, "importance_classification.csv"), index=False)

# ------------------------------------------------------------------ DOCX compilation
from docx import Document
from docx.shared import Pt

doc = Document()
doc.add_heading("Koraput Coffee Early-Warning Prediction v2: Tables", level=0)
doc.add_paragraph(f"One-year-ahead prediction, rolling-origin evaluation (target years "
                  f"{TEST_YEARS[0]}-{TEST_YEARS[-1]}, {len(TEST_YEARS)} folds, {N} samples). Seed 42. "
                  f"v2 adds {len(NEW_FEATURES)} leakage-safe features, tuned+calibrated classification, "
                  f"uncertainty/significance, stress tests, multi-horizon, decision analysis, "
                  f"variance decomposition and robustness checks.")

def add_table(doc, title, df):
    doc.add_heading(title, level=1)
    t = doc.add_table(rows=1 + len(df), cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for j, c in enumerate(df.columns):
        t.rows[0].cells[j].text = str(c)
    for i in range(len(df)):
        for j, c in enumerate(df.columns):
            v = df.iloc[i, j]
            t.rows[i + 1].cells[j].text = f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8)

add_table(doc, "T1. Feature dictionary (all leakage-safe, years <= t only)", T1)
add_table(doc, "T2. Regression, rolling-origin pooled (target: Yield per Ha at t+1)", rnd(T2))
add_table(doc, "T2b. Regression, rolling-origin pooled (target: Production MT at t+1)", rnd(T2b))
add_table(doc, "T3. Baselines and best-model skill vs each baseline (Yield)", rnd(T3))
add_table(doc, f"T4. Feature-group ablation ({best_reg_name}, Yield)", rnd(T4))
add_table(doc, "T5. Early-warning classification, rolling-origin pooled (theta = 0.20, untuned/uncalibrated)", rnd(T5))
add_table(doc, f"T5b. Theta sensitivity ({best_clf_name}, v1 protocol)", rnd(T5_theta))
add_table(doc, f"T6. Split-conformal intervals ({best_reg_name}, Yield)", rnd(T6))
add_table(doc, f"T7a. Per-target-year breakdown ({best_reg_name}; crash recall: {best_clf_name})", rnd(T7_year))
add_table(doc, f"T7b. Per-block breakdown ({best_reg_name})", rnd(T7_block))
add_table(doc, f"T8. Leakage contrast ({best_reg_name})", rnd(T8))
add_table(doc, f"T9. Tuned + calibrated classification, theta=0.20 (calibration: {CALIB_METHOD_USED}; "
               f"proper-train <= Y-2, calibrate on Y-1)", rnd(T9))
add_table(doc, f"T10. Bootstrap 95% CIs and DeLong test ({best9_name} vs naive Yield_dev_clim_pct score)", rnd(T10))
add_table(doc, f"T11. Generalization stress tests ({best9_name})", rnd(T11))
add_table(doc, f"T12. Multi-horizon early warning ({best9_name}, uncalibrated rolling-origin, both horizons)", rnd(T12))
add_table(doc, f"T13a. Decision curve: net benefit across threshold probabilities ({best9_name}, calibrated)", rnd(T13_nb))
add_table(doc, f"T13b. Cost analysis, C_miss:C_falsealarm = 5:1 (cost-optimal threshold = {opt_thr:.2f})", rnd(T13_cost))
add_table(doc, "T13c. Tiered early-warning score (calibrated probability tiers)", rnd(T13_tier))
add_table(doc, "T14. Variance decomposition of next-year yield, TEST POOL 2015-2025 (OLS Type-II SS)", rnd(T14))
add_table(doc, f"T15. Per-block 90% conformal coverage ({best_reg_name}; marginal intervals)", rnd(T15))
add_table(doc, f"T16a. Seed stability ({best9_name}, uncalibrated rolling-origin)", rnd(T16_seed))
add_table(doc, f"T16b. Extended theta sensitivity ({best9_name}, uncalibrated rolling-origin)", rnd(T16_theta))
docx_path = os.path.join(TAB, "Koraput_EarlyWarning_v2_tables.docx")
doc.save(docx_path)

# =============================================================================
# ------------------------------------------------------------------ FIGURES (v1 set)
import matplotlib.patches as mpatches

# F1: rolling-origin schematic
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, Y in enumerate(TEST_YEARS):
    ax.barh(i, (Y - 1) - 2009 + 1, left=2009, height=0.62, color=C_BLUE, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.barh(i, 1, left=Y, height=0.62, color=C_ORANGE, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(TEST_YEARS)))
ax.set_yticklabels([f"Fold {i+1}" for i in range(len(TEST_YEARS))])
ax.invert_yaxis()
ax.set_xlabel("Target year (t+1)")
ax.set_xlim(2008.5, 2026.2)
ax.set_title("Rolling-origin (expanding window) evaluation:\ntrain on target years <= Y-1, test on target year Y")
ax.legend(handles=[mpatches.Patch(color=C_BLUE, alpha=0.85, label="Training target years"),
                   mpatches.Patch(color=C_ORANGE, label="Test target year")],
          loc="lower left")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F1_rolling_origin_schematic.png"))
plt.close(fig)

# F2: predicted vs actual with 90% conformal band
blk_order = T7_block.sort_values("RMSE")["Block"].tolist()
rep_blocks = [blk_order[0], blk_order[len(blk_order)//3], blk_order[2*len(blk_order)//3], blk_order[-1]]
fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
for ax, b in zip(axes.ravel(), rep_blocks):
    g = conf90_df[conf90_df["Block"] == b].sort_values("target_year")
    ax.fill_between(g["target_year"], g["lo"], g["hi"], color=C_BLUE, alpha=0.18,
                    label="90% conformal interval")
    ax.plot(g["target_year"], g["pred"], color=C_BLUE, lw=1.8, marker="o", ms=3.5,
            label=f"Predicted ({best_reg_name})")
    ax.plot(g["target_year"], g["actual"], color=C_VERM, lw=1.8, marker="s", ms=3.5,
            label="Actual")
    ax.set_title(b)
    ax.set_ylabel("Yield per Ha (MT)")
axes[0, 0].legend(fontsize=7.5, loc="best")
for ax in axes[1, :]:
    ax.set_xlabel("Target year")
fig.suptitle("One-year-ahead yield prediction with 90% split-conformal band", y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F2_pred_vs_actual_conformal.png"))
plt.close(fig)

# F3: per-year RMSE model vs persistence
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = T7_year["target_year"].to_numpy()
w = 0.38
ax.bar(x - w/2, T7_year["RMSE"], width=w, color=C_BLUE, label=f"{best_reg_name}")
ax.bar(x + w/2, T7_year["RMSE_pers"], width=w, color=C_GRAY, label="Persistence baseline")
ax.set_xlabel("Target year")
ax.set_ylabel("RMSE (Yield per Ha, MT)")
ax.set_title("Per-year one-year-ahead RMSE: best model vs persistence")
ax.set_xticks(x)
ax.legend()
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F3_skill_by_year.png"))
plt.close(fig)

# F4 / F4b: importance bars
def importance_fig(series, title, path, color):
    s = series.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.barh(s.index, s.values, color=color)
    ax.set_xlabel("Mean |SHAP| (model units)" if "SHAP" in title else "Importance")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

reg_kind = "SHAP mean|value|" if shap_reg is not None else "Permutation importance"
clf_kind = "SHAP mean|value|" if shap_clf is not None else "Permutation importance"
importance_fig(rank_reg, f"Regression feature importance ({best_reg_name}, {reg_kind})",
               os.path.join(FIG, "F4_importance_regression.png"), C_BLUE)
importance_fig(rank_clf, f"Crash-classifier feature importance ({best_clf_name}, {clf_kind})",
               os.path.join(FIG, "F4b_importance_classification.png"), C_GREEN)

# F5: confusion + ROC + PR (v1-style classifier)
y_c, p_c = best_clf_pred["actual"].to_numpy(), best_clf_pred["prob"].to_numpy()
yhat_c = (p_c >= 0.5).astype(int)
cm = confusion_matrix(y_c, yhat_c)
fpr, tpr, _ = roc_curve(y_c, p_c)
prec, rec, _ = precision_recall_curve(y_c, p_c)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
ax = axes[0]
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center",
            color="white" if v > cm.max()/2 else "#1a1a1a", fontsize=11)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No crash", "Crash"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No crash", "Crash"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix ({best_clf_name}, thr 0.5)")
ax.grid(visible=False)
ax = axes[1]
ax.plot(fpr, tpr, color=C_BLUE, lw=1.8, label=f"AUC = {roc_auc_score(y_c, p_c):.3f}")
ax.plot([0, 1], [0, 1], color=C_GRAY, lw=1, ls="--")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC (pooled rolling-origin)")
ax.legend(loc="lower right")
ax = axes[2]
ax.plot(rec, prec, color=C_GREEN, lw=1.8, label=f"AP = {average_precision_score(y_c, p_c):.3f}")
ax.axhline(y_c.mean(), color=C_GRAY, lw=1, ls="--", label=f"Prevalence = {y_c.mean():.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall (pooled)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F5_confusion_and_roc_pr.png"))
plt.close(fig)

# F6: ablation
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
labels = ["A\ncurrent-year", "B\n+lags/trend", "C\n+block hist\n+spatial+v2 (full)"]
rmses = T4["RMSE"].to_numpy()
skills = T4["Skill_pct"].to_numpy()
axes[0].bar(labels, rmses, color=[C_GRAY, C_ORANGE, C_BLUE])
axes[0].set_ylabel("RMSE (Yield per Ha, MT)")
axes[0].set_title(f"Ablation RMSE ({best_reg_name})")
axes[0].grid(axis="x", visible=False)
axes[1].bar(labels, skills, color=[C_GRAY, C_ORANGE, C_BLUE])
axes[1].axhline(0, color="#1a1a1a", lw=0.8)
axes[1].set_ylabel("Skill vs best baseline (%)")
axes[1].set_title("Ablation skill")
axes[1].grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F6_ablation.png"))
plt.close(fig)

# ------------------------------------------------------------------ FIGURES (new v2)
# F7: calibration / reliability curve (best T9 model, calibrated vs uncalibrated)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for probs, lab, col in [(best9_praw, f"Uncalibrated (Brier {best9_brier_raw:.3f})", C_GRAY),
                        (best9_p, f"Calibrated (Brier {best9_brier_cal:.3f})", C_BLUE)]:
    frac_pos, mean_pred = calibration_curve(best9_y, probs, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", ms=4, lw=1.6, color=col, label=lab)
ax.plot([0, 1], [0, 1], color="#1a1a1a", lw=0.9, ls="--", label="Perfect calibration")
ax.set_xlabel("Mean predicted crash probability (bin)")
ax.set_ylabel("Observed crash fraction (bin)")
ax.set_title(f"Reliability curve, {best9_name}\n(pooled rolling-origin test predictions)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F7_calibration_reliability.png"))
plt.close(fig)

# F8: decision curve
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(T13_nb["pt"], T13_nb["NB_model"], color=C_BLUE, lw=1.9, marker="o", ms=3.5,
        label=f"{best9_name} (calibrated)")
ax.plot(T13_nb["pt"], T13_nb["NB_all"], color=C_ORANGE, lw=1.7, ls="--", label="Treat all")
ax.axhline(0.0, color=C_GRAY, lw=1.2, ls=":", label="Treat none")
ax.set_xlabel("Threshold probability p_t")
ax.set_ylabel("Net benefit")
ax.set_ylim(bottom=max(-0.15, float(T13_nb[["NB_model", "NB_all"]].min().min()) - 0.05))
ax.set_title("Decision curve: net benefit of the early-warning model")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F8_decision_curve.png"))
plt.close(fig)

# F9 / F9b: SHAP beeswarm + dependence for the best T9 CLASSIFIER
shap_target_name = best9_name
if best9_name.startswith("SoftVote"):
    _ind = T9[T9["Model"].isin(T9_BASE)].sort_values(["PR_AUC", "F1"], ascending=False)["Model"].iloc[0]
    shap_target_name = _ind
shap_fig_note = ""
try:
    import shap
    tr_s = samples[samples["target_year"] <= 2024]
    X_s = tr_s[FEATURES].to_numpy()
    y_s = tr_s["y_crash"].to_numpy()
    spw_s = (len(y_s) - y_s.sum()) / max(y_s.sum(), 1)
    if shap_target_name == "LogisticRegression":
        mdl_s = make_logreg_pipe(modal_lr[0], modal_lr[1], SEED)
    elif shap_target_name == "RandomForest":
        mdl_s = make_rf_clf(SEED)
    elif shap_target_name == "HistGB":
        mdl_s = make_hgb_clf(SEED)
    else:
        mdl_s = make_xgb_clf(spw_s, SEED)
    mdl_s.fit(X_s, y_s)
    if shap_target_name == "LogisticRegression":
        Xs_sc = mdl_s.named_steps["sc"].transform(X_s)
        ex = shap.LinearExplainer(mdl_s.named_steps["m"], Xs_sc)
        sv = ex.shap_values(Xs_sc)
        X_disp = Xs_sc
        x_note = " (standardized units)"
    else:
        ex = shap.TreeExplainer(mdl_s)
        sv = ex.shap_values(X_s)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        X_disp = X_s
        x_note = ""
    plt.figure(figsize=(7.6, 6.4))
    shap.summary_plot(sv, X_disp, feature_names=FEATURES, show=False, max_display=15)
    plt.title(f"SHAP beeswarm, crash classifier ({shap_target_name}, trained on target years <= 2024)",
              fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "F9_shap_beeswarm_classifier.png"))
    plt.close()
    top_i = int(np.argmax(np.abs(sv).mean(axis=0)))
    top_feat = FEATURES[top_i]
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.scatter(tr_s[top_feat].to_numpy(), sv[:, top_i], s=16, c=C_BLUE, alpha=0.65,
               edgecolors="none")
    ax.axhline(0, color=C_GRAY, lw=0.9, ls="--")
    ax.set_xlabel(f"{top_feat} (raw value)")
    ax.set_ylabel(f"SHAP value for {top_feat}{x_note}")
    ax.set_title(f"SHAP dependence: {top_feat} ({shap_target_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F9b_shap_dependence_top_feature.png"))
    plt.close(fig)
    if best9_name.startswith("SoftVote"):
        shap_fig_note = (f"SHAP shown for {shap_target_name}, the strongest individual component "
                         f"of the {best9_name} ensemble.")
except Exception as e:
    shap_fig_note = f"SHAP failed ({type(e).__name__}: {e}); falling back to permutation importance figure."
    tr_s = samples[samples["target_year"] <= 2024]
    y_s = tr_s["y_crash"].to_numpy()
    spw_s = (len(y_s) - y_s.sum()) / max(y_s.sum(), 1)
    mdl_s = best_factory(spw_s, SEED)
    mdl_s.fit(tr_s[FEATURES].to_numpy(), y_s)
    pi = permutation_importance(mdl_s, tr_s[FEATURES].to_numpy(), y_s, n_repeats=30,
                                random_state=SEED)
    s = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    importance_fig(s, f"Permutation importance fallback ({best9_name})",
                   os.path.join(FIG, "F9_shap_beeswarm_classifier.png"), C_BLUE)

# F10: multi-horizon skill bars
fig, ax = plt.subplots(figsize=(5.6, 4.0))
xpos = np.arange(2)
w = 0.34
ax.bar(xpos - w/2, [h1_m["ROC_AUC"], h2_m["ROC_AUC"]], width=w, color=C_BLUE, label="ROC AUC")
ax.bar(xpos + w/2, [h1_m["PR_AUC"], h2_m["PR_AUC"]], width=w, color=C_GREEN, label="PR AUC")
for i, (a, p_) in enumerate([(h1_m["ROC_AUC"], h1_m["PR_AUC"]), (h2_m["ROC_AUC"], h2_m["PR_AUC"])]):
    ax.text(i - w/2, a + 0.012, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i + w/2, p_ + 0.012, f"{p_:.3f}", ha="center", fontsize=8)
ax.axhline(0.5, color=C_GRAY, lw=1, ls="--")
ax.set_xticks(xpos)
ax.set_xticklabels([f"h = 1\n(prev {prev_h1:.2f})", f"h = 2\n(prev {prev_h2:.2f})"])
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title(f"Multi-horizon early-warning skill ({best9_name}, uncalibrated)")
ax.legend()
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F10_multihorizon_skill.png"))
plt.close(fig)

# F11: LOBO per-block AUC bars (spatial transfer)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
bl_names = list(lobo_block_auc.keys())
bl_vals = [lobo_block_auc[b] for b in bl_names]
cols = [C_BLUE if not np.isnan(v) else C_GRAY for v in bl_vals]
ax.bar(bl_names, [0 if np.isnan(v) else v for v in bl_vals], color=cols)
for i, v in enumerate(bl_vals):
    if np.isnan(v):
        ax.text(i, 0.03, "single-class", rotation=90, ha="center", fontsize=7.5, color="#1a1a1a")
    else:
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0.5, color=C_VERM, lw=1.1, ls="--", label="Chance (AUC 0.5)")
ax.axhline(lobo_m["ROC_AUC"], color=C_GREEN, lw=1.3, ls="-.",
           label=f"Pooled LOBO AUC = {lobo_m['ROC_AUC']:.3f}")
ax.set_ylabel("Within-block ROC AUC")
ax.set_ylim(0, 1.05)
ax.set_title("Leave-one-block-out SPATIAL transfer (NOT one-year-ahead forecasting)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F11_lobo_block_auc.png"))
plt.close(fig)

# F12: variance decomposition stacked bar
fig, ax = plt.subplots(figsize=(5.8, 4.2))
cats = ["Next-year yield\n(level)", "YoY change\n(y - Yield_t)"]
blocks_sh = [vd_level["Block_pct"], vd_change["Block_pct"]]
years_sh = [vd_level["Year_pct"], vd_change["Year_pct"]]
resid_sh = [vd_level["Residual_pct"], vd_change["Residual_pct"]]
ax.bar(cats, blocks_sh, color=C_BLUE, label="Block (spatial)")
ax.bar(cats, years_sh, bottom=blocks_sh, color=C_ORANGE, label="Year (common shock)")
ax.bar(cats, resid_sh, bottom=[b + y_ for b, y_ in zip(blocks_sh, years_sh)],
       color=C_GRAY, label="Residual (idiosyncratic)")
for i in range(2):
    ax.text(i, blocks_sh[i]/2, f"{blocks_sh[i]:.0f}%", ha="center", color="white", fontsize=9)
    ax.text(i, blocks_sh[i] + years_sh[i]/2, f"{years_sh[i]:.0f}%", ha="center", color="#1a1a1a", fontsize=9)
    ax.text(i, blocks_sh[i] + years_sh[i] + resid_sh[i]/2, f"{resid_sh[i]:.0f}%", ha="center",
            color="white", fontsize=9)
ax.set_ylabel("Share of total sum of squares (%)")
ax.set_title("Variance decomposition of next-year yield\n(test pool 2015-2025; OLS Type-II SS)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F12_variance_decomposition.png"))
plt.close(fig)

# F13: refreshed ROC + PR for best calibrated model
fpr9, tpr9, _ = roc_curve(best9_y, best9_p)
prec9, rec9, _ = precision_recall_curve(best9_y, best9_p)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
ax = axes[0]
ax.plot(fpr9, tpr9, color=C_BLUE, lw=1.9, label=f"AUC = {roc_auc_score(best9_y, best9_p):.3f}")
ax.plot([0, 1], [0, 1], color=C_GRAY, lw=1, ls="--")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC, {best9_name} (calibrated, pooled)")
ax.legend(loc="lower right")
ax = axes[1]
ax.plot(rec9, prec9, color=C_GREEN, lw=1.9,
        label=f"AP = {average_precision_score(best9_y, best9_p):.3f}")
ax.axhline(best9_y.mean(), color=C_GRAY, lw=1, ls="--",
           label=f"Prevalence = {best9_y.mean():.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall (calibrated, pooled)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F13_roc_pr_calibrated_best.png"))
plt.close(fig)

# =============================================================================
# ------------------------------------------------------------------ SUMMARY
def fm(x, n=4):
    return f"{x:.{n}f}"

line = "=" * 78
print()
print(line)
print("KORAPUT EARLY-WARNING v2 EXPERIMENTS -- FINAL NUMERIC SUMMARY (seed 42)")
print(line)
print(f"Samples: {N} (10 blocks x base years 2008-2024); target years 2009-2025")
print(f"Features: {len(FEATURES)} total ({len(V1_FEATURES)} v1 + {len(NEW_FEATURES)} new v2)")
print(f"Rolling-origin folds: {len(TEST_YEARS)} (test target years {TEST_YEARS[0]}-{TEST_YEARS[-1]}; "
      f"{len(pool)} pooled test predictions)")
print(f"Crash prevalence (theta=0.20): all samples {fm(prevalence_all,4)} "
      f"({int(samples['y_crash'].sum())}/{N}); test pool {fm(prevalence_test,4)} "
      f"({int(pool['y_crash'].sum())}/{len(pool)})")
print()
print("--- T2: REGRESSION (Yield per Ha at t+1), rolling-origin pooled ---")
print(rnd(T2).to_string(index=False))
print(f"BEST REGRESSION MODEL: {best_reg_name}  "
      f"(RMSE {fm(best_reg_rmse)}, skill {fm(100*(1-best_reg_rmse/best_bl_yield_rmse),2)}% "
      f"vs best baseline = {best_bl_yield_name})")
print()
print("--- T2b: REGRESSION (Production MT at t+1), rolling-origin pooled ---")
print(rnd(T2b).to_string(index=False))
print(f"BEST PRODUCTION MODEL: {best_prod_name}")
print()
print("--- T3: baselines (Yield) & best-model skill vs each ---")
print(rnd(T3).to_string(index=False))
print()
print("--- T4: ABLATION (best model, Yield) ---")
print(rnd(T4).to_string(index=False))
print()
print("--- T5: CLASSIFICATION v1-protocol (untuned/uncalibrated), rolling-origin pooled ---")
print(rnd(T5).to_string(index=False))
print(f"BEST v1-PROTOCOL CLASSIFIER: {best_clf_name}")
print()
print("--- T5b: theta sensitivity (v1 protocol) ---")
print(rnd(T5_theta).to_string(index=False))
print()
print("--- T6: CONFORMAL (best regression model) ---")
print(rnd(T6).to_string(index=False))
print()
print("--- T7a: per-target-year breakdown ---")
print(rnd(T7_year).to_string(index=False))
print()
print("--- T7b: per-block breakdown ---")
print(rnd(T7_block).to_string(index=False))
print()
print("--- T8: LEAKAGE CONTRAST (best regression model) ---")
print(rnd(T8).to_string(index=False))
print()
print("--- T9: TUNED + CALIBRATED CLASSIFICATION (headline; theta=0.20) ---")
print(f"Calibration: proper-train <= Y-2, calibrate on Y-1; method: {CALIB_METHOD_USED}")
print(f"Tuned LogReg (C, penalty) per fold: {chosen_lr_params}")
print(f"Modal tuned LogReg params: C={modal_lr[0]}, penalty={modal_lr[1]}")
if calib_notes:
    for cn in calib_notes:
        print("  NOTE:", cn)
print(rnd(T9).to_string(index=False))
print(f"BEST v2 CLASSIFIER: {best9_name}")
print(f"Brier (best model): calibrated {fm(best9_brier_cal)} vs uncalibrated {fm(best9_brier_raw)}")
print()
print("--- T10: BOOTSTRAP CIs + DeLong TEST ---")
print(rnd(T10).to_string(index=False))
print(f"DeLong: model AUC {fm(dl_auc1)} vs naive-score AUC {fm(dl_auc2)}; "
      f"diff {fm(dl_diff)}; p = {dl_p:.4f}")
print()
print("--- T11: GENERALIZATION STRESS TESTS ---")
print(rnd(T11).to_string(index=False))
print("Per-block LOBO AUC (spatial transfer): "
      + "; ".join(f"{b}={('NA' if np.isnan(v) else f'{v:.3f}')}" for b, v in lobo_block_auc.items()))
print()
print("--- T12: MULTI-HORIZON EARLY WARNING (uncalibrated rolling-origin, both horizons) ---")
print(rnd(T12).to_string(index=False))
print()
print("--- T13a: NET BENEFIT (decision curve) ---")
print(rnd(T13_nb).to_string(index=False))
print()
print("--- T13b: COST ANALYSIS (C_miss : C_falsealarm = 5 : 1) ---")
print(rnd(T13_cost).to_string(index=False))
print(f"COST-OPTIMAL THRESHOLD: {opt_thr:.2f}  "
      f"(Precision {fm(float(T13_cost.loc[opt_i,'Precision']))}, "
      f"Recall {fm(float(T13_cost.loc[opt_i,'Recall']))}, "
      f"F1 {fm(float(T13_cost.loc[opt_i,'F1']))}, "
      f"exp. cost/sample {fm(float(T13_cost.loc[opt_i,'exp_cost_per_sample']))})")
print()
print("--- T13c: TIERED EARLY-WARNING SCORE (calibrated prob tiers) ---")
print(rnd(T13_tier).to_string(index=False))
print()
print("--- T14: VARIANCE DECOMPOSITION (test pool 2015-2025; OLS Type-II SS) ---")
print(rnd(T14).to_string(index=False))
print()
print("--- T15: PER-BLOCK 90% CONFORMAL COVERAGE (marginal intervals) ---")
print(rnd(T15).to_string(index=False))
print()
print("--- T16a: SEED STABILITY (best classifier, uncalibrated rolling-origin) ---")
print(rnd(T16_seed).to_string(index=False))
print(f"ROC_AUC mean +/- sd: {fm(seed_summary['ROC_AUC_mean'])} +/- {fm(seed_summary['ROC_AUC_sd'])}; "
      f"PR_AUC mean +/- sd: {fm(seed_summary['PR_AUC_mean'])} +/- {fm(seed_summary['PR_AUC_sd'])}")
print()
print("--- T16b: EXTENDED THETA SENSITIVITY (best classifier) ---")
print(rnd(T16_theta).to_string(index=False))
print()
print(f"Top-6 regression features ({reg_kind}): "
      + "; ".join(f"{k}={v:.4f}" for k, v in rank_reg.head(6).items()))
print(f"Top-6 classification features ({clf_kind}): "
      + "; ".join(f"{k}={v:.4f}" for k, v in rank_clf.head(6).items()))
if note_reg:
    print("NOTE (regression importance):", note_reg)
if note_clf:
    print("NOTE (classification importance):", note_clf)
if shap_fig_note:
    print("NOTE (SHAP figure):", shap_fig_note)
print()
print(line)
print("V1 -> V2 CLASSIFICATION HEADLINE (theta = 0.20, pooled rolling-origin)")
print(line)
print(f"{'':28s}{'ROC_AUC':>9s}{'PR_AUC':>9s}{'Recall':>9s}{'MCC':>9s}")
print(f"{'v1 LogisticRegression':28s}{0.767:>9.3f}{0.722:>9.3f}{0.70:>9.3f}{0.42:>9.3f}")
print(f"{'v2 ' + best9_name:28s}{best9_metrics['ROC_AUC']:>9.3f}{best9_metrics['PR_AUC']:>9.3f}"
      f"{best9_metrics['Recall']:>9.3f}{best9_metrics['MCC']:>9.3f}")
print(f"(v2 row: tuned + calibrated, {len(FEATURES)}-feature set; "
      f"v2 also reports Brier {fm(best9_brier_cal)} and Recall@Prec>=0.60 = "
      f"{fm(float(T9.loc[T9['Model'] == best9_name, 'Recall_at_Prec60'].iloc[0]))})")
print()
print("LEAKAGE CHECK: PASS (re-confirmed above; every v2 feature covered by corruption test)")
print()
print("Files written:")
for name in tables:
    print("  ", os.path.join(TAB, f"{name}.csv"))
print("  ", os.path.join(TAB, "importance_regression.csv"))
print("  ", os.path.join(TAB, "importance_classification.csv"))
print("  ", docx_path)
for f in ["F1_rolling_origin_schematic.png", "F2_pred_vs_actual_conformal.png",
          "F3_skill_by_year.png", "F4_importance_regression.png",
          "F4b_importance_classification.png", "F5_confusion_and_roc_pr.png",
          "F6_ablation.png", "F7_calibration_reliability.png", "F8_decision_curve.png",
          "F9_shap_beeswarm_classifier.png", "F9b_shap_dependence_top_feature.png",
          "F10_multihorizon_skill.png", "F11_lobo_block_auc.png",
          "F12_variance_decomposition.png", "F13_roc_pr_calibrated_best.png"]:
    print("  ", os.path.join(FIG, f))
print(line)
