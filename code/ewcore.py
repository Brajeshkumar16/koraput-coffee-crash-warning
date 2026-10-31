"""Core pieces of the paper-2 pipeline, copied from earlywarning_v2_koraput.py (which runs on import)
and parametrised by the input panel, so the same code can be run on the full panel, on the panel
without filled block-years, on shuffled panels and on simulated panels.

Changes from the published pipeline are marked CHANGED and listed in the response to reviewers.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
PANEL_FILE = "koraput_panel_analysis_file_flagged.csv"
# input files: data/ beside code/ in the deposit, deposit/data/ in the working folder
_DATA_DIRS = (HERE.parent / "data", HERE / "deposit" / "data")
DATA = next((p for p in _DATA_DIRS if (p / PANEL_FILE).exists()), _DATA_DIRS[0])
SEED = 42
TEST_YEARS = list(range(2015, 2026))  # target seasons, 11 folds
H2_YEARS = list(range(2017, 2026))

V1_FEATURES = ["Yield_t", "Production_t", "Export_t", "Area_t", "Farmers_t", "ExportRatio_t", "Productivity_t",
               "Yield_lag1", "Yield_lag2", "Production_lag1", "Export_lag1", "Yield_yoy", "Yield_yoy_pct",
               "Yield_roll3_mean", "Yield_roll3_slope", "Farmer_to_Area", "Block_hist_mean", "Spatial_LOO_mean", "Year"]
NEW_FEATURES = ["Yield_dev_clim", "Yield_dev_clim_pct", "Yield_z_clim", "Yield_roll3_std", "Yield_cv3", "ExportRatio_yoy",
                "Area_growth", "Farmers_growth", "Decline_run", "Down_years_3", "Spatial_LOO_yoy"]
FEATURES = V1_FEATURES + NEW_FEATURES
INDEX = "Yield_dev_clim_pct"  # the mean-reversion index: (Y_t - mean(Y_<=t)) / mean(Y_<=t)
LOGREG_GRID = [(C, p) for p in ("l1", "l2") for C in (0.03, 0.1, 0.3, 1.0, 3.0)]


def load_panel(path=None):
    raw = pd.read_csv(path or DATA / PANEL_FILE)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")] + ["Cluster", "filled", "fill_method", "copied_row"],
                   errors="ignore")
    raw["Year"] = raw["Year"].astype(int)
    return raw.sort_values(["Block", "Year"]).reset_index(drop=True)


def load_yearly_means():
    """Block-years present in the source spreadsheet, as means of their rows (182 of the 200 block-years)."""
    return pd.read_csv(DATA / "koraput_yearly_means.csv")


def build_features_one(h_all, block, t):
    """All engineered features for (block, t) from rows with Year <= t only (copied unchanged)."""
    h = h_all[h_all["Year"] <= t]
    hb = h[h["Block"] == block].sort_values("Year")

    def v(year, col):
        s = hb.loc[hb["Year"] == year, col]
        return float(s.iloc[0]) if len(s) else np.nan

    f = {"Yield_t": v(t, "Yield per Ha"), "Production_t": v(t, "Production MT"), "Export_t": v(t, "Export MT"),
         "Area_t": v(t, "Area Ha"), "Farmers_t": v(t, "Farmers"), "ExportRatio_t": v(t, "Export Ratio"),
         "Productivity_t": v(t, "Productivity per Farmer"), "Yield_lag1": v(t - 1, "Yield per Ha"),
         "Yield_lag2": v(t - 2, "Yield per Ha"), "Production_lag1": v(t - 1, "Production MT"), "Export_lag1": v(t - 1, "Export MT")}
    f["Yield_yoy"] = f["Yield_t"] - f["Yield_lag1"]
    f["Yield_yoy_pct"] = (f["Yield_t"] - f["Yield_lag1"]) / f["Yield_lag1"]
    win = np.array([f["Yield_lag2"], f["Yield_lag1"], f["Yield_t"]])
    f["Yield_roll3_mean"] = float(np.mean(win))
    f["Yield_roll3_slope"] = float(np.polyfit(np.array([t - 2, t - 1, t], float), win, 1)[0]) if not np.isnan(win).any() else np.nan
    f["Farmer_to_Area"] = f["Farmers_t"] / f["Area_t"]
    f["Block_hist_mean"] = float(hb["Yield per Ha"].mean())
    oth = h[(h["Year"] == t) & (h["Block"] != block)]
    f["Spatial_LOO_mean"] = float(oth["Yield per Ha"].mean())
    f["Year"] = float(t)
    yb = hb["Yield per Ha"].to_numpy(float)
    bh_mean, bh_std = f["Block_hist_mean"], (float(np.std(yb)) if len(yb) >= 2 else 0.0)
    f["Yield_dev_clim"] = f["Yield_t"] - bh_mean
    f["Yield_dev_clim_pct"] = (f["Yield_t"] - bh_mean) / bh_mean
    f["Yield_z_clim"] = (f["Yield_t"] - bh_mean) / bh_std if bh_std > 0 else 0.0
    f["Yield_roll3_std"] = float(np.std(win))
    f["Yield_cv3"] = f["Yield_roll3_std"] / f["Yield_roll3_mean"]
    f["ExportRatio_yoy"] = v(t, "Export Ratio") - v(t - 1, "Export Ratio")
    f["Area_growth"] = (v(t, "Area Ha") - v(t - 1, "Area Ha")) / v(t - 1, "Area Ha")
    f["Farmers_growth"] = (v(t, "Farmers") - v(t - 1, "Farmers")) / v(t - 1, "Farmers")
    diffs = np.diff(yb)
    run = 0
    for d in diffs[::-1]:
        if d < 0:
            run += 1
        else:
            break
    f["Decline_run"] = float(run)
    last3 = diffs[-3:] if len(diffs) else np.array([])
    f["Down_years_3"] = float((last3 < 0).sum())
    oth_t = h[(h["Year"] == t) & (h["Block"] != block)].set_index("Block")["Yield per Ha"]
    oth_t1 = h[(h["Year"] == t - 1) & (h["Block"] != block)].set_index("Block")["Yield per Ha"]
    f["Spatial_LOO_yoy"] = float((oth_t - oth_t1).mean())
    return f


def build_samples(raw):
    """One row per (block, base year t) with features, next-season targets and the three baselines."""
    lut = raw.set_index(["Block", "Year"])
    rows = []
    for b in sorted(raw.Block.unique()):
        for t in range(2008, 2025):
            if (b, t) not in lut.index or (b, t + 1) not in lut.index:
                continue
            f = build_features_one(raw, b, t)
            if any(np.isnan(list(f.values()))):
                continue
            hb = raw[(raw.Block == b) & (raw.Year <= t)]
            f.update(Block=b, base_year=t, target_year=t + 1, y_yield=float(lut.loc[(b, t + 1), "Yield per Ha"]),
                     y_prod=float(lut.loc[(b, t + 1), "Production MT"]))
            for col, tag in (("Yield per Ha", "yield"), ("Production MT", "prod")):
                ys = hb[col].to_numpy(float)
                f[f"bl_pers_{tag}"] = float(lut.loc[(b, t), col])
                f[f"bl_clim_{tag}"] = float(ys.mean())
                b1, b0 = np.polyfit(hb.Year.to_numpy(float), ys, 1)
                f[f"bl_trend_{tag}"] = float(b1 * (t + 1) + b0)
            f["y_yield_h2"] = float(lut.loc[(b, t + 2), "Yield per Ha"]) if (b, t + 2) in lut.index else np.nan
            rows.append(f)
    s = pd.DataFrame(rows).reset_index(drop=True)
    return add_labels(s)


def add_labels(s, theta=0.20):
    """Crash labels. 'cur' is the published definition; 'clim' and 'below' are referenced to the origin's expectation."""
    s = s.copy()
    s["y_crash"] = (s.y_yield <= (1 - theta) * s.Yield_t).astype(int)             # published: relative to Y_t
    s["y_crash_clim"] = (s.y_yield <= (1 - theta) * s.Block_hist_mean).astype(int)  # relative to expanding mean at t
    s["y_below"] = (s.y_yield < s.Block_hist_mean).astype(int)                     # below-normal season
    s["y_crash_h2"] = np.where(s.y_yield_h2.notna(), (s.y_yield_h2 <= (1 - theta) * s.Yield_t).astype(float), np.nan)
    s["target_year_h2"] = s.base_year + 2
    return s


def make_logreg(C=1.0, penalty="l2", seed=SEED):
    solver = "saga" if penalty == "l1" else "lbfgs"
    return Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(C=C, penalty=penalty, solver=solver, class_weight="balanced",
                                                                       max_iter=20000, random_state=seed))])


def tune_logreg(tr, ycol, tycol):
    """Inner rolling-origin tuning on the last (up to) 3 training target years (copied unchanged)."""
    yrs = sorted(tr[tycol].unique())
    inner = [y for y in yrs if y >= yrs[-1] - 2]
    best, best_score = (1.0, "l2"), -np.inf
    for C, p in LOGREG_GRID:
        scores = []
        for Yi in inner:
            itr, iva = tr[tr[tycol] <= Yi - 1], tr[tr[tycol] == Yi]
            if itr[ycol].nunique() < 2 or iva[ycol].nunique() < 2 or len(itr) < 20:
                continue
            m = make_logreg(C, p).fit(itr[FEATURES].to_numpy(), itr[ycol].to_numpy())
            scores.append(average_precision_score(iva[ycol].to_numpy(), m.predict_proba(iva[FEATURES].to_numpy())[:, 1]))
        if scores and float(np.mean(scores)) > best_score:
            best_score, best = float(np.mean(scores)), (C, p)
    return best


def rolling_logreg(s, ycol="y_crash", tycol="target_year", test_years=TEST_YEARS, params="per_fold"):
    """Uncalibrated tuned logistic regression under rolling origin.

    params='per_fold': tuned inside each training window (CHANGED: the leakage-free version).
    params=(C, penalty): fixed settings for every fold (the published run used the modal per-fold choice).
    Returns (predictions, list of per-fold settings)."""
    parts, chosen = [], []
    for Y in test_years:
        tr, te = s[s[tycol] <= Y - 1], s[s[tycol] == Y]
        if te.empty or tr[ycol].nunique() < 2:
            continue
        C, p = tune_logreg(tr, ycol, tycol) if params == "per_fold" else params
        chosen.append((C, p))
        m = make_logreg(C, p).fit(tr[FEATURES].to_numpy(), tr[ycol].to_numpy())
        parts.append(pd.DataFrame({"prob": m.predict_proba(te[FEATURES].to_numpy())[:, 1], "actual": te[ycol].to_numpy().astype(int),
                                   "target_year": te[tycol].to_numpy(), "Block": te.Block.to_numpy(),
                                   "index": te[INDEX].to_numpy()}, index=te.index))
    return pd.concat(parts), chosen


def make_reg_models():
    return {"Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=SEED))]),
            "Lasso": Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.1, random_state=SEED, max_iter=50000))]),
            "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
            "HistGB": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=400, min_samples_leaf=5,
                                                    l2_regularization=1.0, early_stopping=False, random_state=SEED),
            "XGBoost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                    reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=0)}


def rolling_regression(s, target="y_yield"):
    out = {n: [] for n in make_reg_models()}
    for Y in TEST_YEARS:
        tr, te = s[s.target_year <= Y - 1], s[s.target_year == Y]
        for n, m in make_reg_models().items():
            m.fit(tr[FEATURES].to_numpy(), tr[target].to_numpy())
            out[n].append(pd.DataFrame({"pred": m.predict(te[FEATURES].to_numpy()), "actual": te[target].to_numpy(),
                                        "target_year": te.target_year.to_numpy(), "Block": te.Block.to_numpy()}, index=te.index))
    return {n: pd.concat(v) for n, v in out.items()}
