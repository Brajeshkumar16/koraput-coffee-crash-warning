"""Data provenance and plausibility checks for the revision (Reviewer 2, point A2; Reviewer 1, points 2.6 and 3.2).

Writes outputs/tables/P01..P05. Usage: python provenance.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import ewcore as ew

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ew.DATA / "source_spreadsheet_as_supplied.xlsx"
RECEIVED_AS = "Koraput Coffee Data Set- SO Agriculture Koraput .xlsx"  # file name as received by the authors


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    print("wrote", name)


def main():
    src = pd.read_excel(SOURCE)
    yearly = ew.load_yearly_means()
    panel = ew.load_panel()

    # P01: derivation chain
    g = src.groupby(["Block", "Year"]).agg(n=("Production_MT", "size"), P=("Production_MT", "mean"), A=("Area_Ha", "mean")).reset_index()
    m = yearly.merge(g, on=["Block", "Year"])
    steps = pd.DataFrame([
        {"step": "source spreadsheet as supplied", "rows": len(src), "detail": f"file name '{RECEIVED_AS}'; columns {', '.join(src.columns)}"},
        {"step": "rows per block-year in the source", "rows": int(g.n.sum()),
         "detail": "; ".join(f"{k} row(s): {v} block-years" for k, v in g.n.value_counts().sort_index().items())},
        {"step": "yearly sheet = mean of the source rows of each block-year", "rows": len(yearly),
         "detail": f"production max |difference| {np.abs(m['Production MT'] - m.P).max():.6f}; area {np.abs(m['Area Ha'] - m.A).max():.6f}"},
        {"step": "block-years with no source row", "rows": 200 - len(yearly), "detail": "filled in the analysis file (see P02)"},
        {"step": f"analysis file {ew.PANEL_FILE} (final_clustered_data.csv in the original submission)", "rows": len(panel),
         "detail": "ratios recomputed as production/area, export/production and production/farmers"},
        {"step": "non-integer production values in the analysis file", "rows": int((panel["Production MT"] % 1 != 0).sum()),
         "detail": "means of 2 to 7 integer source rows, or interpolated values"},
    ])
    save(steps, "P01_derivation_chain")

    # P02: filled block-years
    real = set(zip(yearly.Block, yearly.Year))
    d = panel.set_index(["Block", "Year"])
    rows = []
    for (b, y), r in d.iterrows():
        if (b, y) in real:
            continue
        cols = ["Production MT", "Export MT", "Area Ha"]
        if (b, y - 1) in d.index and (b, y + 1) in d.index:
            mid = all(abs((d.loc[(b, y - 1), c] + d.loc[(b, y + 1), c]) / 2 - r[c]) < 1e-6 for c in cols)
            how = "linear interpolation (midpoint of neighbours)" if mid else "interior, not a midpoint"
        else:
            nb = (b, y - 1) if (b, y - 1) in d.index else (b, y + 1)
            if all(abs(d.loc[nb, c] - r[c]) < 1e-9 for c in cols + ["Farmers"]):
                how = f"copy of {nb[1]} (all variables identical)"
            else:
                nb2 = (b, y - 2) if nb[1] < y else (b, y + 2)
                lin = all(abs(2 * d.loc[nb, c] - d.loc[nb2, c] - r[c]) < 1e-6 for c in cols) if nb2 in d.index else False
                how = "linear extrapolation from the two nearest years" if lin else "edge year, rule not identified"
        rows.append({"block": b, "year": y, "how_filled": how, "farmers": r.Farmers})
    save(pd.DataFrame(rows), "P02_filled_block_years")

    # P03: distribution and association checks on the source rows
    s = src.copy()
    s["export_share"] = s.Export_MT / s.Production_MT
    s["farmers_per_ha"] = s.Farmers / s.Area_Ha
    rows = []
    for c, lab in (("Production_MT", "production (MT)"), ("Area_Ha", "area (ha)"), ("export_share", "export / production"),
                   ("farmers_per_ha", "farmers per ha")):
        x = s[c].to_numpy(float)
        rows.append({"variable": lab, "n": len(x), "min": x.min(), "max": x.max(), "mean": x.mean(),
                     "uniform_mean": (x.min() + x.max()) / 2, "ks_p_vs_uniform": stats.kstest(x, "uniform", args=(x.min(), x.max() - x.min())).pvalue})
    save(pd.DataFrame(rows), "P03_source_distributions")
    rows = [{"pair": "production vs area (source rows)", "pearson_r": stats.pearsonr(s.Production_MT, s.Area_Ha)[0]},
            {"pair": "production vs farmers (source rows)", "pearson_r": stats.pearsonr(s.Production_MT, s.Farmers)[0]},
            {"pair": "area vs farmers (source rows)", "pearson_r": stats.pearsonr(s.Area_Ha, s.Farmers)[0]},
            {"pair": "yield vs productivity per farmer (analysis file)", "pearson_r": stats.pearsonr(panel["Yield per Ha"], panel["Productivity per Farmer"])[0]}]
    for c in ("Production_MT", "Area_Ha", "Yield_per_Ha"):
        for f in ("Block", "Year"):
            groups = [v[c].to_numpy() for _, v in s.groupby(f)]
            F, p = stats.f_oneway(*groups)
            ssb = sum(len(v) * (v.mean() - s[c].mean()) ** 2 for v in groups)
            rows.append({"pair": f"share of {c} variance explained by {f}", "pearson_r": np.nan, "share": ssb / ((s[c] - s[c].mean()) ** 2).sum(),
                         "chance_expectation": (len(groups) - 1) / (len(s) - 1), "anova_p": p})
    save(pd.DataFrame(rows), "P04_associations")

    # P05: magnitudes against the Coffee Board (Odisha, all of the state)
    dist = panel.groupby("Year").agg(production_mt=("Production MT", "sum"), area_ha=("Area Ha", "sum")).reset_index()
    cb = pd.read_csv(ew.DATA / "coffee_board_odisha.csv")
    od = cb[["season", "final_production_mt"]].dropna().rename(columns={"final_production_mt": "final"})
    oa = cb[["season", "planted_area_ha"]].dropna()
    rows = [{"quantity": "Koraput panel, sum of 10 blocks, production (MT) per year", "min": dist.production_mt.min(), "max": dist.production_mt.max(),
             "median": dist.production_mt.median(), "years": f"{dist.Year.min()}-{dist.Year.max()}"},
            {"quantity": "Koraput panel, sum of 10 blocks, area (ha) per year", "min": dist.area_ha.min(), "max": dist.area_ha.max(),
             "median": dist.area_ha.median(), "years": f"{dist.Year.min()}-{dist.Year.max()}"},
            {"quantity": "Coffee Board, Odisha state, final production (MT)", "min": od.final.min(), "max": od.final.max(),
             "median": od.final.median(), "years": f"{od.season.min()} to {od.season.max()}"},
            {"quantity": "Coffee Board, Odisha state, planted area (ha)", "min": oa.planted_area_ha.min(), "max": oa.planted_area_ha.max(),
             "median": oa.planted_area_ha.median(), "years": f"{oa.season.min()} to {oa.season.max()}"},
            {"quantity": "Koraput panel yield (t/ha), block-years", "min": panel["Yield per Ha"].min(), "max": panel["Yield per Ha"].max(),
             "median": panel["Yield per Ha"].median(), "years": "2006-2025"},
            {"quantity": "Coffee Board Odisha productivity (t/ha) = final / planted area", "min": np.nan, "max": np.nan,
             "median": od.final.median() / oa.planted_area_ha.median(), "years": "medians"}]
    p5 = pd.DataFrame(rows)
    p5["ratio_panel_to_state_median"] = np.nan
    p5.loc[0, "ratio_panel_to_state_median"] = dist.production_mt.median() / od.final.median()
    p5.loc[1, "ratio_panel_to_state_median"] = dist.area_ha.median() / oa.planted_area_ha.median()
    save(p5, "P05_magnitudes_vs_coffee_board")


if __name__ == "__main__":
    main()
