"""Write outputs/results_summary.md from the re-analysis tables, for the check-in before drafting."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T = HERE / "outputs" / "tables"


def r(name):
    return pd.read_csv(T / f"{name}.csv")


def md(df, digits=3):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    return "\n".join([head, "|" + "|".join("---" for _ in df.columns) + "|"] + ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)])


def main():
    L = ["# Paper 2 revision: re-analysis results", ""]
    L += ["## Published numbers reproduced", "", md(r("T01_headline_reproduction")), ""]
    L += ["## Reviewer 2's checks", "", md(r("T02_reviewer2_checks")), ""]
    n = r("T03_nulls")
    L += ["## Observed AUC against shuffled-season and random-series nulls", "",
          md(n[["analysis", "null", "observed_auc", "null_mean", "null_q025", "null_q975", "share_null_ge_observed", "null_reps", "n", "prevalence"]]), ""]
    L += ["## Alternative crash definitions", "", md(r("T04_alternative_definitions")), ""]
    L += ["## 2024-25 holdout", "", md(r("T05_holdout_2024_25")), ""]
    L += ["## Threshold sensitivity", "", md(r("T06_threshold_sensitivity")), ""]
    L += ["## Filled block-years excluded", "", md(r("T07_filled_excluded")), ""]
    L += ["## Cluster bootstrap and the AUC difference", "", md(r("T08_cluster_bootstrap")), ""]
    L += ["## Per-season AUC", "", md(r("T09_per_season")), ""]
    L += ["## Risk tiers with Wilson intervals", "", md(r("T10_risk_tiers")), ""]
    L += ["## Cost ratios (uncalibrated model)", "", md(r("T11b_cost_ratios")), ""]
    L += ["## Regression benchmark (climatology prespecified as reference)", "", md(r("T12_regression_benchmark")), ""]
    L += ["## Variance decomposition", "", md(r("T13_variance_decomposition")), ""]
    L += ["## Conformal intervals", "", md(r("T14a_conformal")), "", md(r("T14b_conformal_per_block")), ""]
    L += ["## Data provenance", "", md(r("P01_derivation_chain")), "", md(r("P02_filled_block_years")), "",
          md(r("P03_source_distributions"), 4), "", md(r("P04_associations"), 4), "", md(r("P05_magnitudes_vs_coffee_board"), 2), ""]
    (HERE / "outputs" / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written outputs/results_summary.md")


if __name__ == "__main__":
    main()
