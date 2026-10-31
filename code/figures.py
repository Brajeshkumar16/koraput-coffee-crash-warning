"""Figures for the revised paper 2, from outputs/tables (run reanalysis.py and provenance.py first).

Fig 1  rolling-origin schematic: unchanged from the original submission (copied).
Fig 2  variance decomposition (residual now has a legend patch; labels outside the bars).
Fig 3  ROC and precision-recall curves of the uncalibrated model tuned per fold and of the index (replaces the calibrated-model figure).
Fig 4  observed crash rate by risk tier with Wilson intervals (replaces the reliability curve).
Fig 5  decision curve of the uncalibrated model (replaces the calibrated version).
Fig 6  null distributions beside each observed AUC (new).
Fig 7  the same test on the Coffee Board's public zone series (new).
Usage: python figures.py
"""
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

import ewcore as ew  # noqa: E402

HERE = Path(__file__).resolve().parent
T = HERE / "outputs" / "tables"
F = HERE / "outputs" / "figures"
F.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#52514e", "#e6e4df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": SURF})


def save(fig, name):
    fig.savefig(F / f"{name}.png")
    plt.close(fig)
    print("written", name)


def fig1():
    src = ew.DATA / "original_submission" / "F1_rolling_origin_schematic.png"
    shutil.copy(src, F / "Fig1_rolling_origin.png")
    print("copied Fig1_rolling_origin")


def fig2():
    v = pd.read_csv(T / "T13_variance_decomposition.csv")
    v = v[v["sample"] == "all samples, test pool"].set_index("target")
    labels = {"y_yield": "next-season yield", "y_change": "season-to-season change"}
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    y = np.arange(len(v))
    left = np.zeros(len(v))
    for col, colr, lab in (("block_pct", BLUE, "block"), ("season_pct", ORANGE, "season"), ("residual_pct", GRAY, "residual")):
        ax.barh(y, v[col], left=left, color=colr, height=0.5, label=lab, edgecolor=SURF, linewidth=1.5)
        left += v[col].to_numpy()
    for yi, (_, r) in zip(y, v.iterrows()):
        ax.text(101, yi, f"residual {r.residual_pct:.0f}%", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, [labels[i] for i in v.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the sum of squares, %  (two-way fixed effects, Type II, test pool n = 110)")
    ax.set_title("Variance decomposition under a block + season model")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3)
    ax.grid(axis="y", visible=False)
    save(fig, "Fig2_variance_decomposition")


def fig3():
    p = pd.read_csv(T / "pred_logreg_perfold.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for score, colr, lab in (("prob", BLUE, "logistic model, tuned within each fold"), ("index", ORANGE, "mean-reversion index")):
        fpr, tpr, _ = roc_curve(p.actual, p[score])
        a.plot(fpr, tpr, color=colr, lw=2, label=f"{lab} (AUC {roc_auc_score(p.actual, p[score]):.3f})")
        pr, rc, _ = precision_recall_curve(p.actual, p[score])
        b.plot(rc, pr, color=colr, lw=2, label=f"{lab} (AP {average_precision_score(p.actual, p[score]):.3f})")
    a.plot([0, 1], [0, 1], color=GRAY, lw=1)
    b.axhline(p.actual.mean(), color=GRAY, lw=1)
    a.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC, test seasons 2015-2025 (n = 110)")
    b.set(xlabel="recall", ylabel="precision", title="Precision-recall")
    a.legend(loc="lower right", fontsize=7.5)
    b.legend(loc="lower left", fontsize=7.5)
    save(fig, "Fig3_roc_pr_uncalibrated")


def fig4():
    t = pd.read_csv(T / "T10_risk_tiers.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    order = ["Low", "Medium", "High"]
    for k, (src, colr, dx) in enumerate((("published (calibrated model)", GRAY, -0.12), ("uncalibrated model tuned per fold", BLUE, 0.12))):
        g = t[t.probabilities == src].set_index("tier").reindex(order).dropna(subset=["n"])
        x = np.array([order.index(i) for i in g.index]) + dx
        ax.errorbar(x, g.observed_rate, yerr=[g.observed_rate - g.wilson_lo, g.wilson_hi - g.observed_rate], fmt="o", color=colr,
                    ms=6, capsize=3, lw=1.5, label=src)
        for xi, (_, r) in zip(x, g.iterrows()):
            ax.annotate(f"{int(r.crashes)}/{int(r.n)}", (xi, r.wilson_hi), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), ["low (< 0.33)", "medium (0.33-0.66)", "high (> 0.66)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("observed crash rate (Wilson 95% interval)")
    ax.set_title("Risk tiers: observed crash rates and their uncertainty")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "Fig4_risk_tiers")


def fig5():
    nb = pd.read_csv(T / "T11a_net_benefit_uncalibrated.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(nb.threshold, nb.net_benefit_model, color=BLUE, lw=2, marker="o", ms=4, label="logistic model, uncalibrated, tuned per fold")
    ax.plot(nb.threshold, nb.net_benefit_warn_all, color=GRAY, lw=2, label="warn every block")
    ax.axhline(0, color=INK2, lw=1, label="warn none")
    ax.set(xlabel="threshold probability", ylabel="net benefit", title="Decision curve (illustrative; inherits the label artefact)")
    ax.set_ylim(min(-0.1, nb.net_benefit_model.min() - 0.02), max(nb.net_benefit_warn_all.max(), nb.net_benefit_model.max()) + 0.03)
    ax.legend(loc="upper right")
    save(fig, "Fig5_decision_curve_uncalibrated")


def fig6():
    n = pd.read_csv(T / "T03_nulls.csv")
    keep = ["index, one season ahead", "logistic model (tuned per fold), one season ahead", "index, two seasons ahead (crash at t+2 relative to Y_t)",
            "logistic model, two seasons ahead", "leave-one-block-out, earlier seasons only", "index, 2024-25 holdout"]
    n = n[n.analysis.isin(keep) & (n.null == "within-block shuffle")].set_index("analysis").reindex(keep).dropna(subset=["observed_auc"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(n))
    for yi, (_, r) in zip(y, n.iterrows()):
        ax.plot([r.null_q025, r.null_q975], [yi, yi], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(r.null_mean, yi, color=GRAY, s=36, zorder=2)
        ax.scatter(r.observed_auc, yi, color=BLUE, s=46, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.scatter([], [], color=GRAY, s=36, label="shuffled-season null: mean")
    ax.plot([], [], color=GRID, lw=6, label="null: 95% range")
    ax.scatter([], [], color=BLUE, s=46, label="observed")
    ax.axvline(0.5, color=INK2, lw=0.8)
    ax.set_yticks(y, [a.replace(" (crash at t+2 relative to Y_t)", "").replace(" (tuned per fold)", "") for a in n.index])
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Every observed AUC lies inside its shuffled-season null")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3)
    ax.grid(axis="y", visible=False)
    save(fig, "Fig6_nulls")


def fig8():
    x = pd.read_csv(T / "X04_detectability.csv")
    a = x[x.design == "AR(1) persistence"]
    b = x[x.design == "predictor known at the origin"]
    ac = pd.read_csv(T / "X02_persistence_and_biennial.csv").iloc[0]
    fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"wspace": 0.45})
    p1.plot(a.value, a.power_origin_referenced_label, color=BLUE, lw=2, marker="o", ms=4)
    p1.axhline(0.8, color=GRAY, lw=1)
    p1.axvspan(max(ac.ci_lo, 0), ac.ci_hi, color=ORANGE, alpha=0.15, lw=0)
    p1.axvline(ac.estimate, color=ORANGE, lw=1.5)
    p1.set(xlabel="lag-one persistence (phi)", ylabel="power of a 5% test", title="A  Persistence", ylim=(0, 1.02))
    p1.annotate("observed (95% CI)", (ac.ci_hi, 0.08), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2)
    p2.plot(b.value, b.power_origin_referenced_label, color=BLUE, lw=2, marker="o", ms=4)
    p2.axhline(0.8, color=GRAY, lw=1)
    p2.set(xlabel="share of next-season variance explained", title="B  Predictor at the origin", ylim=(0, 1.02))
    p3.plot(a.value, a.mean_auc_published_label, color=BLUE, lw=2, marker="o", ms=4)
    p3.set(xlabel="lag-one persistence (phi)", ylabel="mean ROC-AUC", title="C  Original crash label", ylim=(0.5, 0.9))
    for ax in (p1, p2, p3):
        ax.grid(True)
    save(fig, "Fig8_detectability")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig8):
        try:
            f()
        except FileNotFoundError as e:
            print(f"{f.__name__} skipped: {e.filename} missing")
