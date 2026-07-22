"""
Reproducible figure generation for the Cardionix research write-up.

Every figure is rendered from the real data reported in the paper:
  - CVD mortality  : WHO / GBD 2019, 2021, IHME GBD 2022.
  - Corpus growth  : record counts from data/*/annotation.csv.
  - Taxonomy pivot : macro-F1 from W&B runs (5-class vs 3-class).
  - Training curve : macro precision/recall/F1 parsed from checkpoint filenames.
  - Per-class F1   : validation report of the best run (sfqx9n2b).

Run:  python3 docs/make_figures.py
Output:  docs/assets/*.png
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

# A restrained, academic palette.
INK = "#1b1b1f"
MUTED = "#6b6f76"
ACCENT = "#b23a48"      # clinical red
ACCENT2 = "#2f6690"     # slate blue
ACCENT3 = "#3a7d44"     # green
GRID = "#d9dbe0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# ----------------------------------------------------------------------------
# 1. Global CVD mortality: absolute burden rises while the rate falls.
# ----------------------------------------------------------------------------
def fig_cvd():
    years = np.array([1990, 2000, 2010, 2019, 2021, 2022])
    # Absolute CVD deaths, millions (GBD 2019/2021 + IHME GBD 2022 update).
    deaths = np.array([12.1, 14.4, 16.4, 18.6, 19.42, 19.8])
    # Age-standardised death rate, per 100k (GBD 2021 endpoints, interpolated).
    asdr = np.array([358.1, 330.0, 295.0, 245.0, 235.2, 233.0])

    pos = np.arange(len(years))
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    bars = ax1.bar(pos, deaths, width=0.62, color=ACCENT, alpha=0.85,
                   label="Absolute CVD deaths (millions)")
    ax1.set_ylabel("Absolute CVD deaths (millions)", color=ACCENT)
    ax1.set_ylim(0, 23)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    for x, y in zip(pos, deaths):
        ax1.text(x, y + 0.35, f"{y:.1f}", ha="center", va="bottom",
                 fontsize=9, color=ACCENT, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(pos, asdr, "-o", color=ACCENT2, lw=2.2, ms=6,
             label="Age-standardised death rate (per 100k)")
    ax2.set_ylabel("Age-standardised rate (per 100k)", color=ACCENT2)
    ax2.set_ylim(150, 400)
    ax2.tick_params(axis="y", labelcolor=ACCENT2)

    ax1.set_title("Global cardiovascular mortality, 1990–2022\n"
                  "absolute burden +60% while the age-standardised rate falls −34%",
                  fontsize=12, loc="left")
    ax1.set_xlabel("Year")
    ax1.set_xticks(pos)
    ax1.set_xticklabels([str(y) for y in years])
    lines = [bars, ax2.lines[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=False, fontsize=9)
    ax1.text(0.0, -0.20,
             "Sources: WHO CVD fact sheet; GBD 2019 (Roth et al., JACC 2020); "
             "GBD 2021 (JACC 2024); IHME GBD 2022 update.",
             transform=ax1.transAxes, fontsize=7.5, color=MUTED)
    save(fig, "fig_cvd_mortality.png")


# ----------------------------------------------------------------------------
# 2. Corpus growth across three generations.
# ----------------------------------------------------------------------------
def fig_corpus():
    gens = ["DHDataset\n(PASCAL / Kaggle)", "PhysioNet\nCinC 2016", "CardionixDataset\n(merged)"]
    totals = [585, 3240, 3825]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    axa.bar(gens, totals, color=[MUTED, ACCENT2, ACCENT3], alpha=0.9)
    for i, v in enumerate(totals):
        axa.text(i, v + 60, f"{v:,}", ha="center", fontweight="bold", fontsize=10)
    axa.set_ylabel("Labelled recordings")
    axa.set_title("Corpus scale: a ~6.5× expansion\n"
                  "the data-centric pivot, in numbers", fontsize=11, loc="left")
    axa.set_ylim(0, 4300)

    # Final class distribution (CardionixDataset).
    classes = ["normal", "abnormal", "artifact"]
    counts = [2926, 859, 40]
    colors = [ACCENT3, ACCENT, MUTED]
    axb.barh(classes[::-1], counts[::-1], color=colors[::-1], alpha=0.9)
    for i, v in enumerate(counts[::-1]):
        axb.text(v + 40, i, f"{v:,}", va="center", fontweight="bold", fontsize=10)
    axb.set_xlabel("Recordings")
    axb.set_title("Final class distribution\n"
                  "residual imbalance, artifact is scarce", fontsize=11, loc="left")
    axb.set_xlim(0, 3300)
    save(fig, "fig_corpus_growth.png")


# ----------------------------------------------------------------------------
# 3. Taxonomy pivot: 5-class collapse vs 3-class recovery.
# ----------------------------------------------------------------------------
def fig_taxonomy():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.8, 4.3),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # Left: macro-F1 by taxonomy.
    labels = ["5-class\n(run A)", "5-class\n(run B)", "3-class\n(best)"]
    f1 = [0.593, 0.512, 0.863]
    bars = axa.bar(labels, f1, color=[MUTED, MUTED, ACCENT3], alpha=0.9)
    for i, v in enumerate(f1):
        axa.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    axa.axhline(0.863, ls="--", lw=1, color=ACCENT3, alpha=0.6)
    axa.set_ylim(0, 1.0)
    axa.set_ylabel("Validation macro-F1")
    axa.set_title("Collapsing the taxonomy recovered +0.27 macro-F1",
                  fontsize=11, loc="left")

    # Right: per-class val F1 under the 5-class scheme (the collapse).
    cls = ["normal", "murmur", "artifact", "extrahls", "extrastole"]
    val_f1 = [0.813, 1.000, 0.642, 0.286, 0.222]
    colors = [ACCENT3, ACCENT2, MUTED, ACCENT, ACCENT]
    axb.barh(cls[::-1], val_f1[::-1], color=colors[::-1], alpha=0.9)
    for i, v in enumerate(val_f1[::-1]):
        axb.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    axb.set_xlim(0, 1.15)
    axb.set_xlabel("Validation F1 (5-class run)")
    axb.set_title("Minority classes collapse\nextrastole / extrahls starved of data",
                  fontsize=11, loc="left")
    save(fig, "fig_taxonomy_pivot.png")


# ----------------------------------------------------------------------------
# 4. Training curve from real checkpoint filenames.
# ----------------------------------------------------------------------------
def fig_training():
    pat = re.compile(
        r"epoch=(\d+)-precision=([0-9.]+)-recall=([0-9.]+)-f1-score=([0-9.]+)\.ckpt")
    rows = []
    for f in glob.glob(os.path.join(ROOT, "checkpoints", "*.ckpt")):
        m = pat.search(os.path.basename(f))
        if m:
            rows.append(tuple(float(x) for x in m.groups()))
    if not rows:
        print("no checkpoints found, skipping training curve")
        return
    rows.sort()
    ep = [int(r[0]) for r in rows]
    p = [r[1] for r in rows]
    r = [r[2] for r in rows]
    f1 = [r_[3] for r_ in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(ep, p, "-o", color=ACCENT2, lw=2, ms=5, label="macro precision")
    ax.plot(ep, r, "-o", color=ACCENT, lw=2, ms=5, label="macro recall")
    ax.plot(ep, f1, "-o", color=ACCENT3, lw=2.4, ms=6, label="macro F1")
    best = int(np.argmax(f1))
    ax.scatter([ep[best]], [f1[best]], s=140, facecolors="none",
               edgecolors=INK, lw=1.6, zorder=5)
    ax.annotate(f"best macro-F1 = {f1[best]:.2f}  (epoch {ep[best]})",
                (ep[best], f1[best]), textcoords="offset points",
                xytext=(-8, 16), fontsize=9, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro metric")
    ax.set_ylim(0.6, 0.95)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_title("Best run: validation macro precision / recall / F1 per epoch\n"
                 "precision–recall trade-off oscillates before F1 settles",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    save(fig, "fig_training_curve.png")


# ----------------------------------------------------------------------------
# 5. Per-class F1 of the best model.
# ----------------------------------------------------------------------------
def fig_perclass():
    cls = ["abnormal", "healthy", "artifact"]
    prec = [0.929, 0.896, 0.784]
    rec = [1.000, 0.945, 0.643]
    f1 = [0.963, 0.920, 0.707]
    x = np.arange(len(cls))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w, prec, w, label="precision", color=ACCENT2, alpha=0.9)
    ax.bar(x, rec, w, label="recall", color=ACCENT, alpha=0.9)
    ax.bar(x + w, f1, w, label="F1", color=ACCENT3, alpha=0.9)
    for i in range(len(cls)):
        for dx, val in [(-w, prec[i]), (0, rec[i]), (w, f1[i])]:
            ax.text(x[i] + dx, val + 0.012, f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cls)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Best model, per-class validation metrics (3-class)\n"
                 "artifact remains the hardest class (F1 0.71)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    save(fig, "fig_per_class_f1.png")


if __name__ == "__main__":
    fig_cvd()
    fig_corpus()
    fig_taxonomy()
    fig_training()
    fig_perclass()
    print("done ->", os.path.relpath(OUT, ROOT))
