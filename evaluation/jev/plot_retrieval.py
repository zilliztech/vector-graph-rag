# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Create a clean comparison figure from audited results and published baselines."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parent
output = root.parents[1] / "docs/assets/evaluation"
output.mkdir(parents=True, exist_ok=True)
results = json.loads((root / "comparison-chart-data.json").read_text())
methods = [
    "naive",
    "naive_colbert",
    "hippo",
    "ircot",
    "hippo2",
    "gpt-4o-mini",
    "gpt-5-mini",
    "jev/threshold_0.5",
]
labels = [
    "Naive RAG\nBGE-large-en-v1.5",
    "Naive RAG\nColBERTv2",
    "HippoRAG\nColBERTv2",
    "IRCoT + HippoRAG",
    "HippoRAG 2",
    "Vector Graph RAG\n+ GPT-4o-mini",
    "Vector Graph RAG\n+ GPT-5-mini",
    "Vector Graph RAG\n+ Jev",
]
colors = {m: "#7d8999" for m in methods}
colors.update(
    {
        "naive": "#61778e",
        "gpt-4o-mini": "#6884b4",
        "gpt-5-mini": "#3157a6",
        "jev/threshold_0.5": "#008674",
    }
)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(15, 8.8), sharex=True, sharey=True)
background = "#fbfcfe"
fig.patch.set_facecolor(background)
fig.subplots_adjust(left=0.245, right=0.97, top=0.77, bottom=0.145, wspace=0.16)
for ax, ds, title in zip(axes, ["musique", "hotpotqa"], ["MuSiQue", "HotpotQA"]):
    ax.set_facecolor(background)
    ax.axhspan(6.57, 7.43, color="#e2f3ef", zorder=0)
    ax.set(xlim=(45, 102), ylim=(7.65, -0.7), xticks=[50, 60, 70, 80, 90, 100])
    ax.xaxis.grid(True, color="#e3e8ef", linewidth=0.8)
    ax.set_axisbelow(True)
    for i, method in enumerate(methods):
        result = results[ds][method]
        mean, interval, color = result["mean"], result["ci95"], colors[method]
        if interval:
            ax.errorbar(
                mean,
                i,
                xerr=[[mean - interval[0]], [interval[1] - mean]],
                fmt="o",
                color=color,
                markersize=8,
                capsize=4,
                elinewidth=2,
                zorder=3,
            )
        else:
            ax.plot(
                mean,
                i,
                "D",
                color=color,
                markerfacecolor=background,
                markersize=8,
                markeredgewidth=1.8,
            )
        ax.text(
            mean,
            i - 0.25,
            f"{mean:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            color=color,
            fontweight="bold" if method.endswith("0.5") else "medium",
        )
    ax.set_yticks(range(len(methods)), labels)
    ax.tick_params(axis="y", length=0, pad=17, labelsize=12)
    ax.tick_params(axis="x", length=0, pad=9, labelsize=11, colors="#586b80")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Recall@5 (%)  →  higher is better", labelpad=14, fontsize=11, color="#526477")
    ax.set_title(title, loc="left", fontsize=19, fontweight="bold", color="#192b40", pad=17)
    delta = "−4.13 pp" if ds == "musique" else "−1.00 pp"
    ax.text(
        0,
        -0.18,
        f"Jev vs GPT-5-mini:  {delta}",
        transform=ax.transAxes,
        color="#007d6d",
        fontsize=12,
        fontweight="bold",
    )
fig.text(
    0.05, 0.94, "Retrieval Performance Comparison", fontsize=25, fontweight="bold", color="#192b40"
)
fig.text(
    0.05,
    0.891,
    "Recall of supporting evidence among the top 5 retrieved documents",
    fontsize=13,
    color="#526477",
)
legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="#3157a6",
        markersize=7,
        label="Our evaluation · 95% confidence interval",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="#7d8999",
        markerfacecolor=background,
        linestyle="None",
        markersize=7,
        label="Published results",
    ),
]
fig.legend(
    handles=legend,
    loc="upper left",
    bbox_to_anchor=(0.046, 0.857),
    ncol=2,
    frameon=False,
    fontsize=11,
    columnspacing=3,
)
for ext in ["png", "svg"]:
    fig.savefig(output / f"retrieval-comparison.{ext}", dpi=220, facecolor=background)
print(output / "retrieval-comparison.png")

# Keep generated SVGs clean for source control.
for svg in output.glob("*.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
