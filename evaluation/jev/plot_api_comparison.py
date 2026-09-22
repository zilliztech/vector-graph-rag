# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot explicitly illustrative API cost and latency scenarios, not benchmark results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

root = Path(__file__).resolve().parent
output = root.parents[1] / "docs/assets/evaluation"
output.mkdir(parents=True, exist_ok=True)
rows = [
    {
        "label": "HippoRAG 2\n+ Llama-3.3-70B",
        "cost": [0.264, 2.288],
        "latency": [1.5, 15],
        "color": "#7b8798",
        "price_note": "Provider-dependent",
        "cost_label": "$0.26–2.29",
    },
    {
        "label": "HippoRAG 2\n+ GPT-4o-mini (API option)",
        "cost": [0.42, 0.42],
        "latency": [2, 5],
        "color": "#9b92ac",
        "price_note": "Short filtering prompt",
        "cost_label": "$0.42",
    },
    {
        "label": "Vector Graph RAG\n+ GPT-4o-mini",
        "cost": [3.3, 3.3],
        "latency": [4, 10],
        "color": "#718bb6",
        "price_note": "20k input + 500 output",
        "cost_label": "$3.30",
    },
    {
        "label": "Vector Graph RAG\n+ GPT-5-mini",
        "cost": [6, 9],
        "latency": [10, 30],
        "color": "#365ba9",
        "price_note": "Output / reasoning-dependent",
        "cost_label": "$6–9",
    },
    {
        "label": "Vector Graph RAG\n+ Jev",
        "cost": [2.382075318 / 756 * 1000] * 2,
        "latency": [1.2, 4],
        "color": "#008775",
        "price_note": "Based on recorded token usage",
        "cost_label": "$3.15",
    },
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
fig, axs = plt.subplots(1, 2, figsize=(15, 8.2), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
bg = "#fbfcfe"
fig.patch.set_facecolor(bg)
fig.subplots_adjust(left=0.26, right=0.965, top=0.73, bottom=0.23, wspace=0.20)
for ax in axs:
    ax.set_facecolor(bg)
    ax.set_ylim(4.55, -0.65)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.tick_params(axis="y", length=0, pad=18)
    ax.tick_params(axis="x", length=0, pad=9, colors="#53677d")
    ax.xaxis.grid(True, color="#e4e9ef", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhspan(3.58, 4.43, color="#e2f3ef", zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].set_xlim(0, 10)
axs[0].set_xticks([0, 2, 4, 6, 8, 10])
axs[0].set_title(
    "Estimated API cost", loc="left", fontsize=18, fontweight="bold", color="#192b40", pad=30
)
axs[0].text(
    0, 1.015, "USD per 1,000 queries", transform=axs[0].transAxes, color="#64768a", fontsize=11
)
axs[0].set_xlabel("←  cheaper", labelpad=14, color="#526477")
axs[1].set_xscale("log")
axs[1].set_xlim(0.8, 45)
axs[1].xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 40]))
axs[1].xaxis.set_major_formatter(FixedFormatter(["1", "2", "5", "10", "20", "40"]))
axs[1].minorticks_off()
axs[1].set_title(
    "Illustrative latency range",
    loc="left",
    fontsize=18,
    fontweight="bold",
    color="#192b40",
    pad=30,
)
axs[1].text(
    0,
    1.015,
    "Seconds per query · not measured bounds",
    transform=axs[1].transAxes,
    color="#64768a",
    fontsize=11,
)
axs[1].set_xlabel("←  faster  ·  logarithmic scale", labelpad=14, color="#526477")
for i, r in enumerate(rows):
    c = r["color"]
    lo, hi = r["cost"]
    if lo == hi:
        axs[0].barh(i, hi, height=0.13, color=c, alpha=0.8)
        axs[0].plot(hi, i, "o", color=c, ms=7)
    else:
        axs[0].hlines(i, lo, hi, color=c, lw=9, alpha=0.25)
        axs[0].plot([lo, hi], [i, i], "|", ms=12, color=c, markeredgewidth=2)
    axs[0].text(hi + 0.18, i - 0.08, r["cost_label"], fontsize=13, color=c, fontweight="bold")
    axs[0].text(0.13, i + 0.30, r["price_note"], fontsize=9, color="#617387")
    lo, hi = r["latency"]
    axs[1].hlines(i, lo, hi, color=c, lw=10, alpha=0.65)
    axs[1].plot([lo, hi], [i, i], "|", ms=13, color=c, markeredgewidth=2)
    axs[1].text(
        (lo * hi) ** 0.5,
        i - 0.24,
        f"{lo:g}–{hi:g} s",
        ha="center",
        fontsize=13,
        color=c,
        fontweight="bold",
    )
fig.text(0.045, 0.936, "API Cost & Latency", fontsize=26, fontweight="bold", color="#192b40")
fig.text(
    0.045,
    0.884,
    "Online relation filtering / reranking only · quality is not compared",
    fontsize=13,
    color="#526477",
)
fig.text(
    0.045,
    0.831,
    "SCENARIO ESTIMATES — public API prices, external speed reports and our Jev request logs",
    fontsize=10.5,
    color="#8c642c",
)
fig.text(
    0.045,
    0.115,
    "Cost assumptions: HippoRAG 2 = 2k input + 200 output; Vector Graph RAG LLM = 20k input + 500–2k billed output.",
    fontsize=10,
    color="#526477",
)
fig.text(
    0.045,
    0.080,
    "No prompt-cache discounts. Latency ranges are planning scenarios, not confidence intervals or a same-workload benchmark.",
    fontsize=10,
    color="#526477",
)
fig.text(
    0.045,
    0.045,
    "Excludes embeddings, graph / database work, indexing and final answer generation. Price and latency extremes need not coincide.",
    fontsize=10,
    color="#526477",
)
for ext in ["png", "svg"]:
    fig.savefig(output / f"api-cost-latency.{ext}", dpi=220, facecolor=bg)
(root / "api-cost-latency-data.json").write_text(
    json.dumps(
        {
            "rows": rows,
            "scope": "API filtering/reranking only",
            "latency_basis": "Illustrative planning scenarios, not observed ranges or confidence intervals",
            "cache": "No cross-request input caching assumed",
        },
        indent=2,
    )
)
print(output / "api-cost-latency.png")

# Keep generated SVGs clean for source control.
for svg in output.glob("*.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
