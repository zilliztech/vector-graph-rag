# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24"]
# ///
"""Recompute final metrics and query-cluster bootstrap intervals without API calls."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ["naive", "gpt-4o-mini", "gpt-5-mini", "jev/threshold_0.5"]
HISTORICAL = {
    "musique": {"naive_colbert": 49.2, "hippo": 51.9, "ircot": 57.6, "hippo2": 74.7},
    "hotpotqa": {"naive_colbert": 79.3, "hippo": 77.7, "ircot": 83.0, "hippo2": 96.3},
}


def summarize():
    rows = json.loads((ROOT / "results.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())["samples"]
    chart, table = {}, []
    for row in rows:
        gold = set(row["gold"])
        assert gold
        for k in (5, 10):
            # Preserve the existing evaluator's per-position evidence matching.
            score = sum(item in gold for item in row["retrieved"][:k]) / len(gold)
            assert abs(score - row[f"recall_at_{k}"]) < 1e-12
    for dataset in HISTORICAL:
        chart[dataset] = {}
        expected = {(r["index"], r["id"]) for r in manifest[dataset]}
        for method in METHODS:
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            assert len(selected) == len(expected) == 500
            assert {(r["index"], r["id"]) for r in selected} == expected
            groups = defaultdict(list)
            for row in selected:
                groups[row["id"]].append(row["recall_at_5"])
            sums = np.array([sum(v) for v in groups.values()])
            counts = np.array([len(v) for v in groups.values()])
            rng = np.random.default_rng(20260921)
            draws = rng.integers(0, len(groups), size=(10000, len(groups)))
            boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1) * 100
            chart[dataset][method] = {
                "mean": 100 * sums.sum() / counts.sum(),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "rows": len(selected),
                "unique_queries": len(groups),
            }
            for category in ["all"] + sorted({r["category"] for r in selected}):
                subset = (
                    selected
                    if category == "all"
                    else [r for r in selected if r["category"] == category]
                )
                table.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "category": category,
                        "n": len(subset),
                        **{
                            f"recall_at_{k}": 100 * np.mean([r[f"recall_at_{k}"] for r in subset])
                            for k in (5, 10)
                        },
                    }
                )
        for method, mean in HISTORICAL[dataset].items():
            chart[dataset][method] = {
                "mean": mean,
                "ci95": None,
                "basis": "Published 1,000-question aggregate",
                "source": "https://arxiv.org/html/2502.14802v1"
                if method == "hippo2"
                else "https://arxiv.org/html/2405.14831v2",
            }
    (ROOT / "summary.json").write_text(json.dumps(table, indent=2) + "\n")
    (ROOT / "comparison-chart-data.json").write_text(json.dumps(chart, indent=2) + "\n")
    print("Verified 4,000 rows; regenerated summary.json and comparison-chart-data.json.")


if __name__ == "__main__":
    summarize()
