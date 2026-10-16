"""Verify the saved paired bootstrap intervals using fingerprint-only records."""

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
rows = json.loads(gzip.decompress((ROOT / "results.json.gz").read_bytes()))
old = {(r["dataset"], r["id"]): r for r in rows if r["method"] == "jev_two"}
new = [r for r in rows if r["method"] == "jev_two_114"]
audit = json.loads((ROOT / "jev114-audit.json").read_text())
assert len(new) == len(old) == 2000
for r in new:
    assert set(r["gold"]) == set(old[r["dataset"], r["id"]]["gold"])
for expected in audit["results"]:
    selected = [
        r
        for r in new
        if (expected["cohort"] == "all2000" or r["cohort"] == expected["cohort"])
        and (expected["dataset"] == "combined" or r["dataset"] == expected["dataset"])
    ]
    assert len(selected) == expected["n"]
    for k in (5, 10):
        groups = [
            np.array(
                [
                    r[f"r{k}"] - old[r["dataset"], r["id"]][f"r{k}"]
                    for r in selected
                    if r["dataset"] == ds
                ]
            )
            for ds in ("musique", "2wikimultihopqa")
        ]
        groups = [g for g in groups if len(g)]
        # Preserve the archived per-query order for exact bootstrap replay.
        rng = np.random.default_rng(20261016)
        bootstrap = sum(
            rng.choice(g, size=(5000, len(g)), replace=True).mean(axis=1) for g in groups
        ) / len(groups)
        delta = float(np.mean([g.mean() for g in groups]) * 100)
        interval = np.quantile(bootstrap, [0.025, 0.975]) * 100
        assert abs(delta - expected[f"delta_r{k}_pp"]) < 1e-10
        assert np.max(np.abs(interval - expected[f"ci95_r{k}_pp"])) < 1e-10
        print(
            expected["cohort"],
            expected["dataset"],
            f"R@{k}",
            f"{delta:+.3f}pp",
            interval.round(3).tolist(),
        )
