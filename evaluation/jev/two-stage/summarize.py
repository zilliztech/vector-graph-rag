"""Recompute frozen Recall@5/10 without model calls or corpus redistribution."""

import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent


def summarize():
    rows = json.loads(gzip.decompress((ROOT / "results.json.gz").read_bytes()))
    manifest = json.loads((ROOT / "manifest.json").read_text())
    groups = defaultdict(list)
    for row in rows:
        gold = set(row["gold"])
        assert gold
        for k in (5, 10):
            score = len(gold & set(row["retrieved"][:k])) / len(gold)
            assert abs(score - row[f"r{k}"]) < 1e-12
        groups[(row["dataset"], row["method"])].append(row)
    summary = {}
    for (dataset, method), selected in sorted(groups.items()):
        expected = {(r["index"], r["id"]) for r in manifest["samples"][dataset]}
        assert len(selected) == len(expected) == 1000
        assert {(r["index"], r["id"]) for r in selected} == expected
        summary.setdefault(dataset, {})[method] = {
            "n": len(selected),
            **{f"recall_at_{k}": 100 * mean(r[f"r{k}"] for r in selected) for k in (5, 10)},
        }
    assert set(groups) == {
        (dataset, method)
        for dataset in ("musique", "2wikimultihopqa")
        for method in ("naive", "jev_single", "jev_two", "jev_two_114")
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
