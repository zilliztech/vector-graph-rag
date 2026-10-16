"""Recompute the direct-service estimate from archived paired timings."""
import json
import math
from pathlib import Path

root = Path(__file__).parent
data = json.loads((root / 'jev114-latency-calibration.json').read_text())
weighted = 0.0
for stratum in data['strata']:
    pairs = [p for p in data['paired_timings']
             if p['dataset'] == stratum['dataset']
             and p['index'] in stratum['sample_indices']]
    ratio = sum(p['direct_seconds'] for p in pairs) / sum(p['proxy_seconds'] for p in pairs)
    assert math.isclose(ratio, stratum['ratio'], abs_tol=1e-12)
    weighted += stratum['population_queries'] / 2000 * stratum['population_proxy_mean'] * ratio
assert math.isclose(weighted, data['estimated_full_direct_seconds'], abs_tol=1e-12)
chart = json.loads((root / 'chart-data.json').read_text())
assert math.isclose(weighted, chart['points']['jev_two_114']['latency'], abs_tol=1e-12)
print(f'Official-direct estimate: {weighted:.6f} s; {len(data["paired_timings"])} verified timing pairs')
