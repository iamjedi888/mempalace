"""
Benchmark report utilities — JSON output and regression detection.

Each test records metrics via record_metric(). At session end, the
conftest.py pytest_terminal_summary hook writes the collected results.
"""

import json
import os
import tempfile


RESULTS_FILE = os.path.join(tempfile.gettempdir(), "mempalace_bench_results.json")


def record_metric(category: str, metric: str, value):
    """Append a metric to the session results file (JSON on disk)."""
    results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            results = {}

    if category not in results:
        results[category] = {}
    results[category][metric] = value

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def check_regression(current_report: str, baseline_report: str, threshold: float = 0.2):
    """
    Compare current benchmark results against a baseline.

    Returns a list of regression descriptions. Empty list = no regressions.

    threshold: fractional degradation allowed (0.2 = 20% worse is OK).
    """
    with open(current_report) as f:
        current = json.load(f)
    with open(baseline_report) as f:
        baseline = json.load(f)

    regressions = []
    # Metrics where HIGHER is worse (latency, memory, etc.)
    higher_is_worse = {
        "latency", "rss", "memory", "oom", "lock_failures", "elapsed",
        "p50_ms", "p95_ms", "p99_ms", "rss_delta_mb", "peak_rss_mb",
    }
    # Metrics where LOWER is worse (throughput, recall, etc.)
    lower_is_worse = {
        "recall", "throughput", "per_sec", "files_per_sec", "drawers_per_sec",
        "triples_per_sec", "improvement",
    }

    for category in baseline.get("results", {}):
        if category not in current.get("results", {}):
            continue
        for metric, base_val in baseline["results"][category].items():
            if metric not in current["results"][category]:
                continue
            curr_val = current["results"][category][metric]
            if not isinstance(base_val, (int, float)) or not isinstance(curr_val, (int, float)):
                continue
            if base_val == 0:
                continue

            # Determine direction
            is_latency_like = any(kw in metric.lower() for kw in higher_is_worse)
            is_throughput_like = any(kw in metric.lower() for kw in lower_is_worse)

            if is_latency_like:
                # Higher is worse — check if current exceeds baseline by threshold
                if curr_val > base_val * (1 + threshold):
                    pct = ((curr_val - base_val) / base_val) * 100
                    regressions.append(
                        f"{category}/{metric}: {base_val:.2f} -> {curr_val:.2f} ({pct:+.1f}%, threshold {threshold*100:.0f}%)"
                    )
            elif is_throughput_like:
                # Lower is worse — check if current is below baseline by threshold
                if curr_val < base_val * (1 - threshold):
                    pct = ((curr_val - base_val) / base_val) * 100
                    regressions.append(
                        f"{category}/{metric}: {base_val:.2f} -> {curr_val:.2f} ({pct:+.1f}%, threshold {threshold*100:.0f}%)"
                    )

    return regressions
