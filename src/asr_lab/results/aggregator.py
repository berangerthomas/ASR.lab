"""
Results aggregation module for multi-file and cross-language analysis.

Computes aggregated statistics (mean, median, std, min, max) grouped by
arbitrary dimensions (engine, language, degradation, etc.).
"""

import statistics
from typing import Any, Dict, List, Optional


METRIC_KEYS = ("wer", "cer", "mer", "wil", "wip")


def _safe_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute descriptive statistics for a list of numeric values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "count": 0}
    mean = statistics.mean(clean)
    median = statistics.median(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(clean),
        "max": max(clean),
        "count": len(clean),
    }


def aggregate_by(
    results: List[Dict[str, Any]],
    group_keys: List[str],
) -> List[Dict[str, Any]]:
    """Aggregate metric results by one or more grouping keys.

    Args:
        results: List of result dicts (from BenchmarkRunner).
        group_keys: Fields to group by, e.g. ["engine", "language"].

    Returns:
        List of dicts, each containing group key values + aggregated stats
        per metric + average processing time stats.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in results:
        key = tuple(r.get(k, "unknown") for k in group_keys)
        groups.setdefault(key, []).append(r)

    aggregated = []
    for key_vals, group_results in groups.items():
        row: Dict[str, Any] = {}
        for k, v in zip(group_keys, key_vals):
            row[k] = v

        for metric in METRIC_KEYS:
            values = [
                r["metrics"].get(metric) or r["metrics"].get(metric.upper())
                for r in group_results
            ]
            row[metric] = _safe_stats(values)

        # Processing time stats
        times = [
            r.get("transcription", {}).get("processing_time")
            for r in group_results
        ]
        row["processing_time"] = _safe_stats(times)

        aggregated.append(row)

    return aggregated


def cross_language_matrix(
    results: List[Dict[str, Any]],
    metric: str = "wer",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build an engine × language matrix of mean metric values.

    Returns:
        Nested dict: {engine: {language: mean_metric_value}}
    """
    # Group by (engine, language)
    agg = aggregate_by(results, ["engine", "language"])
    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    for row in agg:
        engine = row["engine"]
        lang = row["language"]
        matrix.setdefault(engine, {})[lang] = row[metric]["mean"]
    return matrix


def language_consistency(
    results: List[Dict[str, Any]],
    metric: str = "wer",
) -> List[Dict[str, Any]]:
    """Compute per-engine cross-language consistency (std of mean metric across languages).

    Returns:
        List of dicts with engine, per-language means, and cross-language std.
    """
    matrix = cross_language_matrix(results, metric)
    consistency = []
    for engine, lang_means in matrix.items():
        values = [v for v in lang_means.values() if v is not None]
        consistency.append({
            "engine": engine,
            "languages": lang_means,
            "cross_lang_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "cross_lang_mean": statistics.mean(values) if values else None,
        })
    return consistency
 
