"""Qrel-based validation for SWAG rerank — stable seed_key matching, symmetric lift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCHMARKS = Path(__file__).resolve().parent


def load_qrels(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _text_contains_any(text: str, needles: list[str]) -> bool:
    lowered = _norm(text)
    return any(_norm(needle) in lowered for needle in needles)


def hit_seed_key(hit: dict) -> str | None:
    key = hit.get("seed_key")
    if key:
        return str(key)
    source = str(hit.get("source") or "")
    if source.startswith("validation://"):
        return source.removeprefix("validation://")
    return None


def hit_is_relevant(hit: dict, relevant_seed_keys: list[str]) -> bool:
    key = hit_seed_key(hit)
    return bool(key and key in relevant_seed_keys)


def precision_at_k(hits: list[dict], relevant_seed_keys: list[str], k: int) -> float:
    if not relevant_seed_keys or k <= 0:
        return 0.0
    window = hits[:k]
    if not window:
        return 0.0
    hits_count = sum(1 for hit in window if hit_is_relevant(hit, relevant_seed_keys))
    return hits_count / k


def precision_at_1(hits: list[dict], relevant_seed_keys: list[str]) -> float:
    if not hits or not relevant_seed_keys:
        return 0.0
    return 1.0 if hit_is_relevant(hits[0], relevant_seed_keys) else 0.0


def mrr(hits: list[dict], relevant_seed_keys: list[str]) -> float:
    if not relevant_seed_keys:
        return 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit_is_relevant(hit, relevant_seed_keys):
            return 1.0 / idx
    return 0.0


def _check_top1_content(
    hits: list[dict],
    *,
    must_contain: list[str] | None,
    forbidden: list[str] | None,
) -> tuple[bool, str]:
    if not hits:
        if must_contain:
            return False, "no hits for top1 content check"
        return True, "no hits (content check n/a)"
    top_text = str(hits[0].get("text") or "")
    if forbidden and _text_contains_any(top_text, forbidden):
        return False, "top-1 contains forbidden marker"
    if must_contain and not _text_contains_any(top_text, must_contain):
        return False, "top-1 missing required content"
    return True, "top-1 content ok"


def evaluate_case(case: dict[str, Any], response: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    mode = case.get("mode", "swag")
    hits = list(response.get("hits") or [])
    baseline_sym = list(response.get("baseline_hits") or [])
    baseline_raw = list(response.get("baseline_hits_raw") or baseline_sym)
    lift = response.get("lift") or {}

    relevant_keys = list(case.get("relevant_seed_keys") or [])
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}
    metrics: dict[str, float] = {}

    expect_hits = case.get("expect_hits")
    if expect_hits is False:
        checks["no_hits"] = len(hits) == 0
        details["no_hits"] = f"{len(hits)} hits (want 0)"
    elif case.get("min_hits") is not None:
        minimum = int(case["min_hits"])
        checks["min_hits"] = len(hits) >= minimum
        details["min_hits"] = f"{len(hits)} hits (>={minimum})"
    if case.get("max_hits") is not None:
        maximum = int(case["max_hits"])
        checks["max_hits"] = len(hits) <= maximum
        details["max_hits"] = f"{len(hits)} hits (<={maximum})"

    if relevant_keys:
        metrics["swag_full_p1"] = precision_at_1(hits, relevant_keys)
        metrics["swag_full_mrr"] = mrr(hits, relevant_keys)
        metrics["swag_full_p3"] = precision_at_k(hits, relevant_keys, min(3, top_k))
        checks["p1_relevant"] = metrics["swag_full_p1"] >= 1.0
        details["p1_relevant"] = f"swag_full P@1={metrics['swag_full_p1']:.2f}"

    if case.get("compare") and relevant_keys:
        metrics["baseline_sym_p1"] = precision_at_1(baseline_sym, relevant_keys)
        metrics["baseline_raw_p1"] = precision_at_1(baseline_raw, relevant_keys)
        metrics["baseline_sym_mrr"] = mrr(baseline_sym, relevant_keys)
        swag_rerank = lift.get("swag_rerank") or {}
        metrics["swag_rerank_lift"] = 1.0 if swag_rerank.get("has_positive_lift") else 0.0
        checks["swag_beats_symmetric_p1"] = (
            metrics.get("swag_full_p1", 0) >= metrics["baseline_sym_p1"]
        )
        details["symmetric"] = (
            f"sym P@1={metrics['baseline_sym_p1']:.2f} "
            f"full P@1={metrics.get('swag_full_p1', 0):.2f}"
        )

    ok_top1, top1_detail = _check_top1_content(
        hits,
        must_contain=case.get("top1_must_contain"),
        forbidden=case.get("forbidden_top1_contains"),
    )
    if case.get("top1_must_contain") or case.get("forbidden_top1_contains"):
        checks["top1_content"] = ok_top1
        details["top1_content"] = top1_detail

    forbidden_keys = list(case.get("forbidden_top1_seed_keys") or [])
    if forbidden_keys and hits:
        top_key = hit_seed_key(hits[0])
        ok_forbidden = top_key not in forbidden_keys
        checks["forbidden_top1_seed_key"] = ok_forbidden
        details["forbidden_top1_seed_key"] = (
            f"top-1 seed_key={top_key!r}"
            if ok_forbidden
            else f"top-1 is forbidden chrome doc {top_key!r}"
        )

    passed = all(checks.values()) if checks else False
    top1_preview = None
    if hits:
        top1_preview = str(hits[0].get("text") or "")[:120]

    return {
        "id": case.get("id"),
        "query": case.get("query"),
        "mode": mode,
        "suite": case.get("suite"),
        "passed": passed,
        "checks": checks,
        "details": details,
        "metrics": metrics,
        "top1_preview": top1_preview,
        "top1_seed_key": hit_seed_key(hits[0]) if hits else None,
    }


def summarize_results(
    sheet: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    threshold = float(sheet.get("pass_threshold", 0.8))
    scored = [row for row in case_results if row.get("checks")]
    passed = sum(1 for row in scored if row.get("passed"))
    total = len(scored)
    rate = (passed / total) if total else 0.0
    verdict = "PASS" if rate >= threshold else "FAIL"
    lines = [f"=== Validation eval ({sheet.get('description', '')}) ==="]
    lines.append(f"Verdict: {verdict} ({passed}/{total}, threshold {threshold:.0%})")
    for row in case_results:
        mark = "PASS" if row.get("passed") else "FAIL"
        detail = "; ".join(row.get("details", {}).values()) or row.get("id", "")
        lines.append(f"  {mark} {row.get('id', ''):28} {detail}")
    return {
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "pass_rate": round(rate, 4),
        "pass_threshold": threshold,
        "report_text": "\n".join(lines),
    }
