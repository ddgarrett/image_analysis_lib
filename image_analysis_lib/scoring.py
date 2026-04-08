"""
Shared scoring rules: CSV status strings and collection (img_status, rvw_lvl).

Used by dedupe CSV output, process_images import, and optional GUI recalc.
"""

from __future__ import annotations

# Written to CSV / image_collection for non-duplicate rows (including keepers).
COSINE_SIM_SENTINEL: int = -1


def parse_musiq_score(raw: object) -> float | None:
    """Parse musiq_score from CSV or collection cell; None if missing or invalid."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def csv_status_for_row(
    relative_path: str,
    score: float | None,
    duplicate_to_keeper: dict[str, str],
    *,
    poor_quality_threshold: float,
    best_score_threshold: float,
    tbd_best_score_threshold: float,
) -> tuple[str, str]:
    """
    Return (status, dup_photo) for image_scores_and_status.csv.
    dup_photo is the keeper relative path when status is 'dup', else ''.
    """
    if score is not None and score < poor_quality_threshold:
        return "poor quality", ""
    if relative_path in duplicate_to_keeper:
        return "dup", duplicate_to_keeper[relative_path]
    if score is not None:
        if score > best_score_threshold:
            return "best", ""
        if score > tbd_best_score_threshold:
            return "good", ""
    return "TBD", ""


def status_csv_to_collection_fields(status_raw: str | None) -> tuple[str, str]:
    """Map image_scores_and_status.csv 'status' to (img_status, rvw_lvl as string).

    Unknown or empty status uses ("tbd", "0").
    """
    s = (status_raw or "").strip()
    if s == "poor quality":
        return ("bad", "1")
    if s == "dup":
        return ("dup", "2")
    if s == "best":
        return ("best", "5")
    if s == "good":
        return ("tbd", "4")
    if s == "TBD":
        return ("tbd", "3")
    return ("tbd", "0")


def collection_fields_from_score_bands(
    score: float | None,
    *,
    poor_quality_threshold: float,
    best_score_threshold: float,
    tbd_best_score_threshold: float,
) -> tuple[str, str]:
    """
    img_status and rvw_lvl when the row is not treated as a duplicate.
    Same band rules as csv_status_for_row, without duplicate_to_keeper.
    """
    st, _dup = csv_status_for_row(
        "",
        score,
        {},
        poor_quality_threshold=poor_quality_threshold,
        best_score_threshold=best_score_threshold,
        tbd_best_score_threshold=tbd_best_score_threshold,
    )
    return status_csv_to_collection_fields(st)


def cosine_sim_to_csv_value(sim: float) -> str:
    """Format cosine similarity for CSV; use -1 for sentinel."""
    if sim == float(COSINE_SIM_SENTINEL):
        return str(COSINE_SIM_SENTINEL)
    return f"{sim:.6f}"


def parse_cosine_cell(raw: object) -> float | None:
    """
    Parse collection or CSV cosine_sim cell.
    Returns None if blank/unknown (recalc: skip threshold adjust).
    Returns -1.0 for non-dup sentinel; else float similarity.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None
