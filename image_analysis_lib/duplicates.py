from __future__ import annotations

import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

import exifread
from imagededup.methods import CNN

from .config import ImageAnalysisConfig, default_config
from .scoring import (
    COSINE_SIM_SENTINEL,
    cosine_sim_to_csv_value,
    csv_status_for_row,
    parse_musiq_score,
)


_METERS_PER_DEG_LAT = 111_320


def _exifread_component_to_float(v) -> float:
    """Convert one DMS component from exifread (Rational or number) to float."""
    if v is None:
        return 0.0
    if hasattr(v, "num") and hasattr(v, "den"):
        return float(v.num) / float(v.den) if v.den else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def distance_meters_flat(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two (lat, lon) points in meters (flat-earth; fine for ~200 m)."""
    dlat_deg = lat2 - lat1
    dlon_deg = lon2 - lon1
    lat_mid_rad = math.radians((lat1 + lat2) / 2)
    dlat_m = dlat_deg * _METERS_PER_DEG_LAT
    dlon_m = dlon_deg * _METERS_PER_DEG_LAT * math.cos(lat_mid_rad)
    return math.sqrt(dlat_m * dlat_m + dlon_m * dlon_m)


_EXIFREAD_LAT_TAGS = ["GPS GPSLatitude", "GPS GPSLatitudeRef"]
_EXIFREAD_LON_TAGS = ["GPS GPSLongitude", "GPS GPSLongitudeRef"]
_EXIFREAD_DATETIME_TAGS = [
    "EXIF DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "Image DateTime",
    "GPS GPSDate",
    "GPS GPSDateStamp",
]

_EXIFREAD_MAKE_TAGS = ["Image Make"]
_EXIFREAD_MODEL_TAGS = ["Image Model"]
_EXIFREAD_IMAGE_WIDTH_TAGS = ["Image ImageWidth", "EXIF ExifImageWidth"]
_EXIFREAD_IMAGE_LENGTH_TAGS = ["Image ImageLength", "EXIF ExifImageLength"]
_EXIFREAD_ORIENTATION_TAGS = ["Image Orientation"]
_EXIFREAD_EXPOSURE_TIME_TAGS = ["EXIF ExposureTime"]
_EXIFREAD_FNUMBER_TAGS = ["EXIF FNumber"]
_EXIFREAD_ISO_SPEED_TAGS = ["EXIF ISOSpeedRatings"]


def _exifread_first_value(tags: dict, tag_names: list[str]) -> str:
    """Return first found tag value as string; empty string if none. Handles .values and rationals."""
    for name in tag_names:
        if name not in tags:
            continue
        tag = tags[name]
        if not hasattr(tag, "values"):
            return getattr(tag, "printable", str(tag)) or ""
        val = tag.values
        if isinstance(val, (list, tuple)):
            if len(val) == 1:
                val = val[0]
            elif len(val) == 2:
                n = _exifread_component_to_float(val[0])
                d = _exifread_component_to_float(val[1])
                if not d:
                    return str(int(n)) if n == int(n) else str(n)
                if n == int(n) and d == int(d) and int(d) > 1:
                    return f"{int(n)}/{int(d)}"
                return f"{n / d:.4g}"
        if val is None:
            return ""
        if hasattr(val, "num") and hasattr(val, "den"):
            return str(_exifread_component_to_float(val))
        return str(val)
    return ""


def get_exif_extras(image_path: Path) -> dict[str, str]:
    """
    Read camera, dimensions, orientation, exposure from EXIF via exifread.

    Returns dict with keys: img_make, img_model, exif_image_width, exif_image_length,
    img_orientation, exif_exposure_time, exif_f_number, exif_iso_speed_ratings.
    """
    out: dict[str, str] = {
        "img_make": "",
        "img_model": "",
        "exif_image_width": "",
        "exif_image_length": "",
        "img_orientation": "",
        "exif_exposure_time": "",
        "exif_f_number": "",
        "exif_iso_speed_ratings": "",
    }
    if not image_path or not image_path.is_file():
        return out
    try:
        with open(str(image_path), "rb") as f:
            tags = exifread.process_file(f)
    except Exception:
        return out
    out["img_make"] = _exifread_first_value(tags, _EXIFREAD_MAKE_TAGS)
    out["img_model"] = _exifread_first_value(tags, _EXIFREAD_MODEL_TAGS)
    out["exif_image_width"] = _exifread_first_value(tags, _EXIFREAD_IMAGE_WIDTH_TAGS)
    out["exif_image_length"] = _exifread_first_value(tags, _EXIFREAD_IMAGE_LENGTH_TAGS)
    out["img_orientation"] = _exifread_first_value(tags, _EXIFREAD_ORIENTATION_TAGS)
    out["exif_exposure_time"] = _exifread_first_value(tags, _EXIFREAD_EXPOSURE_TIME_TAGS)
    out["exif_f_number"] = _exifread_first_value(tags, _EXIFREAD_FNUMBER_TAGS)
    out["exif_iso_speed_ratings"] = _exifread_first_value(tags, _EXIFREAD_ISO_SPEED_TAGS)
    return out


def get_gps_from_exif(image_path: Path) -> tuple[float, float] | None:
    """
    Read EXIF GPS using exifread.
    """
    if not image_path or not image_path.is_file():
        return None
    try:
        with open(str(image_path), "rb") as f:
            tags = exifread.process_file(f)
    except Exception:
        return None
    lat_tag = next((t for t in _EXIFREAD_LAT_TAGS if "Ref" not in t and t in tags), None)
    lat_ref_tag = next((t for t in _EXIFREAD_LAT_TAGS if "Ref" in t and t in tags), None)
    lon_tag = next((t for t in _EXIFREAD_LON_TAGS if "Ref" not in t and t in tags), None)
    lon_ref_tag = next((t for t in _EXIFREAD_LON_TAGS if "Ref" in t and t in tags), None)
    if not lat_tag or not lon_tag:
        return None
    try:
        lat_vals = tags[lat_tag].values
        lon_vals = tags[lon_tag].values
        if len(lat_vals) != 3 or len(lon_vals) != 3:
            return None
        lat_dec = (
            _exifread_component_to_float(lat_vals[0])
            + _exifread_component_to_float(lat_vals[1]) / 60.0
            + _exifread_component_to_float(lat_vals[2]) / 3600.0
        )
        lon_dec = (
            _exifread_component_to_float(lon_vals[0])
            + _exifread_component_to_float(lon_vals[1]) / 60.0
            + _exifread_component_to_float(lon_vals[2]) / 3600.0
        )
        lat_ref = tags.get(lat_ref_tag)
        lon_ref = tags.get(lon_ref_tag)
        if lat_ref is not None and hasattr(lat_ref, "values"):
            ref_val = lat_ref.values
            ref_s = ref_val[0] if ref_val else "N"
            if ref_s in ("S", "s"):
                lat_dec = -lat_dec
        if lon_ref is not None and hasattr(lon_ref, "values"):
            ref_val = lon_ref.values
            ref_s = ref_val[0] if ref_val else "E"
            if ref_s in ("W", "w"):
                lon_dec = -lon_dec
        return (lat_dec, lon_dec)
    except (KeyError, TypeError, IndexError, ZeroDivisionError):
        return None


def build_gps_cache(image_root: Path, relative_paths: list[str]) -> dict[str, tuple[float, float] | None]:
    """Return map relative_path -> (lat, lon) or None if no GPS."""
    out: dict[str, tuple[float, float] | None] = {}
    for rel in relative_paths:
        full = image_root / rel
        out[rel] = get_gps_from_exif(full) if full.is_file() else None
    return out


def get_datetime_taken(image_path: Path) -> str:
    """
    Return date/time the photo was taken using exifread.
    Fallback: file mtime (ISO).
    """
    if image_path and image_path.is_file():
        try:
            with open(str(image_path), "rb") as f:
                tags = exifread.process_file(f)
            for tag_name in _EXIFREAD_DATETIME_TAGS:
                if tag_name not in tags:
                    continue
                tag = tags[tag_name]
                if not hasattr(tag, "values"):
                    continue
                val = tag.values
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                if not val:
                    continue
                s = str(val).strip()
                if len(s) >= 19 and ":" in s:
                    date_part = s[:10].replace(":", "-")
                    time_part = s[11:19]
                    return f"{date_part}T{time_part}"
                return s
        except Exception:
            pass
    if image_path:
        try:
            mtime = image_path.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception:
            pass
    return ""


def load_scores_from_musiq_csv(
    image_root: Path,
    size: int,
    prefix: str,
) -> dict[str, float]:
    """Load (relative image path -> musiq_score) from MUSIQ CSV: {prefix}_{size}.csv."""
    import csv

    size_label = "full" if size == 0 else str(size)
    csv_path = image_root / f"{prefix}_{size_label}.csv"
    if not csv_path.exists():
        return {}
    out: dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = _get_relative_image_path_from_row(row)
            if not rel:
                continue
            raw = row.get("musiq_score")
            if raw is None or raw == "":
                continue
            try:
                out[rel] = float(raw)
            except ValueError:
                pass
    return out


def load_full_musiq_csv(
    image_root: Path,
    size: int,
    prefix: str,
) -> list[dict[str, str]]:
    """Load all rows from image_evaluator_musiq CSV as list of dicts (same column names)."""
    import csv

    size_label = "full" if size == 0 else str(size)
    csv_path = image_root / f"{prefix}_{size_label}.csv"
    if not csv_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out_row = {k: ("" if v is None else str(v).strip()) for k, v in row.items()}
            rows.append(out_row)
    return rows


def get_scored_paths_in_order(
    image_root: Path,
    *,
    musiq_csv_size: int,
    prefix: str,
) -> list[tuple[str, float]]:
    """Return list of (relative_path, score) sorted by score descending (best first)."""
    scores = load_scores_from_musiq_csv(image_root, size=musiq_csv_size, prefix=prefix)
    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    return ordered


def build_encoding_map_for_paths(
    image_root: Path,
    relative_paths: list[str],
) -> dict[str, "np.ndarray"]:
    """Build CNN encoding map for the given relative paths. Keys = relative_path."""
    cnn = CNN()
    encodings: dict[str, "np.ndarray"] = {}
    for rel in relative_paths:
        full = image_root / rel
        if not full.is_file():
            continue
        enc = cnn.encode_image(image_file=str(full))
        if enc is not None:
            encodings[rel] = enc
    return encodings


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    a = np.asarray(a).flatten().astype(float)
    b = np.asarray(b).flatten().astype(float)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def find_duplicates_by_score(
    image_root: Path,
    *,
    config: ImageAnalysisConfig = default_config,
    min_similarity_threshold: float | None = None,
    gps_radius_meters: float | None = None,
    musiq_csv_size: int | None = None,
    poor_quality_threshold: float | None = None,
    verbose: bool = False,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, float]]:
    """
    For each image in score-desc order, mark lower-scoring images as duplicates if same scene.

    Comparisons run only within the same directory (same parent as the relative path under
    image_root), matching MUSIQ CSV file_location folder boundaries.

    Returns:
      keeper_to_duplicates: keeper relative_path -> list of duplicate relative_paths
      duplicate_to_keeper: duplicate relative_path -> keeper relative_path
      duplicate_to_cosine_sim: duplicate relative_path -> cosine similarity to assigned keeper
    """

    if min_similarity_threshold is None:
        min_similarity_threshold = config.min_similarity_threshold
    if gps_radius_meters is None:
        gps_radius_meters = config.gps_radius_meters
    if musiq_csv_size is None:
        musiq_csv_size = config.musiq_csv_default_size
    if poor_quality_threshold is None:
        poor_quality_threshold = config.poor_quality_threshold

    ordered_all = get_scored_paths_in_order(
        image_root,
        musiq_csv_size=musiq_csv_size,
        prefix=config.musiq_csv_prefix,
    )
    ordered = [(p, s) for p, s in ordered_all if s >= poor_quality_threshold]
    if not ordered:
        return {}, {}, {}

    paths = [p for p, _ in ordered]
    use_gps = gps_radius_meters is not None and gps_radius_meters > 0
    gps_cache: dict[str, tuple[float, float] | None] = {}
    if use_gps:
        gps_cache = build_gps_cache(image_root, paths)

    encodings = build_encoding_map_for_paths(image_root, paths)
    if len(encodings) < 2:
        return {}, {}, {}

    path_to_idx = {p: i for i, (p, _) in enumerate(ordered)}
    keeper_to_duplicates: dict[str, list[str]] = {}
    duplicate_to_keeper: dict[str, str] = {}
    duplicate_to_cosine_sim: dict[str, float] = {}

    folder_to_paths: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        folder_key = Path(p).parent.as_posix()
        folder_to_paths[folder_key].append(p)

    for bucket_paths in folder_to_paths.values():
        for keeper_path in bucket_paths:
            if keeper_path not in encodings:
                continue
            if keeper_path in duplicate_to_keeper:
                continue
            keeper_idx = path_to_idx[keeper_path]
            keeper_score = ordered[keeper_idx][1]
            keeper_enc = encodings[keeper_path]
            keeper_gps = gps_cache.get(keeper_path) if use_gps else None

            for other_path in bucket_paths:
                if other_path == keeper_path or other_path not in encodings:
                    continue
                if path_to_idx[other_path] <= keeper_idx:
                    continue
                if other_path in duplicate_to_keeper:
                    continue

                if use_gps and keeper_gps is not None:
                    other_gps = gps_cache.get(other_path)
                    if other_gps is not None:
                        dist = distance_meters_flat(
                            keeper_gps[0],
                            keeper_gps[1],
                            other_gps[0],
                            other_gps[1],
                        )
                        if dist > gps_radius_meters:
                            continue

                sim = cosine_similarity(keeper_enc, encodings[other_path])
                if sim >= min_similarity_threshold:
                    duplicate_to_keeper[other_path] = keeper_path
                    duplicate_to_cosine_sim[other_path] = sim
                    keeper_to_duplicates.setdefault(keeper_path, []).append(other_path)

            if verbose:
                n_dups = len(keeper_to_duplicates.get(keeper_path, []))
                print(
                    f"  Processing highest-scoring image (score {keeper_score:.4f}): "
                    f"{keeper_path} - {n_dups} duplicate(s) found"
                )

    if verbose:
        n_dups = len(duplicate_to_keeper)
        n_remain = len(ordered) - n_dups
        print(f"  Duplicates found: {n_dups}")
        print(f"  Non-duplicate images remaining: {n_remain}")

    return keeper_to_duplicates, duplicate_to_keeper, duplicate_to_cosine_sim


STATUS_CSV_BASENAME = "image_scores_and_status.csv"

_EXIF_EXTRAS_KEYS = [
    "img_make",
    "img_model",
    "exif_image_width",
    "exif_image_length",
    "img_orientation",
    "exif_exposure_time",
    "exif_f_number",
    "exif_iso_speed_ratings",
]

def _get_relative_image_path_from_row(row: dict[str, str]) -> str:
    """
    Return a relative image path (relative to the MUSIQ/dup "image_root") from a CSV row.

    MUSIQ rows are produced with:
    - file_location: sys.subdir from process_images/exif_loader.py ('' or '/YYYY-MM-DD' etc.)
    - file_name: basename

    This function returns the internal relative path used by this library for file access:
    - '' + '/' => 'file.jpg' (top-level)
    - '/subdir' + '/' + 'file.jpg' => 'subdir/file.jpg' (leading slash removed for Path joining)
    """
    loc = (row.get("file_location") or "").strip()
    name = (row.get("file_name") or "").strip()
    if not name:
        return ""
    if not loc:
        return name

    # process_images uses a leading '/' for nested folders (sys.subdir),
    # but we must remove it for correct path joining with `image_root / rel`.
    loc = loc.lstrip("/").rstrip("/")
    return f"{loc}/{name}" if loc else name


def write_status_csv(
    image_root: Path,
    rows: list[dict[str, str]],
    duplicate_to_keeper: dict[str, str],
    duplicate_to_cosine_sim: dict[str, float],
    *,
    poor_quality_threshold: float,
    best_score_threshold: float,
    tbd_best_score_threshold: float,
) -> Path:
    """
    Write CSV with same fields as input MUSIQ CSV plus: gps_latitude, gps_longitude,
    date_time_taken, EXIF extras (make, model, dimensions, orientation, exposure),
    status, dup_photo, cosine_sim (-1 for non-duplicates).
    """
    import csv

    if not rows:
        return image_root / STATUS_CSV_BASENAME

    def _norm_rel(r: str) -> str:
        return Path(r).as_posix() if r else ""

    paths = [
        _get_relative_image_path_from_row(r)
        for r in rows
        if _get_relative_image_path_from_row(r)
    ]
    gps_cache = build_gps_cache(image_root, paths)
    extras_cache: dict[str, dict[str, str]] = {}
    for rel in paths:
        full = image_root / rel if rel else None
        key = _norm_rel(rel)
        extras_cache[key] = (
            get_exif_extras(full) if full and full.is_file() else {k: "" for k in _EXIF_EXTRAS_KEYS}
        )
    out_path = image_root / STATUS_CSV_BASENAME
    input_keys = list(rows[0].keys())
    extra_keys = [
        "gps_latitude",
        "gps_longitude",
        "date_time_taken",
        *_EXIF_EXTRAS_KEYS,
        "status",
        "dup_photo",
        "cosine_sim",
    ]
    fieldnames = input_keys + extra_keys

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rel = _get_relative_image_path_from_row(row)
            rel_key = _norm_rel(rel)
            score = parse_musiq_score(row.get("musiq_score"))
            status, dup_photo = csv_status_for_row(
                rel,
                score,
                duplicate_to_keeper,
                poor_quality_threshold=poor_quality_threshold,
                best_score_threshold=best_score_threshold,
                tbd_best_score_threshold=tbd_best_score_threshold,
            )
            if rel in duplicate_to_cosine_sim:
                cosine_cell = cosine_sim_to_csv_value(duplicate_to_cosine_sim[rel])
            else:
                cosine_cell = cosine_sim_to_csv_value(float(COSINE_SIM_SENTINEL))
            gps = gps_cache.get(rel) if rel else None
            gps_lat = f"{gps[0]:.6f}" if gps else ""
            gps_lon = f"{gps[1]:.6f}" if gps else ""
            full = image_root / rel if rel else None
            date_time_taken = get_datetime_taken(full) if full and full.is_file() else ""
            extras = extras_cache.get(rel) or extras_cache.get(rel_key) or {k: "" for k in _EXIF_EXTRAS_KEYS}
            out_row = dict(row)
            out_row["gps_latitude"] = gps_lat
            out_row["gps_longitude"] = gps_lon
            out_row["date_time_taken"] = date_time_taken
            for k in _EXIF_EXTRAS_KEYS:
                out_row[k] = extras.get(k, "")
            out_row["status"] = status
            out_row["dup_photo"] = dup_photo
            out_row["cosine_sim"] = cosine_cell
            writer.writerow(out_row)
    return out_path


BY_STATUS_DIR = "_by_status"

_STATUS_FOLDER_NAMES = {
    "best": "best",
    "good": "good",
    "TBD": "tbd",
    "poor quality": "poor quality",
    "dup": None,  # use dup_<keeper_basename>
}


def copy_images_by_status(
    image_root: Path,
    rows: list[dict[str, str]],
    duplicate_to_keeper: dict[str, str],
    *,
    poor_quality_threshold: float,
    best_score_threshold: float,
    tbd_best_score_threshold: float,
) -> None:
    """
    Copy each image into image_root/_by_status/<status_folder>/ using its original filename.
    Status folders: best, good, tbd, poor quality; for dup use dup_<keeper_basename>.
    Removes any existing _by_status directory before copying.
    """
    base = image_root / BY_STATUS_DIR
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    for row in rows:
        rel = _get_relative_image_path_from_row(row)
        if not rel:
            continue
        src = image_root / rel
        if not src.is_file():
            continue
        score = parse_musiq_score(row.get("musiq_score"))
        status, dup_photo = csv_status_for_row(
            rel,
            score,
            duplicate_to_keeper,
            poor_quality_threshold=poor_quality_threshold,
            best_score_threshold=best_score_threshold,
            tbd_best_score_threshold=tbd_best_score_threshold,
        )
        folder_name = _STATUS_FOLDER_NAMES.get(status)
        if folder_name is None and status == "dup" and dup_photo:
            keeper_basename = Path(dup_photo).name
            folder_name = f"dup_{keeper_basename}"
        elif folder_name is None:
            folder_name = status
        dest_dir = base / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = Path(rel).name
        dest = dest_dir / dest_name
        shutil.copy2(src, dest)

