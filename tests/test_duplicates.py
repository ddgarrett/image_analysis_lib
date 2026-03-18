"""Tests for image_analysis_lib.duplicates."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock heavy deps so we can test pure/CSV helpers without CNN or TensorFlow
import sys
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow_hub"] = MagicMock()
sys.modules["imagededup"] = MagicMock()
sys.modules["imagededup.methods"] = MagicMock()
sys.modules["imagededup.methods.cnn"] = MagicMock()

from image_analysis_lib.duplicates import (
    build_gps_cache,
    cosine_similarity,
    distance_meters_flat,
    get_datetime_taken,
    get_exif_extras,
    get_gps_from_exif,
    get_scored_paths_in_order,
    load_full_musiq_csv,
    load_scores_from_musiq_csv,
)
from image_analysis_lib.duplicates import _parse_score as parse_score
from image_analysis_lib.duplicates import _status_for_row as status_for_row


class TestDistanceMetersFlat:
    """Tests for distance_meters_flat."""

    def test_same_point_returns_zero(self):
        assert distance_meters_flat(0.0, 0.0, 0.0, 0.0) == 0.0
        assert distance_meters_flat(45.0, -122.0, 45.0, -122.0) == 0.0

    def test_north_south_distance(self):
        # ~111 km per degree latitude at equator
        d = distance_meters_flat(0.0, 0.0, 1.0, 0.0)
        assert 110_000 < d < 112_000

    def test_east_west_distance_at_equator(self):
        d = distance_meters_flat(0.0, 0.0, 0.0, 1.0)
        assert 110_000 < d < 112_000

    def test_small_offset(self):
        d = distance_meters_flat(45.0, -122.0, 45.001, -122.001)
        assert d > 0
        assert d < 500  # roughly within a few hundred meters


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_accepts_lists(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


class TestParseScore:
    """Tests for _parse_score."""

    def test_empty_or_none_returns_none(self):
        assert parse_score("") is None
        assert parse_score("   ") is None
        assert parse_score(None) is None

    def test_valid_float(self):
        assert parse_score("5.5") == 5.5
        assert parse_score("10") == 10.0

    def test_invalid_returns_none(self):
        assert parse_score("nope") is None
        assert parse_score("5.5.5") is None


class TestStatusForRow:
    """Tests for _status_for_row."""

    def test_poor_quality_below_threshold(self):
        status, dup = status_for_row("a.jpg", 3.0, {}, poor_quality_threshold=4.0)
        assert status == "poor quality"
        assert dup == ""

    def test_duplicate(self):
        status, dup = status_for_row(
            "dup.jpg", 6.0, {"dup.jpg": "keeper.jpg"}, poor_quality_threshold=4.0
        )
        assert status == "dup"
        assert dup == "keeper.jpg"

    def test_best_above_6(self):
        status, dup = status_for_row("a.jpg", 6.5, {}, poor_quality_threshold=4.0)
        assert status == "best"
        assert dup == ""

    def test_good_between_5_and_6(self):
        status, dup = status_for_row("a.jpg", 5.5, {}, poor_quality_threshold=4.0)
        assert status == "good"
        assert dup == ""

    def test_tbd(self):
        status, dup = status_for_row("a.jpg", 4.5, {}, poor_quality_threshold=4.0)
        assert status == "TBD"
        assert dup == ""


class TestGetExifExtras:
    """Tests for get_exif_extras."""

    def test_nonexistent_path_returns_empty_fields(self):
        out = get_exif_extras(Path("/nonexistent/image.jpg"))
        assert out["img_make"] == ""
        assert out["img_model"] == ""
        assert "exif_image_width" in out
        assert "exif_iso_speed_ratings" in out

    def test_none_or_falsy_path(self):
        out = get_exif_extras(Path(""))
        assert out["img_make"] == ""

    def test_real_image_returns_dict_with_expected_keys(self, test_image_paths):
        """With tests/images: returns dict with all EXIF extra keys."""
        if not test_image_paths:
            pytest.skip("no test images in tests/images/")
        out = get_exif_extras(test_image_paths[0])
        expected_keys = [
            "img_make", "img_model", "exif_image_width", "exif_image_length",
            "img_orientation", "exif_exposure_time", "exif_f_number", "exif_iso_speed_ratings",
        ]
        for key in expected_keys:
            assert key in out
            assert isinstance(out[key], str)


class TestGetGpsFromExif:
    """Tests for get_gps_from_exif."""

    def test_real_image_returns_tuple_or_none(self, test_image_paths):
        """With tests/images: returns (lat, lon) or None if no GPS."""
        if not test_image_paths:
            pytest.skip("no test images in tests/images/")
        for path in test_image_paths:
            result = get_gps_from_exif(path)
            if result is not None:
                lat, lon = result
                assert -90 <= lat <= 90
                assert -180 <= lon <= 180
                break


class TestGetDatetimeTaken:
    """Tests for get_datetime_taken."""

    def test_real_image_returns_string(self, test_image_paths):
        """With tests/images: returns non-empty string or empty."""
        if not test_image_paths:
            pytest.skip("no test images in tests/images/")
        for path in test_image_paths:
            result = get_datetime_taken(path)
            assert isinstance(result, str)
            if result:
                assert "T" in result or "-" in result or ":" in result
            break


class TestBuildGpsCache:
    """Tests for build_gps_cache."""

    def test_empty_paths(self):
        cache = build_gps_cache(Path("/tmp"), [])
        assert cache == {}

    def test_nonexistent_files_return_none(self, tmp_path):
        cache = build_gps_cache(tmp_path, ["missing.jpg", "also_missing.png"])
        assert cache["missing.jpg"] is None
        assert cache["also_missing.png"] is None

    def test_real_images_in_cache(self, test_images_dir, test_image_paths):
        """With tests/images: cache has one entry per path; values are (lat,lon) or None."""
        if not test_image_paths:
            pytest.skip("no test images in tests/images/")
        rel_paths = [p.name for p in test_image_paths]
        cache = build_gps_cache(test_images_dir, rel_paths)
        assert len(cache) == len(rel_paths)
        for rel in rel_paths:
            assert rel in cache
            if cache[rel] is not None:
                lat, lon = cache[rel]
                assert -90 <= lat <= 90
                assert -180 <= lon <= 180


class TestLoadScoresFromMusiqCsv:
    """Tests for load_scores_from_musiq_csv."""

    def test_missing_csv_returns_empty(self, tmp_path):
        out = load_scores_from_musiq_csv(tmp_path, size=1024, prefix="results")
        assert out == {}

    def test_loads_scores(self, tmp_path):
        csv_path = tmp_path / "results_1024.csv"
        csv_path.write_text(
            "relative_path,musiq_score,other\n"
            "a.jpg,5.5,x\n"
            "b.jpg,7.2,y\n"
            "c.jpg,,\n",
            encoding="utf-8",
        )
        out = load_scores_from_musiq_csv(tmp_path, size=1024, prefix="results")
        assert out["a.jpg"] == 5.5
        assert out["b.jpg"] == 7.2
        assert "c.jpg" not in out

    def test_size_zero_uses_full_label(self, tmp_path):
        csv_path = tmp_path / "results_full.csv"
        csv_path.write_text("relative_path,musiq_score\np.jpg,6.0\n", encoding="utf-8")
        out = load_scores_from_musiq_csv(tmp_path, size=0, prefix="results")
        assert out["p.jpg"] == 6.0


class TestLoadFullMusiqCsv:
    """Tests for load_full_musiq_csv."""

    def test_missing_csv_returns_empty_list(self, tmp_path):
        out = load_full_musiq_csv(tmp_path, size=1024, prefix="results")
        assert out == []

    def test_loads_all_columns(self, tmp_path):
        csv_path = tmp_path / "results_1024.csv"
        csv_path.write_text(
            "relative_path,musiq_score,file_size_bytes\n"
            "a.jpg,5.5,1000\n"
            "b.jpg,7.2,2000\n",
            encoding="utf-8",
        )
        rows = load_full_musiq_csv(tmp_path, size=1024, prefix="results")
        assert len(rows) == 2
        assert rows[0]["relative_path"] == "a.jpg"
        assert rows[0]["musiq_score"] == "5.5"
        assert rows[0]["file_size_bytes"] == "1000"
        assert rows[1]["relative_path"] == "b.jpg"


class TestGetScoredPathsInOrder:
    """Tests for get_scored_paths_in_order."""

    def test_no_csv_returns_empty(self, tmp_path):
        from image_analysis_lib.config import default_config

        ordered = get_scored_paths_in_order(
            tmp_path,
            musiq_csv_size=1024,
            prefix=default_config.musiq_csv_prefix,
        )
        assert ordered == []

    def test_ordered_by_score_descending(self, tmp_path):
        csv_path = tmp_path / "image_evaluation_musiq_results_1024.csv"
        csv_path.write_text(
            "relative_path,musiq_score\n"
            "low.jpg,3.0\n"
            "high.jpg,8.0\n"
            "mid.jpg,5.0\n",
            encoding="utf-8",
        )
        ordered = get_scored_paths_in_order(
            tmp_path,
            musiq_csv_size=1024,
            prefix="image_evaluation_musiq_results",
        )
        assert ordered == [
            ("high.jpg", 8.0),
            ("mid.jpg", 5.0),
            ("low.jpg", 3.0),
        ]
