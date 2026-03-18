"""Tests for image_analysis_lib.musiq."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

# Mock TensorFlow so we can test pure helpers without loading the model
import sys
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow_hub"] = MagicMock()

from image_analysis_lib.musiq import (
    IMAGE_EXTENSIONS,
    collect_file_info,
    find_jpeg_files,
    image_to_jpeg_bytes,
    relative_path,
)
from image_analysis_lib.musiq import _is_ignored_path as is_ignored_path
from image_analysis_lib.musiq import _resize_image as resize_image


class TestImageExtensions:
    """Tests for IMAGE_EXTENSIONS constant."""

    def test_includes_common_formats(self):
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".tif" in IMAGE_EXTENSIONS


class TestIsIgnoredPath:
    """Tests for _is_ignored_path."""

    def test_dotfile_under_root_is_ignored(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        assert is_ignored_path(tmp_path / ".hidden" / "file.txt", tmp_path) is True

    def test_underscore_dir_is_ignored(self, tmp_path):
        (tmp_path / "_private").mkdir()
        assert is_ignored_path(tmp_path / "_private" / "file.txt", tmp_path) is True

    def test_normal_path_not_ignored(self, tmp_path):
        (tmp_path / "photos").mkdir()
        assert is_ignored_path(tmp_path / "photos" / "a.jpg", tmp_path) is False

    def test_dot_in_middle_not_ignored(self, tmp_path):
        (tmp_path / "my.photos").mkdir()
        assert is_ignored_path(tmp_path / "my.photos" / "a.jpg", tmp_path) is False


class TestResizeImage:
    """Tests for _resize_image."""

    def test_small_image_unchanged(self):
        img = Image.new("RGB", (100, 50))
        out = resize_image(img, max_size=200)
        assert out.size == (100, 50)

    def test_large_image_resized_by_longest_side(self):
        img = Image.new("RGB", (400, 200))
        out = resize_image(img, max_size=200)
        assert max(out.size) == 200
        assert out.size[0] == 200
        assert out.size[1] == 100

    def test_tall_image_resized(self):
        img = Image.new("RGB", (100, 400))
        out = resize_image(img, max_size=200)
        assert max(out.size) == 200
        assert out.size[1] == 200
        assert out.size[0] == 50


class TestRelativePath:
    """Tests for relative_path."""

    def test_subpath_uses_forward_slashes(self, tmp_path):
        sub = tmp_path / "a" / "b" / "c.jpg"
        sub.parent.mkdir(parents=True, exist_ok=True)
        sub.touch()
        assert relative_path(sub, tmp_path) == "a/b/c.jpg"

    def test_direct_child(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.touch()
        assert relative_path(f, tmp_path) == "photo.jpg"


class TestFindJpegFiles:
    """Tests for find_jpeg_files."""

    def test_empty_dir_returns_empty(self, tmp_path):
        assert find_jpeg_files(tmp_path) == []

    def test_finds_images_by_extension(self, tmp_path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.JPEG").touch()
        (tmp_path / "c.png").touch()
        (tmp_path / "d.txt").touch()
        result = find_jpeg_files(tmp_path)
        paths = [p.name for p in result]
        assert "a.jpg" in paths
        assert "b.JPEG" in paths
        assert "c.png" in paths
        assert "d.txt" not in paths
        assert len(result) == 3

    def test_ignores_dot_underscore_dirs(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "photo.jpg").touch()
        (tmp_path / "_data").mkdir()
        (tmp_path / "_data" / "x.png").touch()
        (tmp_path / "visible").mkdir()
        (tmp_path / "visible" / "a.jpg").touch()
        result = find_jpeg_files(tmp_path)
        rels = [p.relative_to(tmp_path).as_posix() for p in result]
        assert "visible/a.jpg" in rels
        assert ".hidden/photo.jpg" not in rels
        assert "_data/x.png" not in rels

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "z.jpg").touch()
        (tmp_path / "a.jpg").touch()
        result = find_jpeg_files(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_finds_project_test_images(self, test_images_dir, test_image_paths):
        """With tests/images: find_jpeg_files discovers the two sample images."""
        if not test_images_dir.is_dir():
            pytest.skip("tests/images/ not present")
        result = find_jpeg_files(test_images_dir)
        assert len(result) == len(test_image_paths)
        names = {p.name for p in result}
        for path in test_image_paths:
            assert path.name in names


class TestImageToJpegBytes:
    """Tests for image_to_jpeg_bytes with real files."""

    def test_real_image_returns_bytes(self, test_image_paths):
        """With tests/images: returns JPEG bytes."""
        if not test_image_paths:
            pytest.skip("no test images in tests/images/")
        raw = image_to_jpeg_bytes(test_image_paths[0])
        assert isinstance(raw, bytes)
        assert len(raw) > 0
        assert raw[:2] == b"\xff\xd8"  # JPEG SOI

    def test_real_image_resize_reduces_size(self, test_image_paths):
        """With tests/images: max_size reduces payload size."""
        if not test_image_paths:
            pytest.skip("no test images in tests/images/")
        full = image_to_jpeg_bytes(test_image_paths[0], max_size=None)
        small = image_to_jpeg_bytes(test_image_paths[0], max_size=64)
        assert len(small) <= len(full)
        assert small[:2] == b"\xff\xd8"


class TestCollectFileInfo:
    """Tests for collect_file_info with real files."""

    def test_real_image_returns_metadata(self, test_images_dir, test_image_paths):
        """With tests/images: returns relative_path, dimensions, file size."""
        if not test_image_paths:
            pytest.skip("no test images in tests/images/")
        info = collect_file_info(test_image_paths[0], test_images_dir)
        assert info["relative_path"] == test_image_paths[0].name
        assert "file_size_bytes" in info
        assert info["file_size_bytes"] > 0
        assert "file_size_mb" in info
        assert "width" in info
        assert "height" in info
        if info.get("width") and info.get("height"):
            assert info["width"] > 0 and info["height"] > 0
