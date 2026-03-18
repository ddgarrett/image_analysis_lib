"""Pytest fixtures for image_analysis_lib tests."""

from pathlib import Path

import pytest

# Directory containing test images (tests/images/)
TEST_IMAGES_DIR = Path(__file__).resolve().parent / "images"


@pytest.fixture
def test_images_dir():
    """Path to the tests/images directory with sample images."""
    return TEST_IMAGES_DIR


@pytest.fixture
def test_image_paths(test_images_dir):
    """List of paths to test images in tests/images/."""
    if not test_images_dir.is_dir():
        return []
    paths = sorted(
        p for p in test_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif"}
    )
    return paths
