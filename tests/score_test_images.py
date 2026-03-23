"""
Score the test images with MUSIQ. Run from project root with TensorFlow installed:

    PYTHONPATH=. python tests/score_test_images.py

Requires: pip install tensorflow tensorflow_hub
"""

from pathlib import Path
import time


def main():
    try:
        from image_analysis_lib.musiq import score_image
    except ImportError as e:
        print("TensorFlow is required to run MUSIQ scoring.")
        print("Install with: pip install tensorflow tensorflow_hub")
        print(f"Error: {e}")
        print("\nTip: use the same Python that has TensorFlow (e.g. activate your venv, then run this script).")
        raise SystemExit(1) from e

    images_dir = Path(__file__).resolve().parent / "images"
    if not images_dir.is_dir():
        print(f"Directory not found: {images_dir}")
        raise SystemExit(1)

    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif"}
    )
    if not paths:
        print(f"No images in {images_dir}")
        raise SystemExit(1)

    print("MUSIQ scores (1-10, higher = better quality):")
    print("")
    for p in paths:
        t0 = time.perf_counter()
        result = score_image(p)
        elapsed = time.perf_counter() - t0
        score = result.get("musiq_score")
        err = result.get("musiq_error")
        if score is not None:
            print(f"  {p.name}: {score:.4f} (time: {elapsed:.3f} s)")
        else:
            print(f"  {p.name}: error — {err} (time: {elapsed:.3f} s)")


if __name__ == "__main__":
    main()
