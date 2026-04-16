# image_analysis_lib

Shared Python library for image analysis tasks used by `backup_pics` and `process_images`.

It currently provides:

- MUSIQ image-quality scoring (default: local AVA SavedModel under `vendor/musiq_ava/`; optional TensorFlow Hub URL in config).
- Scene duplicate detection using CNN encodings and EXIF GPS.
- A small CLI for running both workflows from the command line.

## Installation

Create or activate a virtual environment with **Python 3.10+** (recommended 3.12 on macOS and Raspberry Pi OS Bookworm).

From the parent directory that contains `image_analysis_lib`:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ./image_analysis_lib
```

This installs the package in editable mode and exposes the `image-analysis` CLI.

## Testing

From the project root, with a virtual environment that has the project dependencies and `pytest` installed (e.g. `pip install pytest` and the project’s runtime deps):

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

To print MUSIQ scores for the sample images in `tests/images/` (requires TensorFlow and TensorFlow Hub):

```bash
PYTHONPATH=. python tests/score_test_images.py
```

If you see `ModuleNotFoundError: No module named 'pkg_resources'` when importing TensorFlow or `image_analysis_lib.musiq`, your environment's `setuptools` may be too new for some dependencies. In that case, in the same virtualenv run:

```bash
pip install "setuptools==68.2.2"
```

and retry the command.

## CLI usage

### MUSIQ scoring

The first time you score with the default config, if `vendor/musiq_ava/` is missing or incomplete, the library downloads the AVA SavedModel from TensorFlow Hub (needs network once, roughly 200 MB). That directory is gitignored.

To score all JPEGs under a directory and write MUSIQ CSV files:

```bash
image-analysis score /path/to/images
```

By default, `--max-size` is `0` (full resolution, no resize). One CSV is written in the image directory, e.g. `image_evaluation_musiq_results_full.csv`.

To score at additional sizes (longest side in pixels), pass one or more values:

```bash
image-analysis score /path/to/images --max-size 1024 0
```

That writes both `image_evaluation_musiq_results_1024.csv` and `image_evaluation_musiq_results_full.csv` when `1024` and `0` are both listed.

These CSVs are compatible with the existing `scene_duplicates_by_score.py` script and with the
`musiq_score` column in `process_images`.

### Scene duplicate detection

After scoring images with MUSIQ, you can find scene duplicates for a single day directory:

```bash
image-analysis dedupe /path/to/day_directory
```

Defaults:

- Reads the full-resolution MUSIQ CSV: `image_evaluation_musiq_results_full.csv` (`--musiq-csv-size` default `0`).
- GPS filtering is off (`--gps-radius-meters` default `0`). Pass e.g. `--gps-radius-meters 200` to only compare pairs that are within that distance when both have EXIF GPS.
- Images are **not** copied into `_by_status/` unless you pass `--copy-by-status`.

This:

- Uses CNN encodings (and optional GPS radius) to detect "same scene, lower score" duplicates.
- Writes `image_scores_and_status.csv` with status (`best`, `good`, `dup`, `poor quality`, `TBD`),
  `cosine_sim` (similarity for `dup` rows, else `-1`), and EXIF extras when MUSIQ data is present.
- Writes `dedup_parms.json` in the same directory with thresholds used for that run.

If you scored at 1024px instead, point dedupe at that CSV:

```bash
image-analysis dedupe /path/to/day_directory --musiq-csv-size 1024
```

You can also print only duplicates for removal scripts:

```bash
image-analysis dedupe /path/to/day_directory --list-remove
```

## Using from `backup_pics`

The `backup_pics` project now delegates MUSIQ scoring and duplicate detection to this library:

- `image_evaluator_musiq.py` calls `image_analysis_lib.musiq.write_scores_csv_for_sizes`.
- `scene_duplicates_by_score.py` calls `image_analysis_lib.duplicates.find_duplicates_by_score`,
  `write_status_csv`, and `copy_images_by_status`.

You can still run those scripts directly as before; they will use the shared implementation.

## Using from `process_images`

`process_images` reads:

- The MUSIQ scores from the CSVs produced by the scoring step (via the `musiq_score` column).
- The review status (`img_status`) and level (`rvw_lvl`) you set interactively.

Typical workflow:

1. **On Raspberry Pi 5** (or on your Mac), run MUSIQ scoring over one or more day directories
   using either `image_evaluator_musiq.py` or the `image-analysis score` CLI.
2. For days where you want automatic duplicate suggestions, run scene-duplicate detection:
   - Either via `scene_duplicates_by_score.py`.
   - Or via the CLI: `image-analysis dedupe /path/to/day_directory ...`.
3. Open the same directory as a collection in `process_images`; it will see:
   - `musiq_score` values (from the MUSIQ CSV).
   - Any pre-labeled statuses from `image_scores_and_status.csv` (e.g. `dup`, `poor quality`).
4. Use the GUI filters and menus in `process_images` to refine statuses and levels.

Because the CSV formats are stable, you can:

- Score on the Raspberry Pi 5, then re-run scoring or duplicate detection on a faster MacBook Air
  later using the same directory tree.
- Re-run with different thresholds (e.g. `--threshold 0.6` vs `0.7`) without changing `backup_pics`
  or `process_images` code.

## Tuning thresholds

The most important knobs are:

- **MUSIQ quality cutoff** (`poor_quality_threshold`, default 4.0):
  - Lower it (e.g. 3.5) to be less aggressive about calling something “poor quality”.
  - Raise it (e.g. 4.5) to skip more images before duplicate checking.
- **Duplicate similarity threshold** (`min_similarity_threshold`, default 0.65):
  - Lower to find more possible duplicates (more sensitive, more false positives).
  - Raise to only flag very similar images (fewer, higher-confidence duplicates).
- **GPS radius** (`gps_radius_meters`, default `0` / disabled in CLI and shared config):
  - Pass a positive value (e.g. `200`) so pairs are only compared when both have GPS and are within that many meters.
  - Leave at `0` to ignore GPS when matching scenes.

For Raspberry Pi 5, you may also want to:

- Use smaller `--max-size` values (e.g. `512` or `1024`) when scoring instead of the default full resolution, to speed up MUSIQ.
- Run scoring in batches of smaller directories (one day at a time).

