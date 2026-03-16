from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from . import duplicates, musiq
from .config import default_config


def _parse_common_dir_arg(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        name,
        type=Path,
        help="Root directory to scan for JPEGs (main and subdirectories).",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Image analysis CLI (MUSIQ scoring and duplicate detection).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # musiq subcommand
    musiq_parser = subparsers.add_parser(
        "score",
        help="Score images with MUSIQ and write CSV file(s).",
    )
    _parse_common_dir_arg(musiq_parser, "directory")
    musiq_parser.add_argument(
        "--max-size",
        type=int,
        nargs="+",
        default=default_config.musiq_default_max_sizes,
        metavar="N",
        help=(
            "One or more max-size values in pixels for resizing the longest side. "
            "Use 0 for full resolution (no resize)."
        ),
    )
    musiq_parser.add_argument(
        "--output-prefix",
        type=str,
        default=default_config.musiq_csv_prefix,
        help=(
            "Prefix for output CSV files. For each max-size value, a CSV named "
            "<prefix>_<size>.csv will be created in the directory being evaluated."
        ),
    )

    # dedupe subcommand
    dedupe_parser = subparsers.add_parser(
        "dedupe",
        help="Find scene duplicates using MUSIQ scores and CNN encodings.",
    )
    _parse_common_dir_arg(dedupe_parser, "day_directory")
    dedupe_parser.add_argument(
        "--threshold",
        type=float,
        default=default_config.min_similarity_threshold,
        metavar="T",
        help="Minimum cosine similarity to consider same scene (default from config).",
    )
    dedupe_parser.add_argument(
        "--gps-radius-meters",
        type=float,
        default=default_config.gps_radius_meters or 0,
        metavar="M",
        help=(
            "When both photos have EXIF GPS, only test for duplicate if candidate is within M meters of keeper. "
            "Use 0 to disable GPS filtering."
        ),
    )
    dedupe_parser.add_argument(
        "--musiq-csv-size",
        type=int,
        default=default_config.musiq_csv_default_size,
        metavar="N",
        help=(
            "Max-size label for MUSIQ CSV: look for "
            f"{default_config.musiq_csv_prefix}_N.csv (default from config). Use 0 for 'full'."
        ),
    )
    dedupe_parser.add_argument(
        "--poor-quality-threshold",
        type=float,
        default=default_config.poor_quality_threshold,
        metavar="S",
        help="Score below this is 'poor quality' and excluded from duplicate check.",
    )
    dedupe_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path for JSON duplicate report. Default: same directory as photos (scene_duplicates_report.json).",
    )
    dedupe_parser.add_argument(
        "--no-copy-by-status",
        action="store_true",
        help="Do not copy images into _by_status/<status>/ subfolders.",
    )
    dedupe_parser.add_argument(
        "--list-remove",
        action="store_true",
        help="Print only the list of duplicate paths (one per line) for piping to removal scripts.",
    )
    dedupe_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress.",
    )

    args = parser.parse_args(argv)

    if args.command == "score":
        root = args.directory.resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        images = musiq.find_jpeg_files(root)
        musiq.write_scores_csv_for_sizes(
            root,
            images,
            max_sizes=[int(v) for v in args.max_size],
            output_prefix=args.output_prefix,
            config=default_config,
        )
        return 0

    if args.command == "dedupe":
        image_root = args.day_directory.resolve()
        if not image_root.is_dir():
            raise SystemExit(f"Not a directory: {image_root}")

        gps_radius = None if args.gps_radius_meters == 0 else args.gps_radius_meters
        keeper_to_dups, dup_to_keeper = duplicates.find_duplicates_by_score(
            image_root,
            config=default_config,
            min_similarity_threshold=args.threshold,
            gps_radius_meters=gps_radius,
            musiq_csv_size=args.musiq_csv_size,
            poor_quality_threshold=args.poor_quality_threshold,
            verbose=args.verbose,
        )

        if args.list_remove:
            for dup in sorted(dup_to_keeper.keys()):
                print(dup)
            return 0

        total_dups = sum(len(d) for d in keeper_to_dups.values())
        musiq_label = "full" if args.musiq_csv_size == 0 else str(args.musiq_csv_size)
        print(f"Day directory: {image_root}")
        print(f"MUSIQ CSV size: {musiq_label}")
        print(f"Similarity threshold: {args.threshold}")
        print(
            "GPS radius (meters): "
            f"{args.gps_radius_meters if args.gps_radius_meters != 0 else 'disabled'}"
        )
        print(f"Keepers with at least one duplicate: {len(keeper_to_dups)}")
        print(f"Total images marked as duplicate (same scene, lower score): {total_dups}")
        print()

        if keeper_to_dups:
            print("=== Keepers and their duplicate(s) (same scene) ===")
            for keeper in sorted(
                keeper_to_dups.keys(),
                key=lambda k: (-len(keeper_to_dups[k]), k),
            ):
                dups = keeper_to_dups[keeper]
                print(f"  Keeper: {keeper}")
                for d in sorted(dups):
                    print(f"    duplicate: {d}")
                print()
        else:
            print("No scene duplicates found at this threshold.")

        report_path = (
            args.output
            if args.output is not None
            else image_root / "scene_duplicates_report.json"
        )
        report = {
            "day_directory": str(image_root),
            "musiq_csv_size": args.musiq_csv_size,
            "threshold": args.threshold,
            "gps_radius_meters": args.gps_radius_meters
            if args.gps_radius_meters != 0
            else None,
            "keeper_to_duplicates": keeper_to_dups,
            "duplicate_to_keeper": dup_to_keeper,
        }
        report_path.write_text(
            __import__("json").dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote report to {report_path}")

        if not args.no_copy_by_status:
            musiq_rows = duplicates.load_full_musiq_csv(
                image_root,
                size=args.musiq_csv_size,
                prefix=default_config.musiq_csv_prefix,
            )
            if musiq_rows:
                status_csv_path = duplicates.write_status_csv(
                    image_root,
                    musiq_rows,
                    dup_to_keeper,
                    poor_quality_threshold=args.poor_quality_threshold,
                )
                print(f"Wrote {status_csv_path.name} (all fields) to {image_root}")
                duplicates.copy_images_by_status(
                    image_root,
                    musiq_rows,
                    dup_to_keeper,
                    poor_quality_threshold=args.poor_quality_threshold,
                )
                print(f"Copied images to {image_root / duplicates.BY_STATUS_DIR}")

        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

