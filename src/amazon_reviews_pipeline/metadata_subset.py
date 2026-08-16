from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_REVIEWS = Path(
    "data/bronze/reviews/category=Electronics/electronics_random_sample.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/bronze/metadata/category=Electronics/electronics_metadata_sample.jsonl"
)
SOURCE_REPOSITORY = "McAuley-Lab/Amazon-Reviews-2023"
SOURCE_COMMIT = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
SOURCE_CONFIG = "raw_meta_Electronics"
SHARD_COUNT = 10
SHARD_NAMES = [f"full-{index:05d}-of-{SHARD_COUNT:05d}.parquet" for index in range(SHARD_COUNT)]
METADATA_FIELDS = (
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "images",
    "videos",
    "store",
    "categories",
    "details",
    "parent_asin",
    "bought_together",
)


def shard_url(name: str) -> str:
    return (
        f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}/resolve/"
        f"{SOURCE_COMMIT}/{SOURCE_CONFIG}/{name}?download=true"
    )


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"none", "null", "nan"}
    if isinstance(value, (list, dict)):
        return not value
    return False


def normalize_json(value: Any) -> Any:
    """Convert Arrow values into strict JSON values, replacing non-finite floats."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def price_status(value: Any) -> str:
    if is_missing(value):
        return "missing"
    if isinstance(value, bool):
        return "invalid"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "invalid"
    return "valid" if math.isfinite(numeric) and numeric >= 0 else "invalid"


def load_review_targets(path: Path) -> tuple[set[str], Counter[str], dict[str, int]]:
    targets: set[str] = set()
    review_counts: Counter[str] = Counter()
    diagnostics = {"review_rows": 0, "malformed_review_rows": 0, "missing_parent_asin_rows": 0}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            diagnostics["review_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                diagnostics["malformed_review_rows"] += 1
                continue
            parent_asin = record.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                diagnostics["missing_parent_asin_rows"] += 1
                continue
            parent_asin = parent_asin.strip()
            targets.add(parent_asin)
            review_counts[parent_asin] += 1
    return targets, review_counts, diagnostics


def download_shard(url: str, destination: Path, retries: int = 3) -> tuple[int, str]:
    for attempt in range(1, retries + 1):
        digest = hashlib.sha256()
        downloaded = 0
        next_progress = 64 * 1024 * 1024
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "mgmt59990-final-project/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                with destination.open("wb") as output:
                    while chunk := response.read(8 * 1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_progress:
                            print(f"    downloaded {downloaded / 1024 / 1024:.0f} MiB", flush=True)
                            next_progress += 64 * 1024 * 1024
            return downloaded, digest.hexdigest()
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == retries:
                raise
            print(f"    download attempt {attempt} failed; retrying", flush=True)
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def scan_shard(
    path: Path,
    targets: set[str],
    matched_records: dict[str, str],
) -> tuple[int, int, int, int]:
    parquet_file = pq.ParquetFile(path)
    source_rows = parquet_file.metadata.num_rows
    matched_rows = 0
    duplicate_keys = 0
    conflicting_duplicates = 0

    for batch in parquet_file.iter_batches(batch_size=8192):
        for raw_record in batch.to_pylist():
            parent_asin = raw_record.get("parent_asin")
            if parent_asin not in targets:
                continue
            matched_rows += 1
            record = normalize_json(raw_record)
            encoded = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            existing = matched_records.get(parent_asin)
            if existing is None:
                matched_records[parent_asin] = encoded
            else:
                duplicate_keys += 1
                if existing != encoded:
                    conflicting_duplicates += 1
    return source_rows, matched_rows, duplicate_keys, conflicting_duplicates


def validate_records(encoded_records: dict[str, str]) -> dict[str, Any]:
    missing_counts = Counter({field: 0 for field in METADATA_FIELDS})
    prices = Counter({"valid": 0, "missing": 0, "invalid": 0})
    main_categories: Counter[str] = Counter()

    for encoded in encoded_records.values():
        record = json.loads(encoded)
        for field in METADATA_FIELDS:
            if is_missing(record.get(field)):
                missing_counts[field] += 1
        prices[price_status(record.get("price"))] += 1
        category = record.get("main_category")
        if not is_missing(category):
            main_categories[str(category)] += 1

    return {
        "missing_field_counts": dict(missing_counts),
        "price_validation": dict(prices),
        "main_category_counts": dict(main_categories.most_common()),
    }


def write_jsonl(records: dict[str, str], output: Path) -> tuple[int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for parent_asin in sorted(records):
            line = records[parent_asin].encode("utf-8") + b"\n"
            handle.write(line)
            digest.update(line)
    temporary.replace(output)
    return output.stat().st_size, digest.hexdigest()


def build_metadata_subset(reviews: Path, output: Path) -> dict[str, Any]:
    targets, review_counts, review_diagnostics = load_review_targets(reviews)
    print(
        f"Loaded {len(targets):,} target parent ASINs from "
        f"{review_diagnostics['review_rows']:,} reviews",
        flush=True,
    )

    matched_records: dict[str, str] = {}
    shard_results: list[dict[str, Any]] = []
    source_rows_total = 0
    source_bytes_downloaded = 0
    matched_source_rows = 0
    duplicate_metadata_rows = 0
    conflicting_duplicate_rows = 0

    with tempfile.TemporaryDirectory(prefix="mgmt59990-metadata-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for index, name in enumerate(SHARD_NAMES, start=1):
            url = shard_url(name)
            local_path = temporary_root / name
            print(f"[{index}/{SHARD_COUNT}] Downloading {name}", flush=True)
            downloaded_bytes, shard_sha256 = download_shard(url, local_path)
            print(f"[{index}/{SHARD_COUNT}] Scanning {name}", flush=True)
            source_rows, matched_rows, duplicates, conflicts = scan_shard(
                local_path, targets, matched_records
            )
            local_path.unlink(missing_ok=True)

            source_rows_total += source_rows
            source_bytes_downloaded += downloaded_bytes
            matched_source_rows += matched_rows
            duplicate_metadata_rows += duplicates
            conflicting_duplicate_rows += conflicts
            shard_results.append(
                {
                    "name": name,
                    "url": url,
                    "downloaded_bytes": downloaded_bytes,
                    "sha256": shard_sha256,
                    "source_rows": source_rows,
                    "matching_rows": matched_rows,
                }
            )
            print(
                f"[{index}/{SHARD_COUNT}] {matched_rows:,} matching rows; "
                f"{len(matched_records):,} unique matches accumulated",
                flush=True,
            )

    output_bytes, output_sha256 = write_jsonl(matched_records, output)
    matched_keys = set(matched_records)
    unmatched = sorted(targets - matched_keys)
    matched_review_rows = sum(review_counts[key] for key in matched_keys)
    review_rows_with_parent = sum(review_counts.values())
    validation = validate_records(matched_records)

    manifest: dict[str, Any] = {
        "source_dataset": SOURCE_REPOSITORY,
        "source_config": SOURCE_CONFIG,
        "source_commit": SOURCE_COMMIT,
        "source_format": "Parquet",
        "source_shards": shard_results,
        "source_rows_scanned": source_rows_total,
        "source_bytes_downloaded": source_bytes_downloaded,
        "filter_key": "parent_asin",
        "review_source": str(reviews),
        **review_diagnostics,
        "unique_review_parent_asins": len(targets),
        "metadata_rows_matching_before_deduplication": matched_source_rows,
        "saved_metadata_rows": len(matched_records),
        "duplicate_metadata_rows": duplicate_metadata_rows,
        "conflicting_duplicate_metadata_rows": conflicting_duplicate_rows,
        "matched_parent_asins": len(matched_keys),
        "unmatched_parent_asins_count": len(unmatched),
        "unmatched_parent_asins": unmatched,
        "parent_asin_coverage": len(matched_keys) / len(targets) if targets else 0.0,
        "reviews_with_parent_asin": review_rows_with_parent,
        "reviews_matched_to_metadata": matched_review_rows,
        "review_level_join_coverage": (
            matched_review_rows / review_rows_with_parent if review_rows_with_parent else 0.0
        ),
        **validation,
        "saved_path": str(output),
        "saved_bytes": output_bytes,
        "sha256": output_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "The review dataset is a seeded distributed byte-range sample, not an exact uniform row sample.",
            "Product prices reflect the source crawl and may be missing or stale.",
            "Only metadata whose parent_asin occurs in the review sample is retained.",
            "The legacy McAuley Lab compressed metadata URL returned HTTP 404, so the official Hugging Face Parquet conversion pinned above was used.",
        ],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate an Electronics metadata subset for the review sample"
    )
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_metadata_subset(args.reviews, args.output)
    summary_keys = (
        "saved_metadata_rows",
        "matched_parent_asins",
        "unmatched_parent_asins_count",
        "parent_asin_coverage",
        "reviews_matched_to_metadata",
        "review_level_join_coverage",
        "duplicate_metadata_rows",
        "conflicting_duplicate_metadata_rows",
        "price_validation",
        "saved_bytes",
        "sha256",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2), flush=True)


if __name__ == "__main__":
    main()
