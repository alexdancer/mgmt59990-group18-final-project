from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amazon_reviews_pipeline.download import SOURCE_URL

DEFAULT_OUTPUT = Path(
    "data/bronze/reviews/category=Electronics/electronics_random_sample.jsonl"
)
CONTENT_RANGE_PATTERN = re.compile(r"bytes\s+\d+-\d+/(\d+)", re.IGNORECASE)


def probe_source_size(url: str) -> int:
    request = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "mgmt59990-group18-checkpoint/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        content_range = response.headers.get("Content-Range", "")
        response.read()
    match = CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
    if not match:
        raise RuntimeError(f"Source did not return a usable Content-Range: {content_range!r}")
    return int(match.group(1))


def choose_offsets(total_size: int, range_bytes: int, ranges: int, seed: int) -> list[int]:
    """Pick one random byte offset from each equal-width file segment."""
    if ranges < 1:
        raise ValueError("ranges must be positive")
    if range_bytes < 1 or range_bytes >= total_size:
        raise ValueError("range_bytes must be positive and smaller than the source")
    segment_width = total_size // ranges
    if range_bytes >= segment_width:
        raise ValueError("range_bytes must be smaller than each file segment")

    generator = random.Random(seed)
    offsets: list[int] = []
    for index in range(ranges):
        segment_start = index * segment_width
        segment_end = total_size if index == ranges - 1 else (index + 1) * segment_width
        latest_start = segment_end - range_bytes
        start = generator.randint(segment_start, latest_start)
        offsets.append(start)
    return offsets


def extract_complete_lines(payload: bytes) -> list[bytes]:
    """Discard both boundary fragments from a non-zero byte-range response."""
    first_newline = payload.find(b"\n")
    last_newline = payload.rfind(b"\n")
    if first_newline < 0 or last_newline <= first_newline:
        return []
    return [line for line in payload[first_newline + 1 : last_newline].splitlines() if line]


def fetch_range(url: str, start: int, byte_count: int, retries: int = 3) -> tuple[int, list[bytes]]:
    end = start + byte_count - 1
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "mgmt59990-group18-checkpoint/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                if getattr(response, "status", None) != 206:
                    raise RuntimeError(f"Expected HTTP 206, received {getattr(response, 'status', None)}")
                payload = response.read(byte_count + 1)
            if len(payload) > byte_count:
                payload = payload[:byte_count]
            return start, extract_complete_lines(payload)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def record_key(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_random_sample(
    url: str,
    output: Path,
    ranges: int,
    range_bytes: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    total_size = probe_source_size(url)
    offsets = choose_offsets(total_size, range_bytes, ranges, seed)
    results: dict[int, list[bytes]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_range, url, start, range_bytes) for start in offsets]
        for future in as_completed(futures):
            start, lines = future.result()
            results[start] = lines

    unique: dict[str, bytes] = {}
    raw_complete_rows = 0
    for start in sorted(results):
        for line in results[start]:
            record = json.loads(line)
            raw_complete_rows += 1
            unique.setdefault(record_key(record), line)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for line in unique.values():
            handle.write(line)
            handle.write(b"\n")

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "source_url": url,
        "source_size_bytes": total_size,
        "sampling_method": "one seeded random byte range per equal-width source-file segment",
        "random_seed": seed,
        "range_count": ranges,
        "range_bytes": range_bytes,
        "requested_bytes": ranges * range_bytes,
        "raw_complete_rows": raw_complete_rows,
        "duplicate_records_removed": raw_complete_rows - len(unique),
        "saved_rows": len(unique),
        "saved_bytes": output.stat().st_size,
        "sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "range_starts": offsets,
        "statistical_note": (
            "Distributed byte-range sampling reduces file-order bias but is not an exact "
            "uniform row sample because JSON records have different byte lengths."
        ),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a distributed random Electronics sample")
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ranges", type=int, default=32)
    parser.add_argument("--range-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--seed", type=int, default=59990)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_random_sample(
        args.url,
        args.output,
        args.ranges,
        args.range_bytes,
        args.seed,
        args.workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

