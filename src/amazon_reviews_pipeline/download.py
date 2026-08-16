from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/Electronics.jsonl?download=true"
)
DEFAULT_OUTPUT = Path("data/bronze/reviews/category=Electronics/electronics_sample.jsonl")


def download_range(url: str, output: Path, byte_count: int) -> dict[str, object]:
    """Download the first byte_count bytes and remove a trailing partial JSON row."""
    if byte_count < 1:
        raise ValueError("byte_count must be positive")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{byte_count - 1}",
            "User-Agent": "mgmt59990-group18-checkpoint/0.1",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        payload = response.read(byte_count + 1)
        status = getattr(response, "status", None)
    if len(payload) > byte_count:
        payload = payload[:byte_count]
    last_newline = payload.rfind(b"\n")
    if last_newline < 0:
        raise RuntimeError("No complete JSONL record was returned")
    complete = payload[: last_newline + 1]
    output.write_bytes(complete)

    # Validate every preserved line before recording the Bronze artifact.
    rows = 0
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)
            rows += 1

    metadata = {
        "source_url": url,
        "http_status": status,
        "requested_bytes": byte_count,
        "saved_bytes": len(complete),
        "complete_rows": rows,
        "sha256": hashlib.sha256(complete).hexdigest(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_method": "first N bytes; incomplete terminal JSONL record removed",
    }
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a bounded Electronics JSONL sample")
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_range(args.url, args.output, args.bytes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

