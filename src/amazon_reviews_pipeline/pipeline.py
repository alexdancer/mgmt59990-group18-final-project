from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT = Path("data/bronze/reviews/category=Electronics/electronics_sample.jsonl")
DEFAULT_DATA_ROOT = Path("data")
REQUIRED_FIELDS = {
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            records.append(row)
    return records, malformed


def transform(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame.from_records(records)
    missing_columns = sorted(REQUIRED_FIELDS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    input_rows = len(frame)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["helpful_vote"] = pd.to_numeric(frame["helpful_vote"], errors="coerce").fillna(0).astype("int64")
    frame["verified_purchase"] = frame["verified_purchase"].astype("boolean")
    frame["title_clean"] = frame["title"].map(normalize_text)
    frame["text_clean"] = frame["text"].map(normalize_text)
    frame["review_text"] = (frame["title_clean"] + ". " + frame["text_clean"]).str.strip(". ")
    frame["review_ts"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")

    invalid_rating = ~frame["rating"].isin([1, 2, 3, 4, 5])
    empty_text = frame["review_text"].str.len().fillna(0).eq(0)
    missing_parent = frame["parent_asin"].isna() | frame["parent_asin"].astype(str).str.len().eq(0)
    invalid_timestamp = frame["review_ts"].isna()
    valid = ~(invalid_rating | empty_text | missing_parent | invalid_timestamp)
    frame = frame.loc[valid].copy()

    frame["rating"] = frame["rating"].astype("int8")
    frame["review_year"] = frame["review_ts"].dt.year.astype("int16")
    frame["review_month"] = frame["review_ts"].dt.strftime("%Y-%m")
    frame["text_length"] = frame["review_text"].str.len().astype("int32")
    frame["review_id"] = frame.apply(
        lambda row: hashlib.sha256(
            f"{row['user_id']}|{row['parent_asin']}|{int(row['timestamp'])}|{row['review_text']}".encode("utf-8")
        ).hexdigest()[:24],
        axis=1,
    )
    duplicate_rows = int(frame.duplicated(subset=["review_id"]).sum())
    frame = frame.drop_duplicates(subset=["review_id"], keep="first")

    keep = [
        "review_id",
        "rating",
        "title_clean",
        "text_clean",
        "review_text",
        "asin",
        "parent_asin",
        "user_id",
        "review_ts",
        "review_year",
        "review_month",
        "helpful_vote",
        "verified_purchase",
        "text_length",
    ]
    frame = frame[keep].sort_values(["review_ts", "review_id"]).reset_index(drop=True)
    counts = frame["rating"].value_counts().sort_index()
    quality = {
        "input_rows": input_rows,
        "valid_rows_before_deduplication": int(valid.sum()),
        "output_rows": len(frame),
        "invalid_rating_rows": int(invalid_rating.sum()),
        "empty_text_rows": int(empty_text.sum()),
        "missing_parent_asin_rows": int(missing_parent.sum()),
        "invalid_timestamp_rows": int(invalid_timestamp.sum()),
        "duplicate_rows_removed": duplicate_rows,
        "verified_purchase_rate": round(float(frame["verified_purchase"].mean()), 4),
        "median_text_length": int(frame["text_length"].median()),
        "rating_distribution": {str(int(key)): int(value) for key, value in counts.items()},
        "min_review_timestamp": frame["review_ts"].min().isoformat(),
        "max_review_timestamp": frame["review_ts"].max().isoformat(),
    }
    return frame, quality


def write_outputs(frame: pd.DataFrame, quality: dict[str, Any], data_root: Path) -> None:
    silver_root = data_root / "silver" / "reviews"
    gold_root = data_root / "gold"
    silver_root.mkdir(parents=True, exist_ok=True)
    gold_root.mkdir(parents=True, exist_ok=True)

    for year, group in frame.groupby("review_year", sort=True):
        partition = silver_root / f"category=Electronics/review_year={int(year)}"
        partition.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["review_year"]).to_parquet(
            partition / "part-00000.parquet", index=False, compression="snappy"
        )

    monthly = (
        frame.groupby(["review_month", "rating"], observed=True)
        .agg(
            review_count=("review_id", "count"),
            verified_review_count=("verified_purchase", "sum"),
            average_helpful_votes=("helpful_vote", "mean"),
            average_text_length=("text_length", "mean"),
        )
        .reset_index()
    )
    monthly["verified_review_count"] = monthly["verified_review_count"].astype("int64")
    monthly.to_parquet(gold_root / "monthly_rating_summary.parquet", index=False, compression="snappy")
    monthly.to_csv(gold_root / "monthly_rating_summary.csv", index=False)

    model_features = frame[["review_id", "review_text", "rating", "review_year"]].copy()
    model_features.to_parquet(gold_root / "model_features.parquet", index=False, compression="snappy")
    (gold_root / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    frame["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="review_count").to_csv(
        gold_root / "rating_distribution.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Silver and Gold data products")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, malformed = load_jsonl(args.input)
    frame, quality = transform(records)
    quality["malformed_json_rows"] = malformed
    write_outputs(frame, quality, args.data_root)
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()

