from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def create_evidence(data_root: Path, artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    silver_glob = str(data_root / "silver/reviews/category=Electronics/review_year=*/part-*.parquet")
    connection = duckdb.connect()
    query = f"""
        SELECT
            rating,
            COUNT(*) AS review_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_reviews,
            ROUND(AVG(text_length), 1) AS avg_text_length,
            ROUND(100.0 * AVG(CASE WHEN verified_purchase THEN 1 ELSE 0 END), 2) AS pct_verified
        FROM read_parquet('{silver_glob}', hive_partitioning = true)
        GROUP BY rating
        ORDER BY rating
    """
    summary = connection.execute(query).df()
    summary.to_csv(artifact_root / "athena_equivalent_rating_query.csv", index=False)

    quality_query = f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT review_id) AS distinct_review_ids,
            SUM(CASE WHEN parent_asin IS NULL OR parent_asin = '' THEN 1 ELSE 0 END) AS missing_parent_asin,
            SUM(CASE WHEN review_text IS NULL OR LENGTH(review_text) = 0 THEN 1 ELSE 0 END) AS empty_review_text,
            MIN(review_ts) AS earliest_review,
            MAX(review_ts) AS latest_review
        FROM read_parquet('{silver_glob}', hive_partitioning = true)
    """
    quality = connection.execute(quality_query).df()
    quality.to_csv(artifact_root / "athena_equivalent_quality_query.csv", index=False)
    (artifact_root / "queries_executed.sql").write_text(query + ";\n\n" + quality_query + ";\n", encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bars = ax.bar(summary["rating"].astype(str), summary["review_count"], color=["#b91c1c", "#ea580c", "#d97706", "#2563eb", "#166534"])
    ax.set_title("Electronics sample: rating distribution", loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Reviews")
    ax.bar_label(bars, labels=[f"{v:,}" for v in summary["review_count"]], padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(artifact_root / "rating_distribution.png", dpi=180)
    plt.close(fig)

    matrix = pd.read_csv("artifacts/model/confusion_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(5), [1, 2, 3, 4, 5])
    ax.set_yticks(range(5), [1, 2, 3, 4, 5])
    ax.set_xlabel("Predicted rating")
    ax.set_ylabel("Actual rating")
    ax.set_title("Baseline model confusion matrix", loc="left", fontsize=14, fontweight="bold")
    for row in range(5):
        for col in range(5):
            value = int(matrix.iloc[row, col])
            ax.text(col, row, f"{value:,}", ha="center", va="center", color="white" if value > matrix.values.max() * 0.55 else "#111827", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(artifact_root / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    metrics = json.loads(Path("artifacts/model/metrics.json").read_text(encoding="utf-8"))
    run_summary = {
        "silver_rows": int(quality.loc[0, "rows"]),
        "distinct_review_ids": int(quality.loc[0, "distinct_review_ids"]),
        "missing_parent_asin": int(quality.loc[0, "missing_parent_asin"]),
        "empty_review_text": int(quality.loc[0, "empty_review_text"]),
        "model_accuracy": metrics["accuracy"],
        "model_macro_f1": metrics["macro_f1"],
    }
    (artifact_root / "checkpoint_run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create checkpoint SQL and chart evidence")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/evidence"))
    args = parser.parse_args()
    create_evidence(args.data_root, args.artifact_root)


if __name__ == "__main__":
    main()

