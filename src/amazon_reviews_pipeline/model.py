from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def train(input_path: Path, output_dir: Path, random_state: int = 42) -> dict[str, object]:
    frame = pd.read_parquet(input_path)
    train_x, test_x, train_y, test_y = train_test_split(
        frame["review_text"],
        frame["rating"],
        test_size=0.2,
        random_state=random_state,
        stratify=frame["rating"],
    )
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(class_weight="balanced", random_state=random_state)),
        ]
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    report = classification_report(test_y, predictions, labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0)
    metrics: dict[str, object] = {
        "model": "TF-IDF (1-2 grams) + class-weighted LinearSVC",
        "random_state": random_state,
        "training_rows": len(train_x),
        "test_rows": len(test_x),
        "accuracy": round(float(accuracy_score(test_y, predictions)), 4),
        "macro_f1": round(float(f1_score(test_y, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(test_y, predictions, average="weighted")), 4),
        "classification_report": report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "baseline_tfidf_linearsvc.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(test_y, predictions, labels=[1, 2, 3, 4, 5]),
        index=[f"actual_{i}" for i in range(1, 6)],
        columns=[f"predicted_{i}" for i in range(1, 6)],
    ).to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "model_card.md").write_text(
        "# Baseline Model Card\n\n"
        "## Intended use\n\n"
        "Predict 1–5 star ratings from Amazon Electronics review title and body text as a sentiment proxy. "
        "The checkpoint model is for feasibility testing, not production routing.\n\n"
        "## Training data\n\n"
        f"Bounded first-byte sample of Amazon Reviews 2023 Electronics: {len(frame):,} cleaned records; "
        f"{len(train_x):,} train and {len(test_x):,} stratified test records.\n\n"
        "## Method\n\n"
        "TF-IDF word unigrams/bigrams (30,000 feature cap) with a class-weighted LinearSVC.\n\n"
        "## Metrics\n\n"
        f"- Accuracy: {metrics['accuracy']:.4f}\n"
        f"- Macro-F1: {metrics['macro_f1']:.4f}\n"
        f"- Weighted-F1: {metrics['weighted_f1']:.4f}\n\n"
        "## Limitations\n\n"
        "The sample is a deterministic byte-range rather than a statistically random draw, review classes are imbalanced, "
        "metadata is not yet joined, and text may encode category-specific language. Production use requires temporal "
        "validation, bias/error analysis, drift monitoring, and human review for safety or defect escalation.\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the five-class checkpoint baseline")
    parser.add_argument("--input", type=Path, default=Path("data/gold/model_features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

