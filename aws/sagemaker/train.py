"""SageMaker training entry point for the binary low-rating model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation threshold with best F1, preferring 0.5 on ties."""
    candidates = np.round(np.linspace(0.05, 0.95, 91), 2)
    return float(
        max(
            candidates,
            key=lambda threshold: (
                f1_score(labels, probabilities >= threshold, zero_division=0),
                -abs(float(threshold) - 0.5),
            ),
        )
    )


def evaluate_binary(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, int | float], np.ndarray]:
    """Return auditable binary metrics and thresholded predictions."""
    predictions = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    majority_rows = max(int((labels == 0).sum()), int((labels == 1).sum()))
    metrics: dict[str, int | float] = {
        "rows": int(len(labels)),
        "actual_positive_rows": int((labels == 1).sum()),
        "predicted_positive_rows": int((predictions == 1).sum()),
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "precision": round(
            float(precision_score(labels, predictions, zero_division=0)), 6
        ),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "baseline_accuracy": round(majority_rows / len(labels), 6),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }
    return metrics, predictions


REQUIRED_COLUMNS = {
    "review_id",
    "user_id",
    "parent_asin",
    "review_text",
    "rating",
    "low_rating",
}


def load_channel(channel_path: Path) -> pd.DataFrame:
    """Load every Athena Parquet data object downloaded into one channel."""
    files = sorted(
        path
        for path in channel_path.rglob("*")
        if path.is_file()
        and not path.name.startswith("_")
        and not path.name.endswith((".manifest", ".metadata"))
    )
    if not files:
        raise ValueError(f"No input data files found under {channel_path}")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    frame["review_text"] = frame["review_text"].fillna("").astype(str).str.strip()
    if frame.empty or (frame["review_text"] == "").any():
        raise ValueError("Each channel must contain nonempty review text")
    frame["low_rating"] = frame["low_rating"].astype(int)
    if not set(frame["low_rating"].unique()).issubset({0, 1}):
        raise ValueError("low_rating must contain only 0 and 1")
    return frame


def _upload_file(local_path: Path, destination_uri: str) -> None:
    parsed = urlparse(destination_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 destination: {destination_uri}")
    import boto3

    boto3.client("s3").upload_file(
        str(local_path), parsed.netloc, parsed.path.lstrip("/")
    )


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Train on Gold train, tune on validation, and evaluate once on test."""
    train_frame = load_channel(Path(args.train))
    validation_frame = load_channel(Path(args.validation))
    test_frame = load_channel(Path(args.test))
    for split_name, frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        if set(frame["low_rating"].unique()) != {0, 1}:
            raise ValueError(f"{split_name} must contain both target classes")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_frame["review_text"])
    validation_matrix = vectorizer.transform(validation_frame["review_text"])
    test_matrix = vectorizer.transform(test_frame["review_text"])

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        random_state=59990,
        solver="liblinear",
    )
    classifier.fit(train_matrix, train_frame["low_rating"])
    validation_probabilities = classifier.predict_proba(validation_matrix)[:, 1]
    threshold = select_threshold(
        validation_frame["low_rating"].to_numpy(), validation_probabilities
    )
    validation_metrics, _ = evaluate_binary(
        validation_frame["low_rating"].to_numpy(),
        validation_probabilities,
        threshold,
    )
    test_probabilities = classifier.predict_proba(test_matrix)[:, 1]
    test_metrics, test_predictions = evaluate_binary(
        test_frame["low_rating"].to_numpy(), test_probabilities, threshold
    )

    metrics: dict[str, object] = {
        "job_name": args.job_name,
        "model": "TF-IDF word unigrams/bigrams + class-weighted logistic regression",
        "target": "low_rating = 1 when rating <= 2",
        "random_state": 59990,
        "threshold_selected_on_validation": round(threshold, 4),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "validation_f1": validation_metrics["f1"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "baseline_accuracy": test_metrics["baseline_accuracy"],
        "true_negative": test_metrics["true_negative"],
        "false_positive": test_metrics["false_positive"],
        "false_negative": test_metrics["false_negative"],
        "true_positive": test_metrics["true_positive"],
    }

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_data_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    confusion_path = output_dir / "confusion_matrix.csv"

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "threshold": threshold,
            "target": metrics["target"],
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "review_id": test_frame["review_id"].astype(str),
            "user_id": test_frame["user_id"].astype(str),
            "parent_asin": test_frame["parent_asin"].astype(str),
            "rating": test_frame["rating"].astype(float),
            "actual_low_rating": test_frame["low_rating"].astype(int),
            "predicted_low_rating": test_predictions,
            "probability_low_rating": np.round(test_probabilities, 8),
        }
    ).to_csv(predictions_path, index=False)
    pd.DataFrame(
        [
            [test_metrics["true_negative"], test_metrics["false_positive"]],
            [test_metrics["false_negative"], test_metrics["true_positive"]],
        ],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(confusion_path)

    output_root = args.output_s3_uri.rstrip("/")
    _upload_file(metrics_path, f"{output_root}/metrics/metrics.json")
    _upload_file(predictions_path, f"{output_root}/predictions/test_predictions.csv")
    _upload_file(confusion_path, f"{output_root}/evidence/confusion_matrix.csv")

    print(f"METRIC test_accuracy={metrics['accuracy']}", flush=True)
    print(f"METRIC test_precision={metrics['precision']}", flush=True)
    print(f"METRIC test_recall={metrics['recall']}", flush=True)
    print(f"METRIC test_f1={metrics['f1']}", flush=True)
    print(f"METRIC test_roc_auc={metrics['roc_auc']}", flush=True)
    print("METRICS_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the binary low-rating model")
    parser.add_argument("--train", default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test", default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--model-dir", default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--output-data-dir",
        default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output/data"),
    )
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
