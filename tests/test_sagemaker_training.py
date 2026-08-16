import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_training_module():
    module_path = PROJECT_ROOT / "aws" / "sagemaker" / "train.py"
    spec = importlib.util.spec_from_file_location("sagemaker_train", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sagemaker_training_entrypoint_exists():
    assert (PROJECT_ROOT / "aws" / "sagemaker" / "train.py").is_file()


def test_select_threshold_uses_validation_f1_with_a_half_threshold_tiebreak():
    training = load_training_module()
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.8])

    assert training.select_threshold(labels, probabilities) == 0.5


def test_binary_evaluation_reports_business_and_confusion_metrics():
    training = load_training_module()
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics, predictions = training.evaluate_binary(labels, probabilities, 0.5)

    assert predictions.tolist() == [0, 0, 1, 1]
    assert metrics == {
        "rows": 4,
        "actual_positive_rows": 2,
        "predicted_positive_rows": 2,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "baseline_accuracy": 0.5,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_training_honors_gold_splits_and_publishes_auditable_outputs():
    source = (
        PROJECT_ROOT / "aws" / "sagemaker" / "train.py"
    ).read_text(encoding="utf-8")

    for channel_variable in (
        "SM_CHANNEL_TRAIN",
        "SM_CHANNEL_VALIDATION",
        "SM_CHANNEL_TEST",
    ):
        assert channel_variable in source
    assert "TfidfVectorizer" in source
    assert "LogisticRegression" in source
    assert 'class_weight="balanced"' in source
    assert "test_predictions.csv" in source
    assert "metrics.json" in source
    assert "model.joblib" in source
    assert "METRIC test_f1=" in source
    assert "train_test_split" not in source


def test_training_source_declares_parquet_runtime_dependency():
    requirements = (
        PROJECT_ROOT / "aws" / "sagemaker" / "requirements.txt"
    ).read_text(encoding="utf-8")

    assert "pyarrow" in requirements
