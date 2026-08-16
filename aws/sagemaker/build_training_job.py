"""Build a deterministic AWS CLI request for the SageMaker training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _root(uri: str) -> str:
    return uri.rstrip("/") + "/"


def build_training_job(
    *,
    job_name: str,
    role_arn: str,
    image_uri: str,
    source_uri: str,
    gold_training_uri: str,
    model_artifacts_uri: str,
    model_output_uri: str,
) -> dict[str, object]:
    """Return the bounded CreateTrainingJob request used by the AWS CLI."""
    gold_root = _root(gold_training_uri)
    channels = []
    for split_name in ("train", "validation", "test"):
        channels.append(
            {
                "ChannelName": split_name,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"{gold_root}split_name={split_name}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            }
        )
    return {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": True,
            "MetricDefinitions": [
                {
                    "Name": f"test:{metric_name}",
                    "Regex": rf"METRIC test_{metric_name}=([0-9.]+)",
                }
                for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc")
            ],
        },
        "HyperParameters": {
            "sagemaker_program": "train.py",
            "sagemaker_submit_directory": source_uri,
            "job-name": job_name,
            "output-s3-uri": _root(model_output_uri),
            "max-features": "50000",
            "min-df": "2",
        },
        "InputDataConfig": channels,
        "OutputDataConfig": {"S3OutputPath": _root(model_artifacts_uri)},
        "ResourceConfig": {
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 20,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
        "Tags": [
            {"Key": "Project", "Value": "MGMT59990-Group18"},
            {"Key": "Course", "Value": "MGMT59990"},
            {"Key": "Layer", "Value": "Gold"},
            {"Key": "Purpose", "Value": "LowRatingModel"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--gold-training-uri", required=True)
    parser.add_argument("--model-artifacts-uri", required=True)
    parser.add_argument("--model-output-uri", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = build_training_job(
        job_name=args.job_name,
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        source_uri=args.source_uri,
        gold_training_uri=args.gold_training_uri,
        model_artifacts_uri=args.model_artifacts_uri,
        model_output_uri=args.model_output_uri,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
