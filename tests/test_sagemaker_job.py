import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_job_module():
    module_path = PROJECT_ROOT / "aws" / "sagemaker" / "build_training_job.py"
    spec = importlib.util.spec_from_file_location("sagemaker_job", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sagemaker_job_builder_exists():
    assert (PROJECT_ROOT / "aws" / "sagemaker" / "build_training_job.py").is_file()


def test_job_uses_three_gold_splits_and_one_bounded_cpu_instance():
    job_builder = load_job_module()

    request = job_builder.build_training_job(
        job_name="mgmt59990-group18-low-rating-20260813-120000",
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole-MGMT59990-Group18",
        image_uri="257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3",
        source_uri="s3://project/scripts/sagemaker/source.tar.gz",
        gold_training_uri="s3://project/gold/model_training/",
        model_artifacts_uri="s3://project/model-artifacts/",
        model_output_uri="s3://project/gold/model_outputs/job/",
    )

    channels = {
        channel["ChannelName"]: channel["DataSource"]["S3DataSource"]["S3Uri"]
        for channel in request["InputDataConfig"]
    }
    assert channels == {
        "train": "s3://project/gold/model_training/split_name=train/",
        "validation": "s3://project/gold/model_training/split_name=validation/",
        "test": "s3://project/gold/model_training/split_name=test/",
    }
    assert request["ResourceConfig"] == {
        "InstanceType": "ml.m5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 20,
    }
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 1800}
    assert request["AlgorithmSpecification"]["TrainingInputMode"] == "File"
    assert request["HyperParameters"]["sagemaker_program"] == "train.py"
    assert request["HyperParameters"]["output-s3-uri"].endswith(
        "/gold/model_outputs/job/"
    )


def test_job_captures_named_test_metrics_for_sagemaker_evidence():
    job_builder = load_job_module()
    request = job_builder.build_training_job(
        job_name="job",
        role_arn="role",
        image_uri="image",
        source_uri="s3://project/source.tar.gz",
        gold_training_uri="s3://project/gold/model_training/",
        model_artifacts_uri="s3://project/model-artifacts/",
        model_output_uri="s3://project/gold/model_outputs/job/",
    )

    metric_names = {
        metric["Name"]
        for metric in request["AlgorithmSpecification"]["MetricDefinitions"]
    }
    assert metric_names == {
        "test:accuracy",
        "test:precision",
        "test:recall",
        "test:f1",
        "test:roc_auc",
    }
    assert request["AlgorithmSpecification"]["EnableSageMakerMetricsTimeSeries"] is True


def test_job_builder_cli_writes_the_aws_request(tmp_path):
    script = PROJECT_ROOT / "aws" / "sagemaker" / "build_training_job.py"
    output = tmp_path / "training-job.json"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--job-name",
            "job",
            "--role-arn",
            "role",
            "--image-uri",
            "image",
            "--source-uri",
            "s3://project/source.tar.gz",
            "--gold-training-uri",
            "s3://project/gold/model_training/",
            "--model-artifacts-uri",
            "s3://project/model-artifacts/",
            "--model-output-uri",
            "s3://project/gold/model_outputs/job/",
            "--output",
            str(output),
        ],
        check=True,
    )

    assert json.loads(output.read_text(encoding="utf-8"))["TrainingJobName"] == "job"


def test_job_is_tagged_for_project_cost_attribution():
    job_builder = load_job_module()
    request = job_builder.build_training_job(
        job_name="job",
        role_arn="role",
        image_uri="image",
        source_uri="s3://project/source.tar.gz",
        gold_training_uri="s3://project/gold/model_training/",
        model_artifacts_uri="s3://project/model-artifacts/",
        model_output_uri="s3://project/gold/model_outputs/job/",
    )

    assert {tag["Key"]: tag["Value"] for tag in request["Tags"]} == {
        "Project": "MGMT59990-Group18",
        "Course": "MGMT59990",
        "Layer": "Gold",
        "Purpose": "LowRatingModel",
    }
