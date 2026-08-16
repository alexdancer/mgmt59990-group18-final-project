# Amazon Electronics Review Risk Analytics

MGMT 59990 · Group 18 · Final Cloud Analytics Project

This project turns Amazon Electronics reviews into product-level risk signals. It implements a governed AWS data lake, validates the data with serverless SQL, trains a text-classification model, and publishes descriptive and predictive results in Amazon QuickSight.

![Implemented AWS architecture](docs/images/aws_architecture.png)

## Business problem

Product and customer-experience teams cannot consistently read tens of thousands of reviews quickly enough to identify emerging quality problems. The project addresses two related questions:

1. Which products have concentrated patterns of one- and two-star reviews?
2. Which new reviews are most likely to be low rated and should be prioritized for human follow-up?

The model is a prioritization aid, not an automated product-removal or customer-action system.

## What was implemented

- A Bronze, Silver, and Gold data lake in Amazon S3.
- An AWS Glue PySpark job that validates reviews, joins product metadata, removes duplicates, and writes Parquet tables.
- Glue Data Catalog tables for `fact_review`, `dim_product`, aggregates, training data, and model predictions.
- Athena quality checks, analysis SQL, and CTAS statements for three Gold data products.
- User-isolated train, validation, and test partitions to prevent one reviewer's language from crossing model splits.
- A bounded SageMaker training job using TF-IDF text features and class-weighted logistic regression.
- A published dashboard in QuickSight backed by three SPICE datasets for portfolio trends and product risk.

See [Architecture](docs/ARCHITECTURE.md) for the service responsibilities and data flow.

## Key results

| Result | Value |
|---|---:|
| Validated review records | 66,004 |
| Matched products | 39,476 |
| Distinct reviewers | 27,359 |
| One- or two-star reviews | 11,408 (17.28%) |
| Held-out test reviews | 6,436 |
| Test recall for low ratings | 87.17% |
| Test precision | 78.60% |
| Test F1 | 82.66% |
| Test ROC AUC | 97.57% |
| Test accuracy | 93.49% |

Recall is emphasized because missing a genuinely low-rated review is more costly for an early-warning workflow than sending some additional reviews for human inspection. Detailed interpretation and limitations are in [Results](docs/RESULTS.md) and the [Model card](docs/MODEL_CARD.md).

## Repository map

```text
.
├── aws/                    # Glue ETL and SageMaker training code
├── docs/                   # Architecture, results, cost, trade-offs, and AI-use note
├── evidence/               # Selected small outputs supporting implementation claims
├── presentation/           # Final presentation deck and presentation-link placeholder
├── report/                 # Location for the final report PDF before publication
├── scripts/                # Small local checkpoint runner
├── sql/athena/             # DDL, quality checks, Gold CTAS, and model-risk SQL
├── src/                    # Local AWS-compatible sampling and transformation pipeline
├── tests/                  # Data-contract and model-job tests
├── .env.example            # Non-secret AWS resource-name template
├── pyproject.toml          # Python package and runtime dependencies
└── uv.lock                 # Reproducible Python dependency lock
```

## Local setup

Requirements:

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/)
- Internet access only when downloading a source sample

```bash
uv sync
uv run reviews-random-sample --ranges 32 --range-bytes 1048576 --seed 59990
uv run reviews-metadata-subset
uv run reviews-transform
uv run reviews-model
uv run reviews-evidence
```

For the smaller checkpoint workflow:

```bash
./scripts/run_checkpoint.sh
```

Raw source files and generated data are written under `data/` and are deliberately excluded from GitHub.

## Running tests

Install the test runner in the project environment, then execute:

```bash
uv pip install -r requirements-dev.txt
uv run pytest -q
```

## AWS implementation materials

The repository focuses on the implementation itself rather than guided deployment tooling:

- `aws/glue_etl.py` contains the Bronze-to-Silver Glue transformation.
- `sql/athena/` contains table definitions, quality checks, Gold CTAS statements, and dashboard/model queries.
- `aws/sagemaker/` contains the bounded training-job request builder and training entry point.
- `.env.example` documents the non-secret resource names used by the architecture.

The AWS services were configured and validated through the AWS Console and CloudShell. Never place AWS access keys in `.env`, source files, or GitHub.

## Evidence and interpretation

- `evidence/athena_quality_results.csv` demonstrates the local equivalent of the row-level Silver quality contract.
- `evidence/model_metrics.json` and `evidence/confusion_matrix.png` document held-out model performance.
- `evidence/quicksight_product_risk.png` and `evidence/quicksight_dashboard.pdf` show the published business-facing analytics.
- `evidence/cost_controls.json` records the bounded training and query controls.

Evidence is intentionally selected rather than dumping logs or cloud metadata into the repository. AWS account identifiers, credentials, raw customer-level data, and temporary outputs are excluded.

## Cost and platform decisions

The build uses short-lived or serverless compute: Athena queries are scan-limited, the SageMaker job had a 30-minute hard cap and completed in 89 seconds, and no persistent model endpoint was created. QuickSight uses imported SPICE datasets and its subscription must be reviewed after submission. See [Cost and cleanup](docs/COST_AND_CLEANUP.md) and [AWS versus GCP](docs/AWS_VS_GCP.md).

## Dataset

The source is the Amazon Reviews 2023 Electronics review and product-metadata collection published by the McAuley Lab. The project used a seeded distributed byte-range sample across the source rather than publishing the multi-gigabyte source file in this repository.

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.

## Submission materials

- [Final presentation deck](presentation/MGMT59990_Group18_Final_Presentation.pptx)
- Final report: add the approved PDF under `report/` before publishing
- Recorded presentation link: add it to `presentation/README.md` before publishing

## Generative AI disclosure

Generative AI supported brainstorming, debugging, code and SQL review, documentation editing, and presentation development. The group executed the AWS steps, inspected the resulting services and files, reconciled the row counts, reviewed model metrics, and approved the final claims. See [Generative AI use](docs/GENERATIVE_AI_USE.md).
