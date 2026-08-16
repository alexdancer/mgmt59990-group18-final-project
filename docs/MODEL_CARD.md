# Low-rating classifier model card

## Intended use

Estimate the probability that an Amazon Electronics review will receive one or two stars and prioritize likely low-rating reviews for human follow-up.

## Out-of-scope use

The model should not automatically remove products, penalize sellers, resolve customer claims, or make safety decisions without review of the underlying evidence.

## Training method

- **Input:** normalized review title and body text.
- **Target:** `low_rating = 1` when `rating <= 2`.
- **Features:** TF-IDF-weighted word unigrams and bigrams.
- **Classifier:** logistic regression with balanced class weights.
- **Splitting:** deterministic user-level hashing assigns every reviewer to train, validation, or test, preventing user leakage.
- **Threshold:** selected on validation data for F1, then applied once to the held-out test set.
- **Runtime:** one SageMaker `ml.m5.xlarge` CPU instance with a 30-minute hard cap; the completed job ran for 89 seconds.

## Test metrics

See `evidence/model_metrics.json` for the machine-readable values and [Results](RESULTS.md) for interpretation.

## Monitoring and production readiness

Production use would require incremental data ingestion, versioned training datasets and models, temporal holdouts, calibration checks, data-quality and drift alarms, a model-approval step, scheduled batch scoring, rollback, and documented human escalation procedures. A real-time endpoint should be introduced only if the business latency requirement justifies persistent compute.
