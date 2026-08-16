# Cost controls and cleanup

## Controls used during implementation

- The source was sampled instead of processing the entire 22.6 GB review file.
- Silver and Gold data use Parquet with Snappy compression to reduce Athena scans.
- The `group18-gold-build` workgroup has a 256 MiB (`268,435,456` byte) scan cutoff per query.
- Athena result files are written to the project bucket rather than an unrelated default location.
- The SageMaker job used one `ml.m5.xlarge` instance, a 20 GB volume, and `MaxRuntimeInSeconds = 1800`.
- The completed training job consumed 89 billable seconds.
- No persistent SageMaker endpoint was created.
- QuickSight imported three small datasets into SPICE to avoid repeated dashboard scans.
- AWS resources use project tags for cost attribution.

## Cost interpretation

The project is designed as a short-lived classroom workload. S3 usage is below the scale of a complete category archive, Athena scans are bounded, Glue and SageMaker compute run only for discrete jobs, and there is no continuously billed inference endpoint. Exact dollar amounts depend on the regional rates and subscription terms in effect when the jobs run, so the report should use the AWS Pricing Calculator with the recorded usage in `evidence/cost_controls.json`.

## Post-submission review

1. Preserve the final report, screenshots, dashboard PDF, query outputs, and model metrics.
2. Confirm that no Glue or SageMaker job is still running.
3. Confirm that no SageMaker endpoint or notebook instance exists.
4. Review the QuickSight subscription immediately after submission; its free trial or author subscription can become recurring.
5. Remove unneeded Athena result files, temporary S3 prefixes, duplicate source archives, and model artifacts after evidence is preserved.
6. Retain only the S3 objects required by the instructor or group.
7. Review AWS Budgets and Cost Explorer for unexpected charges.

Cleanup is intentionally a reviewed manual action because deletion is difficult to reverse and evidence must be retained through grading.
