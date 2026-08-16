# AWS versus GCP trade-offs

| Project responsibility | AWS implementation | Comparable GCP approach |
|---|---|---|
| Object storage | Amazon S3 | Cloud Storage |
| ETL and distributed validation | AWS Glue | Dataflow or Dataproc |
| Shared metadata | Glue Data Catalog | Dataplex Universal Catalog / BigQuery metadata |
| Serverless SQL | Athena | BigQuery external or managed tables |
| Managed ML training | SageMaker | Vertex AI |
| Business intelligence | QuickSight | Looker or Looker Studio |

## Why AWS fit this build

AWS kept the evidence in one S3 data lake while allowing Glue, Athena, SageMaker, and QuickSight to share cataloged schemas. The group could scale each compute service independently and avoid a persistent database or inference endpoint. Athena also provided a direct path from Parquet tables to both validation queries and QuickSight.

## Where GCP could be attractive

BigQuery provides a highly integrated warehouse and SQL experience, while Vertex AI and Looker can form a strong managed analytics stack. A BigQuery-centered design might reduce the distinction between external-table querying and a separate BI import layer. However, moving this project would require rebuilding IAM, metadata, SQL dialect details, orchestration, evidence procedures, and dashboard connections. Given that the implemented services, roles, data locations, and validation scripts were already AWS-native, AWS minimized delivery risk for the final project.

The trade-off is not that one cloud is universally better. AWS was the lower-risk choice for this implemented architecture; GCP would be reasonable if the organization standardized on BigQuery, Vertex AI, and Looker or had stronger GCP governance and skills.
