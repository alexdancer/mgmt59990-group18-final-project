"""AWS Glue 5.1 PySpark job for the Bronze-to-Silver star schema.

Expected job arguments:
  --REVIEWS_PATH s3://<bucket>/bronze/reviews/category=Electronics/
  --METADATA_PATH s3://<bucket>/bronze/metadata/category=Electronics/
  --SILVER_ROOT_PATH s3://<bucket>/silver/

The Bronze prefixes also contain JSON manifests. ``pathGlobFilter`` ensures
that Spark reads only the JSONL data objects.
"""

from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "REVIEWS_PATH", "METADATA_PATH", "SILVER_ROOT_PATH"],
)
context = GlueContext(SparkContext.getOrCreate())
spark = context.spark_session
spark.conf.set("spark.sql.session.timeZone", "UTC")
job = Job(context)
job.init(args["JOB_NAME"], args)


def normalized_text(column_name: str) -> F.Column:
    """Normalize whitespace while preserving the source text content."""
    return F.trim(
        F.regexp_replace(
            F.coalesce(F.col(column_name).cast("string"), F.lit("")),
            r"\s+",
            " ",
        )
    )


def build_fact_review(reviews: DataFrame) -> DataFrame:
    prepared = (
        reviews.select(
            F.col("rating").cast("double").alias("rating_raw"),
            normalized_text("title").alias("title_clean"),
            normalized_text("text").alias("text_clean"),
            F.trim(F.col("asin").cast("string")).alias("asin"),
            F.trim(F.col("parent_asin").cast("string")).alias("parent_asin"),
            F.trim(F.col("user_id").cast("string")).alias("user_id"),
            F.col("timestamp").cast("long").alias("timestamp_ms"),
            F.coalesce(F.col("helpful_vote").cast("long"), F.lit(0)).alias(
                "helpful_vote"
            ),
            F.col("verified_purchase").cast("boolean").alias("verified_purchase"),
        )
        .withColumn(
            "review_text",
            F.regexp_replace(
                F.concat(F.col("title_clean"), F.lit(". "), F.col("text_clean")),
                r"^[. ]+|[. ]+$",
                "",
            ),
        )
        .withColumn(
            "review_ts",
            F.to_timestamp(F.from_unixtime(F.col("timestamp_ms") / F.lit(1000))),
        )
        .filter(F.col("rating_raw").isin(1.0, 2.0, 3.0, 4.0, 5.0))
        .filter(F.length("review_text") > 0)
        .filter(F.length("parent_asin") > 0)
        .filter(F.length("user_id") > 0)
        .filter(F.col("review_ts").isNotNull())
        .withColumn("rating", F.col("rating_raw").cast("tinyint"))
        .withColumn("category", F.lit("Electronics"))
        .withColumn("review_year", F.year("review_ts").cast("smallint"))
        .withColumn("review_month", F.date_format("review_ts", "yyyy-MM"))
        .withColumn("text_length", F.length("review_text"))
        .withColumn(
            "review_id",
            F.substring(
                F.sha2(
                    F.concat_ws(
                        "|",
                        F.col("user_id"),
                        F.col("parent_asin"),
                        F.col("timestamp_ms").cast("string"),
                        F.col("review_text"),
                    ),
                    256,
                ),
                1,
                24,
            ),
        )
        .dropDuplicates(["review_id"])
    )

    return prepared.select(
        "review_id",
        "rating",
        "title_clean",
        "text_clean",
        "review_text",
        "asin",
        "parent_asin",
        "user_id",
        "review_ts",
        "review_month",
        "helpful_vote",
        "verified_purchase",
        "text_length",
        "category",
        "review_year",
    )


def build_dim_product(metadata: DataFrame) -> DataFrame:
    raw_price = F.trim(F.col("price").cast("string"))
    parsed_price = raw_price.cast("double")
    missing_price = F.col("price").isNull() | F.lower(raw_price).isin(
        "", "none", "null", "nan"
    )
    valid_price = parsed_price.isNotNull() & (parsed_price >= 0)

    return (
        metadata.select(
            F.trim(F.col("parent_asin").cast("string")).alias("parent_asin"),
            F.trim(F.col("title").cast("string")).alias("product_title"),
            F.trim(F.col("main_category").cast("string")).alias("main_category"),
            F.when(valid_price, parsed_price).cast("double").alias("price"),
            F.when(missing_price, F.lit("missing"))
            .when(valid_price, F.lit("valid"))
            .otherwise(F.lit("invalid"))
            .alias("price_status"),
            F.col("average_rating").cast("double").alias("average_rating"),
            F.col("rating_number").cast("long").alias("rating_number"),
            F.trim(F.col("store").cast("string")).alias("store"),
        )
        .filter(F.length("parent_asin") > 0)
        .dropDuplicates(["parent_asin"])
    )


reviews = (
    spark.read.option("pathGlobFilter", "*.jsonl")
    .json(args["REVIEWS_PATH"])
)
metadata = (
    spark.read.option("pathGlobFilter", "*.jsonl")
    .json(args["METADATA_PATH"])
)

fact_review = build_fact_review(reviews)
dim_product = build_dim_product(metadata)

# The validated local Bronze artifacts have complete parent-ASIN coverage. Fail
# before writing Silver if AWS Bronze does not preserve that contract.
unmatched_reviews = fact_review.join(
    dim_product.select("parent_asin"), on="parent_asin", how="left_anti"
).count()
if unmatched_reviews:
    raise RuntimeError(
        f"Metadata coverage validation failed: {unmatched_reviews} reviews have no product"
    )

silver_root = args["SILVER_ROOT_PATH"].rstrip("/")
fact_review_path = f"{silver_root}/fact_review/"
dim_product_path = f"{silver_root}/dim_product/"

(
    fact_review.write.mode("overwrite")
    .partitionBy("category", "review_year")
    .parquet(fact_review_path)
)
dim_product.write.mode("overwrite").parquet(dim_product_path)

print(
    {
        "fact_review_rows": fact_review.count(),
        "dim_product_rows": dim_product.count(),
        "unmatched_review_rows": unmatched_reviews,
        "fact_review_path": fact_review_path,
        "dim_product_path": dim_product_path,
    }
)

job.commit()
