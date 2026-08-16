-- Replace ${BUCKET} with the existing project bucket name before execution.
CREATE DATABASE IF NOT EXISTS group18_reviews;

CREATE EXTERNAL TABLE IF NOT EXISTS group18_reviews.fact_review (
  review_id string,
  rating tinyint,
  title_clean string,
  text_clean string,
  review_text string,
  asin string,
  parent_asin string,
  user_id string,
  review_ts timestamp,
  review_month string,
  helpful_vote bigint,
  verified_purchase boolean,
  text_length integer
)
PARTITIONED BY (
  category string,
  review_year smallint
)
STORED AS PARQUET
LOCATION 's3://${BUCKET}/silver/fact_review/'
TBLPROPERTIES (
  'classification'='parquet',
  'parquet.compression'='SNAPPY'
);

MSCK REPAIR TABLE group18_reviews.fact_review;

CREATE EXTERNAL TABLE IF NOT EXISTS group18_reviews.dim_product (
  parent_asin string,
  product_title string,
  main_category string,
  price double,
  price_status string,
  average_rating double,
  rating_number bigint,
  store string
)
STORED AS PARQUET
LOCATION 's3://${BUCKET}/silver/dim_product/'
TBLPROPERTIES (
  'classification'='parquet',
  'parquet.compression'='SNAPPY'
);
