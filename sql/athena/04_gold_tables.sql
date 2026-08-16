-- Replace __BUCKET__ with the project bucket before execution.
-- Run each CTAS statement separately in the non-enforcing Gold-build workgroup.
-- Every external_location must be empty, and the target table must not exist.

-- Dashboard-ready monthly rating metrics. The individual star counts support
-- both the monthly trend and an overall rating-distribution visualization.
CREATE TABLE group18_reviews.agg_rating_monthly
WITH (
  format = 'PARQUET',
  write_compression = 'SNAPPY',
  external_location = 's3://__BUCKET__/gold/agg_rating_monthly/'
)
AS
SELECT
  review_month,
  review_year,
  COUNT(*) AS review_count,
  ROUND(AVG(rating), 4) AS average_rating,
  SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS low_rating_count,
  ROUND(100.0 * AVG(CASE WHEN rating <= 2 THEN 1.0 ELSE 0.0 END), 4) AS low_rating_pct,
  SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS one_star_count,
  SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) AS two_star_count,
  SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) AS three_star_count,
  SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) AS four_star_count,
  SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) AS five_star_count,
  SUM(CASE WHEN verified_purchase THEN 1 ELSE 0 END) AS verified_purchase_count,
  ROUND(100.0 * AVG(CASE WHEN verified_purchase THEN 1.0 ELSE 0.0 END), 4) AS verified_purchase_pct,
  SUM(helpful_vote) AS helpful_vote_count
FROM group18_reviews.fact_review
WHERE category = 'Electronics'
GROUP BY review_month, review_year;

-- Product-by-month dashboard metrics. No minimum-review filter is applied here
-- so the Gold product table still reconciles to all 66,004 Silver fact rows.
CREATE TABLE group18_reviews.agg_product_monthly
WITH (
  format = 'PARQUET',
  write_compression = 'SNAPPY',
  external_location = 's3://__BUCKET__/gold/agg_product_monthly/'
)
AS
SELECT
  f.review_month,
  f.review_year,
  f.parent_asin,
  d.product_title,
  d.main_category,
  d.store,
  d.price,
  d.price_status,
  COUNT(*) AS review_count,
  ROUND(AVG(f.rating), 4) AS average_rating,
  SUM(CASE WHEN f.rating <= 2 THEN 1 ELSE 0 END) AS low_rating_count,
  ROUND(100.0 * AVG(CASE WHEN f.rating <= 2 THEN 1.0 ELSE 0.0 END), 4) AS low_rating_pct,
  SUM(CASE WHEN f.verified_purchase THEN 1 ELSE 0 END) AS verified_purchase_count,
  ROUND(AVG(f.helpful_vote), 4) AS average_helpful_votes,
  SUM(f.helpful_vote) AS helpful_vote_count
FROM group18_reviews.fact_review AS f
JOIN group18_reviews.dim_product AS d
  ON f.parent_asin = d.parent_asin
WHERE f.category = 'Electronics'
GROUP BY
  f.review_month,
  f.review_year,
  f.parent_asin,
  d.product_title,
  d.main_category,
  d.store,
  d.price,
  d.price_status;

-- Binary-model training extract. Bucketing by a deterministic user hash keeps
-- every review from the same user in one split. The partition column is last
-- in the SELECT list, as required for a partitioned Athena CTAS statement.
CREATE TABLE group18_reviews.model_training
WITH (
  format = 'PARQUET',
  write_compression = 'SNAPPY',
  external_location = 's3://__BUCKET__/gold/model_training/',
  partitioned_by = ARRAY['split_name']
)
AS
WITH prepared AS (
  SELECT
    review_id,
    user_id,
    parent_asin,
    review_text,
    rating,
    CASE WHEN rating <= 2 THEN 1 ELSE 0 END AS low_rating,
    MOD(crc32(to_utf8(user_id)), 10) AS user_split_bucket,
    review_year
  FROM group18_reviews.fact_review
  WHERE category = 'Electronics'
)
SELECT
  review_id,
  user_id,
  parent_asin,
  review_text,
  rating,
  low_rating,
  user_split_bucket,
  review_year,
  CASE
    WHEN user_split_bucket <= 7 THEN 'train'
    WHEN user_split_bucket = 8 THEN 'validation'
    ELSE 'test'
  END AS split_name
FROM prepared;
