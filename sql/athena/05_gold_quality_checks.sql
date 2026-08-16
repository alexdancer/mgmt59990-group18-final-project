-- Gold reconciliation and model-split integrity. Expected invariants:
-- 66,004 fact, aggregate, and training rows; 66,004 distinct training review
-- IDs; zero missing fields, invalid labels, or users in multiple splits; and
-- nonempty train, validation, and test splits.
WITH
fact_summary AS (
  SELECT COUNT(*) AS fact_rows
  FROM group18_reviews.fact_review
  WHERE category = 'Electronics'
),
rating_summary AS (
  SELECT
    COALESCE(SUM(review_count), 0) AS rating_aggregate_review_rows,
    COALESCE(
      SUM(
        one_star_count
        + two_star_count
        + three_star_count
        + four_star_count
        + five_star_count
      ),
      0
    ) AS rating_star_count_rows
  FROM group18_reviews.agg_rating_monthly
),
product_summary AS (
  SELECT COALESCE(SUM(review_count), 0) AS product_aggregate_review_rows
  FROM group18_reviews.agg_product_monthly
),
training_summary AS (
  SELECT
    COUNT(*) AS training_rows,
    COUNT(DISTINCT review_id) AS distinct_training_review_ids,
    SUM(
      CASE
        WHEN review_id IS NULL OR review_id = ''
          OR user_id IS NULL OR user_id = ''
          OR parent_asin IS NULL OR parent_asin = ''
          OR review_text IS NULL OR length(review_text) = 0
          OR rating NOT BETWEEN 1 AND 5
          OR low_rating NOT IN (0, 1)
          OR split_name NOT IN ('train', 'validation', 'test')
        THEN 1 ELSE 0
      END
    ) AS missing_training_fields,
    SUM(
      CASE
        WHEN low_rating <> CASE WHEN rating <= 2 THEN 1 ELSE 0 END
        THEN 1 ELSE 0
      END
    ) AS invalid_low_rating_labels,
    SUM(CASE WHEN split_name = 'train' THEN 1 ELSE 0 END) AS train_rows,
    SUM(CASE WHEN split_name = 'validation' THEN 1 ELSE 0 END) AS validation_rows,
    SUM(CASE WHEN split_name = 'test' THEN 1 ELSE 0 END) AS test_rows
  FROM group18_reviews.model_training
),
leaked_users AS (
  SELECT user_id
  FROM group18_reviews.model_training
  GROUP BY user_id
  HAVING COUNT(DISTINCT split_name) > 1
),
leakage_summary AS (
  SELECT COUNT(*) AS users_in_multiple_splits
  FROM leaked_users
)
SELECT
  f.fact_rows,
  r.rating_aggregate_review_rows,
  r.rating_star_count_rows,
  p.product_aggregate_review_rows,
  t.training_rows,
  t.distinct_training_review_ids,
  t.missing_training_fields,
  t.invalid_low_rating_labels,
  l.users_in_multiple_splits,
  t.train_rows,
  t.validation_rows,
  t.test_rows
FROM fact_summary AS f
CROSS JOIN rating_summary AS r
CROSS JOIN product_summary AS p
CROSS JOIN training_summary AS t
CROSS JOIN leakage_summary AS l;
