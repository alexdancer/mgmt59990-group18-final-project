-- Replace __BUCKET__ and __JOB_NAME__ before execution.
-- Run each statement separately in the validated Athena checkpoint workgroup.

CREATE EXTERNAL TABLE group18_reviews.model_predictions (
  review_id string,
  user_id string,
  parent_asin string,
  rating double,
  actual_low_rating int,
  predicted_low_rating int,
  probability_low_rating double
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\'
)
LOCATION 's3://__BUCKET__/gold/model_outputs/__JOB_NAME__/predictions/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- Reconcile the Athena prediction table to the SageMaker confusion matrix.
SELECT
  COUNT(*) AS prediction_rows,
  SUM(CASE WHEN actual_low_rating = 0 AND predicted_low_rating = 0 THEN 1 ELSE 0 END) AS true_negative,
  SUM(CASE WHEN actual_low_rating = 0 AND predicted_low_rating = 1 THEN 1 ELSE 0 END) AS false_positive,
  SUM(CASE WHEN actual_low_rating = 1 AND predicted_low_rating = 0 THEN 1 ELSE 0 END) AS false_negative,
  SUM(CASE WHEN actual_low_rating = 1 AND predicted_low_rating = 1 THEN 1 ELSE 0 END) AS true_positive,
  SUM(
    CASE
      WHEN actual_low_rating IN (0, 1) AND predicted_low_rating IN (0, 1) THEN 1
      ELSE 0
    END
  ) AS confusion_total,
  ROUND(AVG(probability_low_rating), 6) AS average_low_rating_probability
FROM group18_reviews.model_predictions;

-- Business-facing products with the highest predicted low-rating risk.
SELECT
  p.parent_asin,
  d.product_title,
  d.main_category,
  COUNT(*) AS test_review_count,
  SUM(p.predicted_low_rating) AS predicted_low_rating_count,
  ROUND(100.0 * AVG(CAST(p.predicted_low_rating AS double)), 2) AS predicted_low_rating_pct,
  ROUND(AVG(p.probability_low_rating), 4) AS average_low_rating_probability
FROM group18_reviews.model_predictions AS p
JOIN group18_reviews.dim_product AS d
  ON p.parent_asin = d.parent_asin
GROUP BY p.parent_asin, d.product_title, d.main_category
HAVING COUNT(*) >= 3
ORDER BY average_low_rating_probability DESC, test_review_count DESC
LIMIT 25;
