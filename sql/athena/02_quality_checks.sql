-- Review fact integrity.
SELECT
  COUNT(*) AS review_rows,
  COUNT(DISTINCT review_id) AS distinct_review_ids,
  SUM(CASE WHEN review_id IS NULL OR review_id = '' THEN 1 ELSE 0 END) AS missing_review_id,
  SUM(CASE WHEN parent_asin IS NULL OR parent_asin = '' THEN 1 ELSE 0 END) AS missing_parent_asin,
  SUM(CASE WHEN user_id IS NULL OR user_id = '' THEN 1 ELSE 0 END) AS missing_user_id,
  SUM(CASE WHEN review_text IS NULL OR length(review_text) = 0 THEN 1 ELSE 0 END) AS empty_review_text,
  SUM(CASE WHEN rating NOT BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS invalid_rating,
  SUM(CASE WHEN review_ts IS NULL THEN 1 ELSE 0 END) AS missing_review_ts,
  MIN(review_ts) AS earliest_review,
  MAX(review_ts) AS latest_review
FROM group18_reviews.fact_review
WHERE category = 'Electronics';

-- Product dimension integrity and price parsing outcomes.
SELECT
  COUNT(*) AS product_rows,
  COUNT(DISTINCT parent_asin) AS distinct_parent_asins,
  SUM(CASE WHEN parent_asin IS NULL OR parent_asin = '' THEN 1 ELSE 0 END) AS missing_parent_asin,
  SUM(CASE WHEN product_title IS NULL OR product_title = '' THEN 1 ELSE 0 END) AS missing_product_title,
  SUM(CASE WHEN main_category IS NULL OR main_category = '' THEN 1 ELSE 0 END) AS missing_main_category,
  SUM(CASE WHEN price_status = 'valid' THEN 1 ELSE 0 END) AS valid_price_rows,
  SUM(CASE WHEN price_status = 'missing' THEN 1 ELSE 0 END) AS missing_price_rows,
  SUM(CASE WHEN price_status = 'invalid' THEN 1 ELSE 0 END) AS invalid_price_rows,
  SUM(CASE WHEN price < 0 THEN 1 ELSE 0 END) AS negative_price_rows
FROM group18_reviews.dim_product;

-- Review-to-product join coverage. Expected result: 66,004 matched reviews,
-- zero unmatched reviews, and 100.00 percent coverage.
SELECT
  COUNT(*) AS review_rows,
  SUM(CASE WHEN d.parent_asin IS NOT NULL THEN 1 ELSE 0 END) AS matched_review_rows,
  SUM(CASE WHEN d.parent_asin IS NULL THEN 1 ELSE 0 END) AS unmatched_review_rows,
  ROUND(
    100.0 * SUM(CASE WHEN d.parent_asin IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*),
    2
  ) AS metadata_join_coverage_pct
FROM group18_reviews.fact_review AS f
LEFT JOIN group18_reviews.dim_product AS d
  ON f.parent_asin = d.parent_asin
WHERE f.category = 'Electronics';
