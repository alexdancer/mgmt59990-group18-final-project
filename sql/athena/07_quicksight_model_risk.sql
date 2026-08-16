-- Paste this query into the QuickSight Athena custom-SQL editor.
-- It produces one dashboard row per product from the held-out test split.
SELECT
  p.parent_asin,
  d.product_title,
  d.main_category,
  d.store,
  d.price,
  d.price_status,
  COUNT(*) AS test_review_count,
  SUM(p.actual_low_rating) AS actual_low_rating_count,
  SUM(p.predicted_low_rating) AS predicted_low_rating_count,
  ROUND(100.0 * AVG(CAST(p.actual_low_rating AS double)), 2) AS actual_low_rating_pct,
  ROUND(100.0 * AVG(CAST(p.predicted_low_rating AS double)), 2) AS predicted_low_rating_pct,
  ROUND(AVG(p.probability_low_rating), 4) AS average_low_rating_probability
FROM group18_reviews.model_predictions AS p
JOIN group18_reviews.dim_product AS d
  ON p.parent_asin = d.parent_asin
GROUP BY
  p.parent_asin,
  d.product_title,
  d.main_category,
  d.store,
  d.price,
  d.price_status
HAVING COUNT(*) >= 3
ORDER BY average_low_rating_probability DESC, test_review_count DESC;
