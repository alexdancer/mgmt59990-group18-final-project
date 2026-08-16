-- Rating distribution for dashboard and class-balance review.
SELECT
  rating,
  COUNT(*) AS review_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_reviews,
  ROUND(AVG(text_length), 1) AS avg_text_length,
  ROUND(100.0 * AVG(CASE WHEN verified_purchase THEN 1.0 ELSE 0.0 END), 2) AS pct_verified
FROM group18_reviews.fact_review
WHERE category = 'Electronics'
GROUP BY rating
ORDER BY rating;

-- Monthly rating and low-rating trend. Low rating is the approved binary
-- target: one or two stars.
SELECT
  review_month,
  COUNT(*) AS review_count,
  ROUND(AVG(rating), 2) AS average_rating,
  SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS low_rating_count,
  ROUND(100.0 * AVG(CASE WHEN rating <= 2 THEN 1.0 ELSE 0.0 END), 2) AS low_rating_pct,
  ROUND(100.0 * AVG(CASE WHEN verified_purchase THEN 1.0 ELSE 0.0 END), 2) AS verified_purchase_pct
FROM group18_reviews.fact_review
WHERE category = 'Electronics'
GROUP BY review_month
ORDER BY review_month;

-- Product-level dashboard summary. Requiring at least five sampled reviews
-- keeps the ranking from being dominated by one-off products.
SELECT
  f.parent_asin,
  d.product_title,
  d.main_category,
  d.price,
  COUNT(*) AS review_count,
  ROUND(AVG(f.rating), 2) AS sampled_average_rating,
  ROUND(100.0 * AVG(CASE WHEN f.rating <= 2 THEN 1.0 ELSE 0.0 END), 2) AS low_rating_pct,
  ROUND(AVG(f.helpful_vote), 2) AS average_helpful_votes
FROM group18_reviews.fact_review AS f
JOIN group18_reviews.dim_product AS d
  ON f.parent_asin = d.parent_asin
WHERE f.category = 'Electronics'
GROUP BY f.parent_asin, d.product_title, d.main_category, d.price
HAVING COUNT(*) >= 5
ORDER BY low_rating_pct DESC, review_count DESC;

-- Binary-model training extract. The deterministic user bucket keeps every
-- review from one user in the same train/validation/test split.
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
WHERE category = 'Electronics';
