import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AwsContractTests(unittest.TestCase):
    def test_glue_job_is_valid_python_and_ignores_json_manifests(self):
        source = (PROJECT_ROOT / "aws" / "glue_etl.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('"SILVER_ROOT_PATH"', source)
        self.assertEqual(source.count('option("pathGlobFilter", "*.jsonl")'), 2)

    def test_glue_job_writes_fact_and_dimension_tables(self):
        source = (PROJECT_ROOT / "aws" / "glue_etl.py").read_text(encoding="utf-8")
        self.assertIn('f"{silver_root}/fact_review/"', source)
        self.assertIn('f"{silver_root}/dim_product/"', source)
        self.assertIn('.partitionBy("category", "review_year")', source)
        self.assertIn("Metadata coverage validation failed", source)

    def test_athena_schema_matches_glue_outputs(self):
        ddl = (PROJECT_ROOT / "sql" / "athena" / "01_external_tables.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("group18_reviews.fact_review", ddl)
        self.assertIn("group18_reviews.dim_product", ddl)
        self.assertIn("silver/fact_review/", ddl)
        self.assertIn("silver/dim_product/", ddl)
        self.assertIn("price_status string", ddl)

    def test_analysis_uses_binary_target_and_user_hash(self):
        sql = (PROJECT_ROOT / "sql" / "athena" / "03_analysis.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("CASE WHEN rating <= 2 THEN 1 ELSE 0 END AS low_rating", sql)
        self.assertIn("crc32(to_utf8(user_id))", sql)

    def test_gold_ctas_builds_three_parquet_products(self):
        sql = (PROJECT_ROOT / "sql" / "athena" / "04_gold_tables.sql").read_text(
            encoding="utf-8"
        )
        for table_name, prefix in (
            ("agg_rating_monthly", "gold/agg_rating_monthly/"),
            ("agg_product_monthly", "gold/agg_product_monthly/"),
            ("model_training", "gold/model_training/"),
        ):
            self.assertIn(f"CREATE TABLE group18_reviews.{table_name}", sql)
            self.assertIn(prefix, sql)
        self.assertEqual(sql.count("format = 'PARQUET'"), 3)
        self.assertEqual(sql.count("write_compression = 'SNAPPY'"), 3)
        self.assertIn("partitioned_by = ARRAY['split_name']", sql)

    def test_gold_quality_checks_prevent_user_leakage(self):
        sql = (
            PROJECT_ROOT / "sql" / "athena" / "05_gold_quality_checks.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("users_in_multiple_splits", sql)
        self.assertIn("HAVING COUNT(DISTINCT split_name) > 1", sql)
        self.assertIn("distinct_training_review_ids", sql)

    def test_predictions_are_cataloged_for_athena(self):
        sql = (
            PROJECT_ROOT / "sql" / "athena" / "06_model_predictions.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("CREATE EXTERNAL TABLE group18_reviews.model_predictions", sql)
        self.assertIn("actual_low_rating", sql)
        self.assertIn("predicted_low_rating", sql)
        self.assertIn("probability_low_rating", sql)
        self.assertIn("confusion_total", sql)
        self.assertNotIn("DROP TABLE", sql)

    def test_model_risk_sql_is_dashboard_ready(self):
        sql = (
            PROJECT_ROOT / "sql" / "athena" / "07_quicksight_model_risk.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("group18_reviews.model_predictions", sql)
        self.assertIn("group18_reviews.dim_product", sql)
        self.assertIn("average_low_rating_probability", sql)
        self.assertIn("predicted_low_rating_pct", sql)
        self.assertIn("HAVING COUNT(*) >= 3", sql)
        self.assertNotIn("LIMIT", sql.upper())


if __name__ == "__main__":
    unittest.main()
