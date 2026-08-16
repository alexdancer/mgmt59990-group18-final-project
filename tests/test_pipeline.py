import unittest

from amazon_reviews_pipeline.pipeline import normalize_text, transform
from amazon_reviews_pipeline.random_sample import choose_offsets, extract_complete_lines


class PipelineTests(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Great\n\tproduct  "), "Great product")

    def test_transform_filters_invalid_and_deduplicates(self):
        valid = {
            "rating": 5.0,
            "title": "Great",
            "text": "Works well",
            "asin": "VARIANT1",
            "parent_asin": "PARENT1",
            "user_id": "USER1",
            "timestamp": 1_700_000_000_000,
            "helpful_vote": 2,
            "verified_purchase": True,
        }
        invalid = {**valid, "rating": 9, "user_id": "USER2"}
        frame, quality = transform([valid, valid.copy(), invalid])
        self.assertEqual(len(frame), 1)
        self.assertEqual(quality["duplicate_rows_removed"], 1)
        self.assertEqual(quality["invalid_rating_rows"], 1)
        self.assertEqual(frame.iloc[0]["review_text"], "Great. Works well")

    def test_random_offsets_are_reproducible_and_distributed(self):
        first = choose_offsets(total_size=10_000, range_bytes=100, ranges=4, seed=18)
        second = choose_offsets(total_size=10_000, range_bytes=100, ranges=4, seed=18)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        for index, offset in enumerate(first):
            self.assertGreaterEqual(offset, index * 2_500)
            self.assertLessEqual(offset, (index + 1) * 2_500 - 100)

    def test_extract_complete_lines_removes_boundary_fragments(self):
        payload = b'partial record\n{"a":1}\n{"a":2}\npartial'
        self.assertEqual(extract_complete_lines(payload), [b'{"a":1}', b'{"a":2}'])


if __name__ == "__main__":
    unittest.main()
