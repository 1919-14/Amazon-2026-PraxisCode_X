"""Small executable checks for the challenge metric and TSV reader."""

import tempfile
import unittest
from pathlib import Path

from audit import audit_file
from metrics import entity_f05, macro_f05


class MetricTests(unittest.TestCase):
    def test_supplied_example(self) -> None:
        self.assertAlmostEqual(entity_f05({"S2-a", "S3-b"}, {"S2-a", "S3-b", "S2-c"}), 5 / 7)

    def test_empty_cases(self) -> None:
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05(set(), {"S2-a"}), 0.0)
        self.assertEqual(entity_f05({"S2-a"}, set()), 0.0)

    def test_perfect_and_disjoint(self) -> None:
        self.assertEqual(entity_f05({"S2-a"}, {"S2-a"}), 1.0)
        self.assertEqual(entity_f05({"S2-a"}, {"S2-b"}), 0.0)

    def test_macro_not_micro(self) -> None:
        self.assertEqual(macro_f05({"S1-a": set(), "S1-b": {"S2-b"}},
                                   {"S1-a": set(), "S1-b": set()}), 0.5)

    def test_reference_coverage(self) -> None:
        with self.assertRaises(ValueError):
            macro_f05({"S1-a": set()}, {})
        with self.assertRaises(ValueError):
            macro_f05({}, {})


class AuditTests(unittest.TestCase):
    def test_unicode_tabs_and_empty_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_source1.tsv"
            path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                            "S1-a\tCaf\u00e9, Paris\t\tFrance\n", encoding="utf-8")
            result = audit_file(path)
            self.assertEqual(result["countries"], {"France": 1})
            self.assertEqual(result["missing_or_blank"], {"business_address": 1})
            self.assertEqual(result["non_ascii_rows"]["business_name"], 1)

    def test_singletons_and_label_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_ground_truth.tsv"
            path.write_text("source1_entity_id\tmatched_entity_ids\n"
                            "S1-a\t\nS1-b\tS2-a,S3-a\n", encoding="utf-8")
            result = audit_file(path)
            self.assertEqual(result["positive_pairs"], 2)
            self.assertEqual(result["all_empty_macro_f05"], 0.5)
            self.assertEqual(audit_file(path, limit=1)["records_scanned"], 1)

    def test_malformed_record_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_source1.tsv"
            path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                            "S1-a\tName\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_file(path)


if __name__ == "__main__":
    unittest.main()