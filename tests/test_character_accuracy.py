"""Unit and integration tests for character recognition accuracy across all OCR algorithms.

Evaluates recognition accuracy percentage for each individual character class,
tests algorithm performance across digits, uppercase, lowercase, and symbols,
and displays a console summary report.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from typing import Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.evaluator import (
    AlgorithmAccuracyReport,
    AccuracyComparisonReport,
    CharacterAccuracy,
    CharacterAccuracyEvaluator,
)
from src.engines import TemplateMatchingOCR, HogKnnOCR, TensorFlowOCR
from src.utils.helpers import get_resource_path


class TestCharacterAccuracy(unittest.TestCase):
    """Test suite evaluating character recognition accuracy for all OCR algorithms."""

    evaluator: CharacterAccuracyEvaluator
    tm_engine: TemplateMatchingOCR
    knn_engine: HogKnnOCR
    tf_engine: TensorFlowOCR
    test_dataset: list[tuple[str, any]]
    comparison_report: AccuracyComparisonReport | None = None

    @classmethod
    def setUpClass(cls) -> None:
        """Initializes engines and builds a standardized evaluation dataset."""
        cls.evaluator = CharacterAccuracyEvaluator()

        # Load standard algorithms
        cls.tm_engine = TemplateMatchingOCR()
        cls.knn_engine = HogKnnOCR()
        cls.tf_engine = TensorFlowOCR()

        # Pre-build balanced evaluation dataset (e.g. 2 diverse samples per character for speed during test suite)
        cls.test_dataset = cls.evaluator.build_evaluation_dataset(samples_per_char=2)
        print(f"\n[Test Setup] Loaded evaluation dataset with {len(cls.test_dataset)} character samples.")

    def test_evaluator_dataset_loading(self) -> None:
        """Verifies evaluator loads valid characters and samples from templates."""
        available_chars = self.evaluator.get_available_characters()
        self.assertGreater(len(available_chars), 50)
        self.assertIn("A", available_chars)
        self.assertIn("z", available_chars)
        self.assertIn("0", available_chars)

        sample_dataset = self.evaluator.build_evaluation_dataset(
            samples_per_char=1,
            characters=["A", "B", "9"],
        )
        self.assertEqual(len(sample_dataset), 3)
        for char, glyph in sample_dataset:
            self.assertEqual(glyph.shape, (32, 32))

    def test_accuracy_template_matching(self) -> None:
        """Evaluates Template Matching OCR character recognition accuracy and percentages."""
        if not self.tm_engine.templates:
            self.skipTest("Template matching templates not found.")

        report = self.evaluator.evaluate_engine(
            self.tm_engine,
            dataset=self.test_dataset,
        )

        # Print detailed per-character table to console
        print("\n" + report.format_console_table() + "\n")

        self.assertEqual(report.algorithm_name, "Template Matching")
        self.assertGreater(report.total_samples, 0)
        self.assertGreater(report.overall_accuracy_percent, 75.0)

        # Verify per-character accuracy calculation
        for char, metric in report.character_metrics.items():
            self.assertIsInstance(metric, CharacterAccuracy)
            self.assertGreaterEqual(metric.accuracy_percent, 0.0)
            self.assertLessEqual(metric.accuracy_percent, 100.0)
            self.assertEqual(metric.total, metric.correct + metric.incorrect)

    def test_accuracy_hog_knn(self) -> None:
        """Evaluates HOG + k-NN OCR character recognition accuracy and percentages."""
        if not self.knn_engine.is_ready:
            self.skipTest("HOG + k-NN model artifact not trained or loaded.")

        report = self.evaluator.evaluate_engine(
            self.knn_engine,
            dataset=self.test_dataset,
        )

        # Print detailed per-character table to console
        print("\n" + report.format_console_table() + "\n")

        self.assertEqual(report.algorithm_name, "HOG + k-NN")
        self.assertGreater(report.total_samples, 0)
        self.assertGreater(report.overall_accuracy_percent, 85.0)

        # Verify subgroup accuracy
        digits_acc = report.get_group_accuracy_percent("digits")
        upper_acc = report.get_group_accuracy_percent("uppercase")
        lower_acc = report.get_group_accuracy_percent("lowercase")

        self.assertGreater(digits_acc, 80.0)
        self.assertGreater(upper_acc, 80.0)
        self.assertGreater(lower_acc, 80.0)

    def test_accuracy_comparison_all_algorithms(self) -> None:
        """Compares recognition accuracy and character percentages across all active algorithms."""
        active_engines = []
        if self.tm_engine.templates:
            active_engines.append(self.tm_engine)
        if self.knn_engine.is_ready:
            active_engines.append(self.knn_engine)

        if not active_engines:
            self.skipTest("No active OCR engines available.")

        comparison = self.evaluator.evaluate_all(
            engines=active_engines,
            samples_per_char=2,
        )

        print("\n" + comparison.format_comparison_table() + "\n")

        self.assertEqual(len(comparison.reports), len(active_engines))
        for name, rep in comparison.reports.items():
            self.assertGreater(rep.overall_accuracy_percent, 70.0)

    def test_uninitialized_engine_handling(self) -> None:
        """Verifies evaluator gracefully handles engines under development or not ready."""
        # TensorFlow model is currently in development / not ready
        report = self.evaluator.evaluate_engine(self.tf_engine, dataset=[("A", self.test_dataset[0][1])])
        self.assertIsNotNone(report)


def run_accuracy_benchmark(
    samples_per_char: int | None = 3,
    characters: Sequence[str] | None = None,
    engine_filter: str | None = None,
) -> int:
    """Runs character accuracy evaluation CLI and prints comprehensive console report."""
    print("=" * 80)
    print("           OCR CHARACTER ACCURACY BENCHMARK & TEST SUITE")
    print("=" * 80)

    evaluator = CharacterAccuracyEvaluator()
    engines = []

    if engine_filter in (None, "all", "tm", "template_matching"):
        tm = TemplateMatchingOCR()
        if tm.templates:
            engines.append(tm)

    if engine_filter in (None, "all", "knn", "hog_knn"):
        knn = HogKnnOCR()
        if knn.is_ready:
            engines.append(knn)

    if not engines:
        print("[ERROR] No trained/ready OCR engines found to evaluate.")
        return 1

    print(f"Evaluating {len(engines)} OCR algorithm(s)...")
    dataset = evaluator.build_evaluation_dataset(
        samples_per_char=samples_per_char,
        characters=characters,
    )
    print(f"Total test characters: {len(set(c for c, _ in dataset))}")
    print(f"Total test samples   : {len(dataset)}")
    print("-" * 80)

    comparison = AccuracyComparisonReport()
    for eng in engines:
        print(f"Running character accuracy evaluation for '{eng.name}'...")
        rep = evaluator.evaluate_engine(eng, dataset=dataset)
        comparison.reports[eng.name] = rep
        print(rep.format_console_table())
        print()

    if len(engines) > 1:
        print(comparison.format_comparison_table())

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test character recognition accuracy with percent per algorithm.")
    parser.add_argument(
        "--samples-per-char",
        type=int,
        default=3,
        help="Number of samples tested per character (default: 3, 0 or negative for all samples).",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Evaluate on all available template samples in the dataset.",
    )
    parser.add_argument(
        "--engine",
        choices=["all", "template_matching", "hog_knn"],
        default="all",
        help="Specific OCR engine to evaluate (default: all).",
    )
    parser.add_argument(
        "--unittest",
        action="store_true",
        help="Run standard unittest test runner instead of benchmark CLI.",
    )

    args, unknown = parser.parse_known_args()

    if args.unittest or "--verbose" in sys.argv or "-v" in sys.argv:
        unittest.main(argv=[sys.argv[0]] + unknown)
    else:
        samples_count = None if args.all_samples or args.samples_per_char <= 0 else args.samples_per_char
        sys.exit(run_accuracy_benchmark(samples_per_char=samples_count, engine_filter=args.engine))
