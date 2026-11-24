"""Character recognition accuracy evaluation module for OCR engines.

Computes character-level accuracy metrics, percentage breakdowns, confusion statistics,
and comparative reports across multiple OCR algorithms.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.core.base import BaseOCREngine
from src.utils.helpers import get_resource_path

logger = logging.getLogger(__name__)


@dataclass
class CharacterAccuracy:
    """Accuracy metrics for a single character class."""

    char: str
    total: int = 0
    correct: int = 0
    confusions: dict[str, int] = field(default_factory=dict)

    @property
    def incorrect(self) -> int:
        return max(0, self.total - self.correct)

    @property
    def accuracy_percent(self) -> float:
        """Returns accuracy as a percentage (0.0 to 100.0)."""
        if self.total == 0:
            return 0.0
        return (self.correct / self.total) * 100.0

    def top_confusions(self, top_n: int = 3) -> str:
        """Returns a formatted string of the most common misrecognitions."""
        if not self.confusions:
            return "-"
        sorted_conf = sorted(self.confusions.items(), key=lambda item: item[1], reverse=True)
        items = [f"{repr(k)}: {v}" for k, v in sorted_conf[:top_n]]
        return ", ".join(items)


@dataclass
class AlgorithmAccuracyReport:
    """Complete evaluation report for a single OCR algorithm."""

    algorithm_name: str
    character_metrics: dict[str, CharacterAccuracy] = field(default_factory=dict)
    total_samples: int = 0
    total_correct: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_incorrect(self) -> int:
        return max(0, self.total_samples - self.total_correct)

    @property
    def overall_accuracy_percent(self) -> float:
        """Returns overall accuracy as a percentage (0.0 to 100.0)."""
        if self.total_samples == 0:
            return 0.0
        return (self.total_correct / self.total_samples) * 100.0

    def get_character_accuracy(self, char: str) -> float:
        """Returns accuracy percent for a specific character."""
        if char in self.character_metrics:
            return self.character_metrics[char].accuracy_percent
        return 0.0

    def get_group_accuracy_percent(self, group_name: str) -> float:
        """Calculates accuracy for character subgroups: 'digits', 'uppercase', 'lowercase', 'symbols'."""
        group_chars: set[str] = set()
        if group_name == "digits":
            group_chars = set("0123456789")
        elif group_name == "uppercase":
            group_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        elif group_name == "lowercase":
            group_chars = set("abcdefghijklmnopqrstuvwxyz")
        elif group_name == "symbols":
            digits_and_letters = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            group_chars = {c for c in self.character_metrics if c not in digits_and_letters}

        filtered = [m for c, m in self.character_metrics.items() if c in group_chars]
        total = sum(m.total for m in filtered)
        correct = sum(m.correct for m in filtered)
        if total == 0:
            return 0.0
        return (correct / total) * 100.0

    def format_console_table(self) -> str:
        """Formats per-character accuracy table and category summary for console output."""
        lines: list[str] = []
        bar = "=" * 90
        sub_bar = "-" * 90

        lines.append(bar)
        lines.append(f" ACCURACY REPORT: {self.algorithm_name.upper()}")
        lines.append(bar)
        lines.append(
            f"{'Char':<6} {'Expected':<10} {'Tested':>8} {'Correct':>9} {'Errors':>8} {'Accuracy':>12}   {'Top Confusions'}"
        )
        lines.append(sub_bar)

        sorted_chars = sorted(
            self.character_metrics.keys(),
            key=lambda c: (
                0 if c.isdigit() else 1 if c.isupper() else 2 if c.islower() else 3,
                c,
            ),
        )

        for char in sorted_chars:
            m = self.character_metrics[char]
            char_repr = repr(char)
            conf_str = m.top_confusions(top_n=2)
            lines.append(
                f"{char:<6} {char_repr:<10} {m.total:>8} {m.correct:>9} {m.incorrect:>8} {m.accuracy_percent:>11.2f}%   {conf_str}"
            )

        lines.append(sub_bar)
        lines.append("SUBGROUP ACCURACY BREAKDOWN:")
        for group in ["digits", "uppercase", "lowercase", "symbols"]:
            pct = self.get_group_accuracy_percent(group)
            lines.append(f"  • {group.capitalize():<12}: {pct:>6.2f}%")

        lines.append(sub_bar)
        lines.append(
            f"OVERALL ACCURACY: {self.overall_accuracy_percent:.2f}% "
            f"({self.total_correct}/{self.total_samples} correct, {self.total_incorrect} errors)"
        )
        lines.append(f"Execution time  : {self.elapsed_seconds:.2f} seconds")
        lines.append(bar)
        return "\n".join(lines)


@dataclass
class AccuracyComparisonReport:
    """Comparison report across multiple OCR algorithms."""

    reports: dict[str, AlgorithmAccuracyReport] = field(default_factory=dict)

    def format_comparison_table(self) -> str:
        """Formats comparative summary table across all evaluated algorithms."""
        lines: list[str] = []
        bar = "=" * 80
        sub_bar = "-" * 80

        lines.append(bar)
        lines.append("              OCR ALGORITHMS ACCURACY COMPARISON SUMMARY")
        lines.append(bar)
        lines.append(
            f"{'Algorithm':<30} {'Total':>8} {'Correct':>9} {'Accuracy (%)':>14} {'Time (s)':>10}"
        )
        lines.append(sub_bar)

        for name, r in self.reports.items():
            lines.append(
                f"{name:<30} {r.total_samples:>8} {r.total_correct:>9} {r.overall_accuracy_percent:>13.2f}% {r.elapsed_seconds:>9.2f}s"
            )

        lines.append(sub_bar)

        # Highlight characters where accuracy differs significantly
        all_chars = sorted(
            set(
                c
                for r in self.reports.values()
                for c in r.character_metrics.keys()
            )
        )
        if len(self.reports) > 1 and all_chars:
            lines.append("\nPER-CHARACTER ACCURACY COMPARISON (Selected):")
            header = f"{'Char':<6}" + "".join(f"{name[:16]:>18}" for name in self.reports.keys())
            lines.append(header)
            lines.append("-" * len(header))
            for c in all_chars:
                rates = [r.get_character_accuracy(c) for r in self.reports.values()]
                diff = max(rates) - min(rates)
                # Show chars with difference or low performance
                if diff > 15.0 or min(rates) < 80.0:
                    row = f"{repr(c):<6}" + "".join(f"{rate:>17.1f}%" for rate in rates)
                    lines.append(row)

        lines.append(bar)
        return "\n".join(lines)


class CharacterAccuracyEvaluator:
    """Evaluates character recognition accuracy for OCR engines using template dataset or custom samples."""

    def __init__(self, templates_dir: Path | str | None = None) -> None:
        if templates_dir is None:
            self.templates_dir = get_resource_path("data/templates")
        else:
            self.templates_dir = Path(templates_dir)

        self._metadata_cache: list[dict[str, Any]] | None = None
        self._samples_by_char: dict[str, list[Path]] | None = None

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Loads and caches templates index.json metadata."""
        if self._metadata_cache is not None:
            return self._metadata_cache

        index_path = self.templates_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Template index file not found at: {index_path}")

        with index_path.open("r", encoding="utf-8") as f:
            metadata: list[dict[str, Any]] = json.load(f)

        self._metadata_cache = metadata
        return metadata

    def get_available_characters(self) -> list[str]:
        """Returns sorted list of all unique characters in the dataset."""
        metadata = self._load_metadata()
        chars = sorted(list({entry["char"] for entry in metadata if "char" in entry}))
        return chars

    def build_evaluation_dataset(
        self,
        samples_per_char: int | None = 3,
        characters: Sequence[str] | None = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Builds a balanced evaluation dataset of (expected_char, glyph_image) pairs.

        Args:
            samples_per_char: Maximum number of sample images per character.
                If None, uses all available templates for each character.
            characters: Optional sequence of characters to evaluate.
                If None, evaluates all available characters.

        Returns:
            List of (expected_char, glyph_image_array) tuples.
        """
        metadata = self._load_metadata()
        target_char_set = set(characters) if characters is not None else None

        # Group template filenames by character
        char_entries: dict[str, list[str]] = defaultdict(list)
        for entry in metadata:
            char = entry.get("char")
            filename = entry.get("filename")
            if char is None or not filename:
                continue
            if target_char_set is not None and char not in target_char_set:
                continue
            char_entries[char].append(filename)

        dataset: list[tuple[str, np.ndarray]] = []

        for char, filenames in sorted(char_entries.items()):
            selected_files = filenames
            if samples_per_char is not None and samples_per_char > 0:
                selected_files = filenames[:samples_per_char]

            for fname in selected_files:
                img_path = self.templates_dir / fname
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                dataset.append((char, img))

        return dataset

    def evaluate_engine(
        self,
        engine: BaseOCREngine,
        dataset: list[tuple[str, np.ndarray]] | None = None,
        samples_per_char: int | None = 3,
        characters: Sequence[str] | None = None,
    ) -> AlgorithmAccuracyReport:
        """Evaluates character recognition accuracy for an OCR engine.

        Args:
            engine: BaseOCREngine instance (e.g. TemplateMatchingOCR or HogKnnOCR).
            dataset: Pre-built dataset of (expected_char, glyph_image) pairs.
                If None, automatically builds one using samples_per_char and characters.
            samples_per_char: Number of samples per character if dataset is built dynamically.
            characters: Target characters to evaluate if dataset is built dynamically.

        Returns:
            AlgorithmAccuracyReport containing detailed per-character accuracy and percentages.
        """
        if not getattr(engine, "is_ready", True):
            logger.warning("Engine %s is not ready/initialized.", engine.name)
            return AlgorithmAccuracyReport(
                algorithm_name=engine.name,
                character_metrics={},
                total_samples=0,
                total_correct=0,
                elapsed_seconds=0.0,
            )

        if dataset is None:
            dataset = self.build_evaluation_dataset(
                samples_per_char=samples_per_char,
                characters=characters,
            )

        metrics_by_char: dict[str, CharacterAccuracy] = defaultdict(
            lambda: CharacterAccuracy(char="")
        )

        # Initialize all target characters
        all_chars = sorted(list({expected for expected, _ in dataset}))
        for char in all_chars:
            metrics_by_char[char] = CharacterAccuracy(char=char)

        total_correct = 0
        total_samples = len(dataset)

        start_time = time.perf_counter()

        for expected_char, glyph_image in dataset:
            try:
                predicted_char = engine.predict_glyph(glyph_image)
            except Exception as e:
                logger.debug("Prediction error for char %r: %s", expected_char, e)
                predicted_char = "?"

            char_metric = metrics_by_char[expected_char]
            char_metric.total += 1

            if predicted_char == expected_char:
                char_metric.correct += 1
                total_correct += 1
            else:
                char_metric.confusions[predicted_char] = (
                    char_metric.confusions.get(predicted_char, 0) + 1
                )

        elapsed = time.perf_counter() - start_time

        return AlgorithmAccuracyReport(
            algorithm_name=engine.name,
            character_metrics=dict(metrics_by_char),
            total_samples=total_samples,
            total_correct=total_correct,
            elapsed_seconds=elapsed,
        )

    def evaluate_all(
        self,
        engines: Sequence[BaseOCREngine],
        samples_per_char: int | None = 3,
        characters: Sequence[str] | None = None,
    ) -> AccuracyComparisonReport:
        """Evaluates multiple OCR engines on the same dataset and compiles a comparison report.

        Args:
            engines: Sequence of BaseOCREngine instances.
            samples_per_char: Number of samples per character.
            characters: Target characters to evaluate.

        Returns:
            AccuracyComparisonReport containing reports for all evaluated algorithms.
        """
        dataset = self.build_evaluation_dataset(
            samples_per_char=samples_per_char,
            characters=characters,
        )

        comparison = AccuracyComparisonReport()
        for engine in engines:
            report = self.evaluate_engine(
                engine=engine,
                dataset=dataset,
            )
            comparison.reports[engine.name] = report

        return comparison
