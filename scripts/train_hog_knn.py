"""Training pipeline script for HOG + k-NN OCR Engine.

Loads normalized character templates from data/templates, extracts Histogram of Oriented
Gradients (HOG) feature descriptors, trains a k-Nearest Neighbors (k-NN) classifier,
validates the model artifact, and serializes the trained weights to data/models/hog_knn_model.pkl.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Ensure project root is in sys.path when executed directly from repository root or scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engines.hog_knn import HogKnnOCR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_hog_knn")


def build_dataset_from_templates(
    templates_dir: Path | str,
    ocr_engine: HogKnnOCR,
) -> tuple[np.ndarray, list[str]]:
    """Loads templates from metadata index and extracts HOG features.

    Args:
        templates_dir: Directory containing template PNGs and index.json.
        ocr_engine: Instance of HogKnnOCR to use for HOG feature extraction.

    Returns:
        Tuple of (feature_matrix, label_list).
    """
    t_dir = Path(templates_dir)
    index_path = t_dir / "index.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Templates index file not found at: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        metadata: list[dict[str, Any]] = json.load(f)

    features: list[np.ndarray] = []
    labels: list[str] = []

    for entry in metadata:
        filename = entry.get("filename")
        char = entry.get("char")
        if not filename or char is None:
            continue

        img_path = t_dir / filename
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        feat = ocr_engine.extract_hog_features(img)
        features.append(feat)
        labels.append(char)

    if not features:
        raise RuntimeError(f"No valid template images found in '{t_dir}'.")

    return np.asarray(features, dtype=np.float32), labels


def train_hog_knn(
    templates_dir: Path | str = "data/templates",
    output_model: Path | str = "data/models/hog_knn_model.pkl",
    n_neighbors: int = 1,
    test_sample: Path | str | None = "data/samples/text-sample1.png",
) -> Path:
    """Trains, validates, and serializes the HOG + k-NN OCR model.

    Args:
        templates_dir: Path to directory containing character templates and index.json.
        output_model: Output path for the serialized (.pkl) model artifact.
        n_neighbors: Number of nearest neighbors for the k-NN classifier.
        test_sample: Optional path to a test image to verify inference post-training.

    Returns:
        Path to the saved model file.

    Raises:
        FileNotFoundError: If templates_dir or index.json does not exist.
        RuntimeError: If training fails or produces an empty dataset.
    """
    templates_path = Path(templates_dir)
    if not templates_path.is_absolute():
        templates_path = PROJECT_ROOT / templates_path

    index_file = templates_path / "index.json"
    if not templates_path.exists() or not index_file.exists():
        raise FileNotFoundError(
            f"Templates directory or index.json not found at '{templates_path}'. "
            "Please generate character templates first using: python scripts/generate_templates.py"
        )

    output_path = Path(output_model)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    logger.info("Initializing HOG + k-NN training pipeline...")
    logger.info("Templates source: %s", templates_path)
    logger.info("Target model artifact: %s", output_path)
    logger.info("Hyperparameters: k-neighbors=%d, metric=euclidean", n_neighbors)

    start_time = time.perf_counter()

    # Create dummy engine for feature extraction
    feature_extractor = HogKnnOCR(model_path=None)

    # Ingest and extract features from template dataset
    X, y = build_dataset_from_templates(templates_path, feature_extractor)

    # Train k-NN classifier
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        classifier.fit(X, y)

    num_samples = len(X)
    classes = classifier.classes_
    num_classes = len(classes)
    feat_dim = X.shape[1]

    train_elapsed = time.perf_counter() - start_time
    logger.info(
        "Training complete in %.2f seconds (Samples: %d, Classes: %d, Feature Dim: %d).",
        train_elapsed,
        num_samples,
        num_classes,
        feat_dim,
    )

    # Save model artifact to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(classifier, f)

    artifact_size_kb = output_path.stat().st_size / 1024.0
    logger.info("Model artifact successfully saved to: %s (%.1f KB)", output_path, artifact_size_kb)

    # Optional quality gate / verification on test sample
    if test_sample:
        sample_path = Path(test_sample)
        if not sample_path.is_absolute():
            sample_path = PROJECT_ROOT / sample_path

        if sample_path.exists():
            test_img = cv2.imread(str(sample_path))
            if test_img is not None:
                val_start = time.perf_counter()
                eval_engine = HogKnnOCR(model_path=output_path)
                recognized_text = eval_engine.recognize(test_img)
                val_elapsed = time.perf_counter() - val_start
                logger.info(
                    "Inference verification on '%s' (%.3f s):\n  Output: %r",
                    sample_path.name,
                    val_elapsed,
                    recognized_text,
                )
        else:
            logger.debug("Verification sample '%s' not found, skipping inference check.", sample_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and serialize HOG + k-NN OCR classifier from character templates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--templates-dir",
        "-t",
        type=str,
        default="data/templates",
        help="Path to directory containing template PNGs and index.json",
    )
    parser.add_argument(
        "--output-model",
        "-o",
        type=str,
        default="data/models/hog_knn_model.pkl",
        help="Destination path for the serialized model artifact (.pkl)",
    )
    parser.add_argument(
        "--n-neighbors",
        "-k",
        type=int,
        default=1,
        help="Number of nearest neighbors for classification",
    )
    parser.add_argument(
        "--test-sample",
        "-s",
        type=str,
        default="data/samples/text-sample1.png",
        help="Optional test image to run post-training inference verification",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable post-training sample inference verification",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose / debug logging output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    test_sample = None if args.no_verify else args.test_sample

    try:
        train_hog_knn(
            templates_dir=args.templates_dir,
            output_model=args.output_model,
            n_neighbors=args.n_neighbors,
            test_sample=test_sample,
        )
        return 0
    except Exception as e:
        logger.error("Training pipeline failed: %s", e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
