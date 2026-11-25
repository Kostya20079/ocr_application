# Optical Character Recognition (OCR) Application

Desktop application for optical character recognition (OCR) with a graphical user interface (GUI), support for bundled sample materials or custom user images, and instant clipboard copying of recognized text. The end target of this project is a standalone `.exe` executable (packaged using PyInstaller).

---

## Character Recognition Methods

The application includes / plans 3 independent OCR engines:
1. **Template Matching** – A classical approach that compares cropped character segments against a set of reference character templates (`data/templates/`).
2. **HOG + k-NN** – Feature extraction using Histogram of Oriented Gradients (HOG) combined with a k-Nearest Neighbors (k-NN) classifier (trained via `scripts/train_hog_knn.py` and cached at `data/models/hog_knn_model.pkl`).
3. **Custom TensorFlow Model + OpenCV** – A custom deep convolutional neural network (CNN) trained in TensorFlow/Keras integrated with OpenCV for document layout analysis, contour detection, and character/line segmentation.

---

## Project Structure

```text
ocr_application/
├── data/
│   ├── samples/                    # Sample images for instant testing
│   │   ├── text_sample.png         # Light-background text sample
│   │   └── text_sample1.png        # Dark-background text sample (terminal screenshot)
│   ├── templates/                  # Character templates for Template Matching engine (git-ignored)
│   └── models/                     # Trained model weights (k-NN, TensorFlow/Keras)
├── locales/                        # Internationalization / Localization files
│   ├── pl/messages.json            # Polish translations (Default)
│   └── en/messages.json            # English translations
├── src/
│   ├── app/
│   │   ├── config.py               # App configuration, constants, resource paths
│   │   └── gui/
│   │       └── main_window.py      # Main GUI window (image preview, engine selector, language switcher, text copier)
│   ├── core/
│   │   ├── base.py                 # BaseOCREngine abstract base class (unified engine interface)
│   │   └── preprocessor.py         # ImagePreprocessor: grayscale, Otsu binarization, paragraph/line/character segmentation, normalization
│   ├── engines/
│   │   ├── template_matching.py    # Engine 1: Template Matching
│   │   ├── hog_knn.py              # Engine 2: HOG + k-NN
│   │   └── tf_model.py             # Engine 3: Custom TensorFlow CNN
│   └── utils/
│       ├── helpers.py              # Resource path resolution (compatible with PyInstaller sys._MEIPASS)
│       └── language.py             # Language support manager (PL & EN)
├── tests/
│   ├── test_skeleton.py            # Skeleton / smoke tests
│   ├── test_image_preprocessor.py  # Unit tests for ImagePreprocessor pipeline
│   ├── test_character_accuracy.py  # Per-character accuracy benchmark & tests
│   └── visualize_methods.py        # Visual inspection of segmentation pipeline (paragraphs, lines, characters, normalization)
├── scripts/
│   ├── build_exe.py                # Automated build script to compile to a standalone .exe
│   ├── generate_templates.py       # Generates normalized 32×32 character templates for OCR engines
│   └── train_hog_knn.py            # MLOps pipeline script to train and serialize the HOG + k-NN model
├── app.py                          # Main application entry point
├── OCRApplication.spec             # PyInstaller build specification
├── requirements.txt                # Project dependencies
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Virtual Environment

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows
```cmd
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```

### Running Tests
```bash
python -m unittest discover tests
```

### Character Recognition Accuracy Test & Benchmark
To evaluate per-character accuracy percentages, confusion matrices, and comparison tables for all OCR algorithms:
```bash
# Run unit tests with console accuracy report:
python -m unittest tests/test_character_accuracy.py

# Or run the benchmark tool directly with formatted console output:
python tests/test_character_accuracy.py

# Optional: customize samples per character or evaluate all dataset templates:
python tests/test_character_accuracy.py --samples-per-char 5
python tests/test_character_accuracy.py --all-samples
```

### Visualizing Preprocessing & Segmentation Pipeline
To visually inspect binarization, paragraph grouping, line isolation, character segmentation, and glyph normalization:
```bash
python tests/visualize_methods.py
```

### Generating Character Templates Dataset
To generate normalized character templates for the Template Matching and HOG + k-NN engines:
```bash
# Generate standard Latin letters (A-Z, a-z):
python scripts/generate_templates.py --letters-only --size 32

# Or generate all standard letters, digits, and punctuation:
python scripts/generate_templates.py --size 32
```
*Note: Generated image files in `data/templates/` are excluded from git via `.gitignore` to keep repository size lean. Run this script whenever you need to produce or refresh templates.*

### Training HOG + k-NN Model
The HOG + k-NN engine uses pre-trained weights serialized to `data/models/hog_knn_model.pkl`. To train or retrain the classifier using the template dataset:

```bash
# Train on data/templates and save to data/models/hog_knn_model.pkl:
python scripts/train_hog_knn.py

# Optional: customize template directory, destination path, or k neighbors:
python scripts/train_hog_knn.py --templates-dir data/templates --output-model data/models/hog_knn_model.pkl -k 1

# Train with verbose logging:
python scripts/train_hog_knn.py -v
```

The training pipeline automatically:
1. Validates the templates directory and metadata index (`data/templates/index.json`).
2. Extracts 324-dimensional HOG feature vectors for each template image.
3. Fits the k-NN classifier (`k=1`, Euclidean distance).
4. Executes quality gate verification on sample text (`data/samples/text-sample2.png`).
5. Serializes the trained model artifact to `data/models/hog_knn_model.pkl`.

---

## Building Standalone Executable (.exe)

Before building for the first time, make sure you have activated your virtual environment, installed dependencies, generated templates (`python scripts/generate_templates.py --size 32`), and trained the HOG + k-NN model (`python scripts/train_hog_knn.py`).

You can compile the application into a standalone executable (or binary on Linux/macOS) using the provided build script:

```bash
python scripts/build_exe.py
```

The output executable along with bundled resources (`data/`, `locales/`, and `_internal/`) will be generated inside the `dist/OCRApplication/` directory.

To run the built application:
- **Linux / macOS:**
  ```bash
  ./dist/OCRApplication/OCRApplication
  ```
- **Windows:**
  ```cmd
  dist\OCRApplication\OCRApplication.exe
  ```

> **Note:** Always run the executable from `dist/OCRApplication/`, not `build/OCRApplication/`. The `build/` directory only contains intermediate PyInstaller build files and does not include the `_internal/` shared libraries.
