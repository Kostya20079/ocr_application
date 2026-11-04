# Optical Character Recognition (OCR) Application

Desktop application for optical character recognition (OCR) with a graphical user interface (GUI), support for bundled sample materials or custom user images, and instant clipboard copying of recognized text. The end target of this project is a standalone `.exe` executable (packaged using PyInstaller).

---

## Character Recognition Methods

The application includes / plans 3 independent OCR engines:
1. **Template Matching** – A classical approach that compares cropped character segments against a set of reference character templates (`data/templates/`).
2. **HOG + k-NN** – Feature extraction using Histogram of Oriented Gradients (HOG) combined with a k-Nearest Neighbors (k-NN) classifier.
3. **Custom TensorFlow Model + OpenCV** – A custom deep convolutional neural network (CNN) trained in TensorFlow/Keras integrated with OpenCV for document layout analysis, contour detection, and character/line segmentation.

---

## Project Structure

```text
ocr_application/
├── data/
│   ├── samples/               # Sample images for instant testing
│   ├── templates/             # Character templates for the Template Matching engine
│   └── models/                # Trained model weights (k-NN, TensorFlow/Keras)
├── locales/                   # Internationalization / Localization files
│   ├── pl/messages.json       # Polish translations (Default)
│   └── en/messages.json       # English translations
├── src/
│   ├── app/
│   │   ├── config.py          # App configuration, constants, resource paths
│   │   └── gui/
│   │       ├── main_window.py # Main GUI window (image preview, engine selector, language switcher, text copier)
│   │       └── components/    # Reusable UI components
│   ├── core/
│   │   ├── base.py            # BaseOCREngine abstract base class (unified engine interface)
│   │   └── preprocessor.py    # Image preprocessing (grayscale, binarization, denoising, segmentation)
│   ├── engines/
│   │   ├── template_matching.py # Engine 1: Template Matching
│   │   ├── hog_knn.py           # Engine 2: HOG + k-NN
│   │   └── tf_model.py          # Engine 3: Custom TensorFlow CNN
│   └── utils/
│       ├── helpers.py         # Resource path resolution (compatible with PyInstaller sys._MEIPASS)
│       └── language.py        # Language support manager (PL & EN)
├── tests/                     # Unit tests for preprocessing, engines, and utilities
│   └── test_skeleton.py
├── scripts/
│   └── build_exe.py           # Automated build script to compile to a standalone .exe
├── app.py                     # Main application entry point
├── requirements.txt           # Project dependencies
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

---

## Building Standalone Executable (.exe)

You can compile the application into a standalone executable (or binary on Linux/macOS) using the provided build script:

```bash
python scripts/build_exe.py
```
The output executable along with bundled resources (`data/samples`, models, templates) will be generated inside the `dist/OCRApplication` directory.
