import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

from src.app.config import SAMPLES_DIR, DEFAULT_LOCALE
from src.engines import TemplateMatchingOCR, HogKnnOCR, TensorFlowOCR
from src.utils.language import language

class OCRAppGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        language.set_locale(DEFAULT_LOCALE)
        self.root.title(language("app_title"))
        self.root.geometry("880x620")
        self.root.minsize(720, 500)

        # Initialize OCR engines
        self.engines = {
            "template_matching": TemplateMatchingOCR(),
            "hog_knn": HogKnnOCR(),
            "tf_model": TensorFlowOCR(),
        }

        self.selected_image_path: Path | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self._setup_ui()
        self._load_sample_files()

    def _setup_ui(self) -> None:
        # Top Language & Settings bar
        top_bar = ttk.Frame(self.root, padding=(10, 5, 10, 0))
        top_bar.pack(fill=tk.X)

        self.lbl_language = ttk.Label(top_bar, text=language("language_label"))
        self.lbl_language.pack(side=tk.LEFT, padx=5)

        self.lang_combo = ttk.Combobox(top_bar, values=["Polski (PL)", "English (EN)"], state="readonly", width=15)
        self.lang_combo.current(0 if language.locale == "pl" else 1)
        self.lang_combo.pack(side=tk.LEFT, padx=5)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_changed)

        # Controls header: image source & method selection
        self.control_frame = ttk.LabelFrame(self.root, text=language("config_frame_title"), padding=10)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Engine selector
        self.lbl_method = ttk.Label(self.control_frame, text=language("ocr_method_label"))
        self.lbl_method.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.engine_keys = ["template_matching", "hog_knn", "tf_model"]
        self.engine_combo = ttk.Combobox(self.control_frame, state="readonly", width=38)
        self._refresh_engine_combo_values()
        self.engine_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Sample selection
        self.lbl_samples = ttk.Label(self.control_frame, text=language("samples_label"))
        self.lbl_samples.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.samples_combo = ttk.Combobox(self.control_frame, state="readonly", width=38)
        self.samples_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.samples_combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

        # Browse custom image button
        self.btn_browse = ttk.Button(self.control_frame, text=language("browse_button"), command=self._browse_custom_image)
        self.btn_browse.grid(row=1, column=2, padx=10, pady=5)

        # Run OCR button
        self.btn_run = ttk.Button(self.control_frame, text=language("run_button"), command=self._run_ocr)
        self.btn_run.grid(row=0, column=2, padx=10, pady=5)

        # Main horizontal split pane for preview and text result
        content_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Image preview frame
        self.img_frame = ttk.LabelFrame(content_pane, text=language("preview_frame_title"), padding=10)
        content_pane.add(self.img_frame, weight=1)

        self.image_label = ttk.Label(self.img_frame, text=language("no_image_label"), anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Text result frame
        self.text_frame = ttk.LabelFrame(content_pane, text=language("result_frame_title"), padding=10)
        content_pane.add(self.text_frame, weight=1)

        self.text_result = tk.Text(self.text_frame, wrap=tk.WORD)
        self.text_result.pack(fill=tk.BOTH, expand=True)

        # Bottom action bar (copy to clipboard & status)
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X)

        self.btn_copy = ttk.Button(bottom_frame, text=language("copy_button"), command=self._copy_to_clipboard)
        self.btn_copy.pack(side=tk.RIGHT)

        self.lbl_status = ttk.Label(bottom_frame, text=language("status_ready"))
        self.lbl_status.pack(side=tk.LEFT)

    def _refresh_engine_combo_values(self) -> None:
        current_idx = self.engine_combo.current()
        if current_idx < 0:
            current_idx = 0
        display_names = [language(f"engines.{k}") for k in self.engine_keys]
        self.engine_combo["values"] = display_names
        self.engine_combo.current(current_idx)

    def _on_language_changed(self, _event=None) -> None:
        chosen = self.lang_combo.get()
        new_locale = "pl" if "PL" in chosen else "en"
        language.set_locale(new_locale)

        # Update all UI texts dynamically
        self.root.title(language("app_title"))
        self.lbl_language.config(text=language("language_label"))
        self.control_frame.config(text=language("config_frame_title"))
        self.lbl_method.config(text=language("ocr_method_label"))
        self.lbl_samples.config(text=language("samples_label"))
        self.btn_browse.config(text=language("browse_button"))
        self.btn_run.config(text=language("run_button"))
        self.img_frame.config(text=language("preview_frame_title"))
        if not self.selected_image_path:
            self.image_label.config(text=language("no_image_label"))
        self.text_frame.config(text=language("result_frame_title"))
        self.btn_copy.config(text=language("copy_button"))
        self.lbl_status.config(text=language("status_ready"))
        self._refresh_engine_combo_values()
        self._load_sample_files()

    def _load_sample_files(self) -> None:
        """Loads available sample image files from the data/samples directory."""
        if SAMPLES_DIR.exists():
            samples = [f.name for f in SAMPLES_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
            if samples:
                self.samples_combo["values"] = samples
                self.samples_combo.current(0)
                self._on_sample_selected()
            else:
                self.samples_combo["values"] = [language("sample_no_files")]
                self.samples_combo.current(0)
        else:
            self.samples_combo["values"] = [language("sample_no_files")]
            self.samples_combo.current(0)

    def _on_sample_selected(self, _event=None) -> None:
        selected_name = self.samples_combo.get()
        sample_path = SAMPLES_DIR / selected_name
        if sample_path.exists() and sample_path.is_file():
            self._display_image(sample_path)

    def _browse_custom_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title=language("browse_button"),
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self._display_image(Path(file_path))

    def _display_image(self, path: Path) -> None:
        self.selected_image_path = path
        try:
            if Image is not None and ImageTk is not None:
                img = Image.open(path)
                # Resize for preview preserving aspect ratio
                img.thumbnail((380, 400))
                self.tk_image = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.tk_image, text="")
            else:
                self.image_label.configure(text=f"Selected: {path.name}")
            self.lbl_status.config(text=language("status_loaded", filename=path.name))
        except Exception as e:
            messagebox.showerror("Error", language("error_load_failed", error=str(e)))

    def _run_ocr(self) -> None:
        if not self.selected_image_path or not self.selected_image_path.exists():
            messagebox.showwarning("Warning", language("warning_no_image"))
            return

        current_idx = self.engine_combo.current()
        engine_key = self.engine_keys[current_idx] if 0 <= current_idx < len(self.engine_keys) else None
        engine = self.engines.get(engine_key)
        if not engine:
            messagebox.showerror("Error", language("error_no_engine"))
            return

        engine_display_name = language(f"engines.{engine_key}")
        self.lbl_status.config(text=language("status_processing", method=engine_display_name))
        self.root.update_idletasks()

        try:
            image_name = self.selected_image_path.name if self.selected_image_path else None
            if Image is not None:
                import numpy as np
                img_np = np.array(Image.open(self.selected_image_path))
                try:
                    result = engine.recognize(img_np, image_name=image_name)
                except TypeError:
                    result = engine.recognize(img_np)
            else:
                try:
                    result = engine.recognize(self.selected_image_path, image_name=image_name)
                except TypeError:
                    result = engine.recognize(self.selected_image_path)
            self.text_result.delete("1.0", tk.END)
            self.text_result.insert(tk.END, result)
            self.lbl_status.config(text=language("status_done"))
        except Exception as e:
            messagebox.showerror("Error", language("error_ocr_failed", error=str(e)))
            self.lbl_status.config(text=language("status_error"))

    def _copy_to_clipboard(self) -> None:
        text = self.text_result.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Clipboard", language("info_empty_clipboard"))
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.lbl_status.config(text=language("status_copied"))

def run() -> None:
    root = tk.Tk()
    _app = OCRAppGUI(root)
    root.mainloop()
