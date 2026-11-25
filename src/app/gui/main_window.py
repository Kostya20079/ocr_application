"""Main Graphical User Interface (GUI) module for the OCR Application.

Provides a clean, intuitive, modern desktop interface with:
- Image source selection (bundled samples or custom files).
- OCR engine selection with descriptive hints.
- Real-time side-by-side view (dynamically scaling image preview and recognized text).
- Text metrics (characters, words, lines), clipboard copying, and file export.
- Asynchronous background execution with live status and progress indicator.
- Dynamic runtime language switching (Polish / English).
"""

import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

from src.app.config import SAMPLES_DIR, DEFAULT_LOCALE, APP_NAME, APP_VERSION
from src.engines import TemplateMatchingOCR, HogKnnOCR, TensorFlowOCR
from src.utils.language import language


def _enable_high_dpi() -> None:
    """Enables high DPI awareness on Windows to prevent blurry rendering."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _setup_theme_styles(style: ttk.Style) -> None:
    """Configures a modern, colorful, and high-contrast theme palette."""
    try:
        style.theme_use("clam")
    except Exception:
        pass

    bg_main = "#f1f5f9"
    card_bg = "#ffffff"
    text_main = "#0f172a"
    text_muted = "#64748b"

    header_bg = "#1e293b"
    header_fg = "#ffffff"
    header_sub = "#94a3b8"

    style.configure(".", background=bg_main, foreground=text_main, font=("Segoe UI", 9))
    style.configure("TFrame", background=bg_main)
    style.configure("Card.TFrame", background=card_bg, relief="solid", borderwidth=1)
    style.configure("TLabelframe", background=card_bg, relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=card_bg, foreground=text_main, font=("Segoe UI", 10, "bold"))

    style.configure("TLabel", background=bg_main, foreground=text_main, font=("Segoe UI", 9))
    style.configure("Card.TLabel", background=card_bg, foreground=text_main, font=("Segoe UI", 9))

    # Header bar
    style.configure("Header.TFrame", background=header_bg, relief="flat")
    style.configure("HeaderInner.TFrame", background=header_bg)
    style.configure("HeaderTitle.TLabel", background=header_bg, foreground=header_fg, font=("Segoe UI", 13, "bold"))
    style.configure("HeaderSubtitle.TLabel", background=header_bg, foreground=header_sub, font=("Segoe UI", 9))
    style.configure("HeaderLang.TLabel", background=header_bg, foreground=header_fg, font=("Segoe UI", 9))

    # Badges & hints
    style.configure("SectionStep.TLabel", background="#eff6ff", foreground="#1d4ed8", font=("Segoe UI", 9, "bold"))
    style.configure("Hint.TLabel", background=card_bg, foreground=text_muted, font=("Segoe UI", 8))
    style.configure("Stats.TLabel", background=card_bg, foreground=text_muted, font=("Segoe UI", 8))
    style.configure("FileBadge.TLabel", background="#f0fdf4", foreground="#15803d", font=("Segoe UI", 8, "bold"))
    style.configure("Placeholder.TLabel", background=card_bg, foreground=text_muted, font=("Segoe UI", 10))

    # Buttons
    style.configure("TButton", font=("Segoe UI", 9), padding=(8, 4))

    # Primary Run Button (Vibrant Blue)
    style.configure(
        "Primary.TButton",
        font=("Segoe UI", 10, "bold"),
        background="#2563eb",
        foreground="#ffffff",
        padding=(14, 6),
        borderwidth=0
    )
    style.map(
        "Primary.TButton",
        background=[("pressed", "#1e40af"), ("active", "#3b82f6"), ("disabled", "#94a3b8")],
        foreground=[("disabled", "#e2e8f0")]
    )

    # Browse Button (Sky Blue)
    style.configure(
        "Browse.TButton",
        font=("Segoe UI", 9, "bold"),
        background="#0ea5e9",
        foreground="#ffffff",
        padding=(8, 4),
        borderwidth=0
    )
    style.map(
        "Browse.TButton",
        background=[("pressed", "#0369a1"), ("active", "#38bdf8")]
    )

    # Copy Button (Emerald Green)
    style.configure(
        "Success.TButton",
        font=("Segoe UI", 9, "bold"),
        background="#10b981",
        foreground="#ffffff",
        padding=(10, 4),
        borderwidth=0
    )
    style.map(
        "Success.TButton",
        background=[("pressed", "#047857"), ("active", "#34d399")]
    )

    # Save Button (Indigo)
    style.configure(
        "Indigo.TButton",
        font=("Segoe UI", 9, "bold"),
        background="#6366f1",
        foreground="#ffffff",
        padding=(10, 4),
        borderwidth=0
    )
    style.map(
        "Indigo.TButton",
        background=[("pressed", "#4338ca"), ("active", "#818cf8")]
    )

    # Clear Button (Soft Crimson / Danger)
    style.configure(
        "Danger.TButton",
        font=("Segoe UI", 9, "bold"),
        background="#ef4444",
        foreground="#ffffff",
        padding=(8, 4),
        borderwidth=0
    )
    style.map(
        "Danger.TButton",
        background=[("pressed", "#b91c1c"), ("active", "#f87171")]
    )

    # Progress bar and Status
    style.configure("Horizontal.TProgressbar", background="#2563eb", troughcolor="#e2e8f0", borderwidth=0)
    style.configure("Status.TFrame", background="#f8fafc", relief="solid", borderwidth=1)
    style.configure("Status.TLabel", background="#f8fafc", foreground=text_main, font=("Segoe UI", 9))


class OCRAppGUI:
    """Main Application GUI window controller."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        _enable_high_dpi()

        language.set_locale(DEFAULT_LOCALE)
        self.root.title(language("app_title"))
        self.root.geometry("960x680")
        self.root.minsize(800, 560)
        self.root.configure(bg="#f1f5f9")

        self.style = ttk.Style(self.root)
        _setup_theme_styles(self.style)

        # Initialize OCR engines
        self.engines = {
            "template_matching": TemplateMatchingOCR(),
            "hog_knn": HogKnnOCR(),
            "tf_model": TensorFlowOCR(),
        }
        self.engine_keys = ["template_matching", "hog_knn", "tf_model"]

        self.selected_image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self._is_processing = False
        self._last_preview_size = (0, 0)
        self._ocr_queue: queue.Queue = queue.Queue()

        self._setup_ui()
        self._setup_shortcuts()
        self._load_sample_files()

    def _setup_ui(self) -> None:
        """Constructs the application layout."""
        # Main container with outer margin
        main_box = ttk.Frame(self.root, padding=(12, 10, 12, 10))
        main_box.pack(fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # 1. Header Card (App Brand & Language Selector)
        # -------------------------------------------------------------
        header_card = ttk.Frame(main_box, style="Header.TFrame", padding=(14, 10, 14, 10))
        header_card.pack(fill=tk.X, pady=(0, 8))

        brand_frame = ttk.Frame(header_card, style="HeaderInner.TFrame")
        brand_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_title = ttk.Label(brand_frame, text=language("app_title"), style="HeaderTitle.TLabel")
        self.lbl_title.pack(anchor=tk.W)

        self.lbl_subtitle = ttk.Label(brand_frame, text=language("app_subtitle"), style="HeaderSubtitle.TLabel")
        self.lbl_subtitle.pack(anchor=tk.W)

        lang_frame = ttk.Frame(header_card, style="HeaderInner.TFrame")
        lang_frame.pack(side=tk.RIGHT, anchor=tk.CENTER)

        self.lbl_language = ttk.Label(lang_frame, text=language("language_label"), style="HeaderLang.TLabel")
        self.lbl_language.pack(side=tk.LEFT, padx=(0, 6))

        self.lang_combo = ttk.Combobox(
            lang_frame,
            values=["Polski (PL)", "English (EN)"],
            state="readonly",
            width=14
        )
        self.lang_combo.current(0 if language.locale == "pl" else 1)
        self.lang_combo.pack(side=tk.LEFT)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_changed)

        # -------------------------------------------------------------
        # 2. Configuration & Controls Card
        # -------------------------------------------------------------
        self.control_frame = ttk.LabelFrame(
            main_box,
            text=language("config_frame_title"),
            padding=(12, 8, 12, 10)
        )
        self.control_frame.pack(fill=tk.X, pady=(0, 8))

        # Grid config for responsive column distribution
        self.control_frame.columnconfigure(1, weight=1)

        # Step 1: Image source
        self.lbl_step1 = ttk.Label(self.control_frame, text=language("step1_title"), style="SectionStep.TLabel")
        self.lbl_step1.grid(row=0, column=0, sticky=tk.W, padx=6, pady=3)

        source_box = ttk.Frame(self.control_frame, style="Card.TFrame")
        source_box.grid(row=0, column=1, sticky=tk.EW, padx=6, pady=3)

        self.lbl_samples = ttk.Label(source_box, text=language("samples_label"), style="Card.TLabel")
        self.lbl_samples.pack(side=tk.LEFT, padx=(0, 6))

        self.samples_combo = ttk.Combobox(source_box, state="readonly", width=28)
        self.samples_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.samples_combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

        self.btn_browse = ttk.Button(
            source_box,
            text=language("browse_button"),
            style="Browse.TButton",
            command=self._browse_custom_image
        )
        self.btn_browse.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_current_file = ttk.Label(source_box, text="", style="FileBadge.TLabel")
        self.lbl_current_file.pack(side=tk.LEFT, padx=(4, 0))

        # Step 2: OCR Engine selection
        self.lbl_step2 = ttk.Label(self.control_frame, text=language("step2_title"), style="SectionStep.TLabel")
        self.lbl_step2.grid(row=1, column=0, sticky=tk.W, padx=6, pady=3)

        engine_box = ttk.Frame(self.control_frame, style="Card.TFrame")
        engine_box.grid(row=1, column=1, sticky=tk.EW, padx=6, pady=3)

        self.lbl_method = ttk.Label(engine_box, text=language("ocr_method_label"), style="Card.TLabel")
        self.lbl_method.pack(side=tk.LEFT, padx=(0, 6))

        self.engine_combo = ttk.Combobox(engine_box, state="readonly", width=36)
        self._refresh_engine_combo_values()
        self.engine_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_selected)

        self.lbl_engine_hint = ttk.Label(engine_box, text="", style="Hint.TLabel")
        self.lbl_engine_hint.pack(side=tk.LEFT, padx=(6, 0))
        self._update_engine_hint()

        # Action Buttons on Right
        action_box = ttk.Frame(self.control_frame, style="Card.TFrame")
        action_box.grid(row=0, column=2, rowspan=2, padx=(10, 4), pady=2, sticky=tk.E)

        self.btn_run = ttk.Button(
            action_box,
            text=f"▶  {language('run_button')}",
            style="Primary.TButton",
            command=self._run_ocr
        )
        self.btn_run.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        self.btn_clear = ttk.Button(
            action_box,
            text=language("clear_button"),
            style="Danger.TButton",
            command=self._clear_all
        )
        self.btn_clear.pack(side=tk.TOP, fill=tk.X)

        # -------------------------------------------------------------
        # 3. Content Split (Image Preview vs Recognized Text)
        # -------------------------------------------------------------
        content_pane = ttk.PanedWindow(main_box, orient=tk.HORIZONTAL)
        content_pane.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # --- Left: Preview Frame ---
        self.img_frame = ttk.LabelFrame(content_pane, text=language("preview_frame_title"), padding=8)
        content_pane.add(self.img_frame, weight=1)

        # Fixed container for preview to prevent resize oscillations
        self.preview_container = ttk.Frame(self.img_frame, style="Card.TFrame")
        self.preview_container.pack(fill=tk.BOTH, expand=True)
        self.preview_container.pack_propagate(False)
        self.preview_container.bind("<Configure>", self._on_preview_resize)

        self.image_label = ttk.Label(
            self.preview_container,
            text=language("no_image_label"),
            style="Placeholder.TLabel",
            anchor=tk.CENTER,
            justify=tk.CENTER
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.lbl_image_info = ttk.Label(self.img_frame, text="", style="Stats.TLabel", anchor=tk.W)
        self.lbl_image_info.pack(fill=tk.X, pady=(4, 0))

        # --- Right: Result Frame ---
        self.text_frame = ttk.LabelFrame(content_pane, text=language("result_frame_title"), padding=8)
        content_pane.add(self.text_frame, weight=1)

        # Toolbar above text result (Stats & Action buttons)
        text_toolbar = ttk.Frame(self.text_frame, style="Card.TFrame")
        text_toolbar.pack(fill=tk.X, pady=(0, 6))

        self.lbl_stats = ttk.Label(text_toolbar, text="", style="Stats.TLabel")
        self.lbl_stats.pack(side=tk.LEFT)

        self.btn_save = ttk.Button(
            text_toolbar,
            text=f"💾  {language('save_button')}",
            style="Indigo.TButton",
            command=self._save_to_file
        )
        self.btn_save.pack(side=tk.RIGHT, padx=(4, 0))

        self.btn_copy = ttk.Button(
            text_toolbar,
            text=f"📋  {language('copy_button')}",
            style="Success.TButton",
            command=self._copy_to_clipboard
        )
        self.btn_copy.pack(side=tk.RIGHT, padx=(4, 0))

        # Scrollable text result area
        text_area_box = ttk.Frame(self.text_frame, style="Card.TFrame")
        text_area_box.pack(fill=tk.BOTH, expand=True)

        self.text_result = tk.Text(
            text_area_box,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#0f172a",
            relief="solid",
            bd=1,
            padx=10,
            pady=10,
            selectbackground="#bfdbfe",
            undo=True
        )
        self.text_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_result.bind("<KeyRelease>", self._update_text_stats)

        self.text_scroll = ttk.Scrollbar(text_area_box, orient=tk.VERTICAL, command=self.text_result.yview)
        self.text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_result["yscrollcommand"] = self.text_scroll.set
        self._update_text_stats()

        # -------------------------------------------------------------
        # 4. Bottom Status Bar
        # -------------------------------------------------------------
        status_bar = ttk.Frame(main_box, style="Status.TFrame", padding=(10, 6, 10, 6))
        status_bar.pack(fill=tk.X)

        self.lbl_status = ttk.Label(status_bar, text=f"● {language('status_ready')}", style="Status.TLabel")
        self.lbl_status.pack(side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(status_bar, mode="indeterminate", length=160, style="Horizontal.TProgressbar")

    def _setup_shortcuts(self) -> None:
        """Binds convenient keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda _e: self._browse_custom_image())
        self.root.bind("<Control-O>", lambda _e: self._browse_custom_image())
        self.root.bind("<Control-r>", lambda _e: self._run_ocr())
        self.root.bind("<Control-R>", lambda _e: self._run_ocr())
        self.root.bind("<F5>", lambda _e: self._run_ocr())
        self.root.bind("<Control-s>", lambda _e: self._save_to_file())
        self.root.bind("<Control-S>", lambda _e: self._save_to_file())

    def _refresh_engine_combo_values(self) -> None:
        current_idx = self.engine_combo.current()
        if current_idx < 0:
            current_idx = 0
        display_names = [language(f"engines.{k}") for k in self.engine_keys]
        self.engine_combo["values"] = display_names
        self.engine_combo.current(current_idx)

    def _update_engine_hint(self) -> None:
        idx = self.engine_combo.current()
        if 0 <= idx < len(self.engine_keys):
            key = self.engine_keys[idx]
            hint = language(f"engine_hints.{key}")
            self.lbl_engine_hint.config(text=hint)

    def _on_engine_selected(self, _event=None) -> None:
        self._update_engine_hint()

    def _on_language_changed(self, _event=None) -> None:
        """Handles dynamic locale change."""
        chosen = self.lang_combo.get()
        new_locale = "pl" if "PL" in chosen else "en"
        language.set_locale(new_locale)

        # Update text strings dynamically
        self.root.title(language("app_title"))
        self.lbl_title.config(text=language("app_title"))
        self.lbl_subtitle.config(text=language("app_subtitle"))
        self.lbl_language.config(text=language("language_label"))
        self.control_frame.config(text=language("config_frame_title"))
        self.lbl_step1.config(text=language("step1_title"))
        self.lbl_step2.config(text=language("step2_title"))
        self.lbl_samples.config(text=language("samples_label"))
        self.btn_browse.config(text=language("browse_button"))
        self.lbl_method.config(text=language("ocr_method_label"))
        self.btn_run.config(text=f"▶  {language('run_button')}")
        self.btn_clear.config(text=language("clear_button"))
        self.btn_copy.config(text=f"📋  {language('copy_button')}")
        self.btn_save.config(text=f"💾  {language('save_button')}")
        self.img_frame.config(text=language("preview_frame_title"))
        self.text_frame.config(text=language("result_frame_title"))

        if not self.selected_image_path:
            self.image_label.config(text=language("no_image_label"))

        self._refresh_engine_combo_values()
        self._update_engine_hint()
        self._update_text_stats()
        self._update_image_meta()
        self.lbl_status.config(text=f"● {language('status_ready')}")

    def _load_sample_files(self) -> None:
        """Loads available sample images from the data/samples folder."""
        if SAMPLES_DIR.exists():
            samples = sorted([
                f.name for f in SAMPLES_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
            ])
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
        if not selected_name or selected_name == language("sample_no_files"):
            return
        sample_path = SAMPLES_DIR / selected_name
        if sample_path.exists() and sample_path.is_file():
            self._display_image(sample_path)

    def _browse_custom_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title=language("browse_button"),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._display_image(Path(file_path))

    def _display_image(self, path: Path) -> None:
        """Loads and prepares an image for rendering."""
        self.selected_image_path = path
        self.lbl_current_file.config(text=f"📁 {path.name}")
        try:
            if Image is not None:
                self.original_image = Image.open(path)
                self._update_image_meta()
                # Request container geometry to resize smoothly
                self.root.update_idletasks()
                w = self.preview_container.winfo_width()
                h = self.preview_container.winfo_height()
                self._render_preview(max(w, 200), max(h, 200))
            else:
                self.image_label.configure(text=f"Selected: {path.name}")
            self.lbl_status.config(text=f"● {language('status_loaded', filename=path.name)}")
        except Exception as e:
            messagebox.showerror("Error", language("error_load_failed", error=str(e)))

    def _update_image_meta(self) -> None:
        """Updates resolution and file size label below preview."""
        if not self.selected_image_path or self.original_image is None:
            self.lbl_image_info.config(text="")
            return
        w, h = self.original_image.size
        size_bytes = self.selected_image_path.stat().st_size
        if size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        self.lbl_image_info.config(text=f"ℹ  {language('image_meta', width=w, height=h, size=size_str)}")

    def _on_preview_resize(self, _event=None) -> None:
        """Handles responsive resize of image preview preserving aspect ratio."""
        if not self.selected_image_path or self.original_image is None:
            return
        w = self.preview_container.winfo_width()
        h = self.preview_container.winfo_height()
        if w <= 10 or h <= 10:
            return
        if abs(self._last_preview_size[0] - w) < 6 and abs(self._last_preview_size[1] - h) < 6:
            return
        self._last_preview_size = (w, h)
        self._render_preview(w, h)

    def _render_preview(self, max_w: int, max_h: int) -> None:
        """Scales image maintaining aspect ratio and displays on label."""
        if self.original_image is None or Image is None or ImageTk is None:
            return
        orig_w, orig_h = self.original_image.size
        if orig_w == 0 or orig_h == 0:
            return

        avail_w = max(50, max_w - 16)
        avail_h = max(50, max_h - 16)
        scale = min(avail_w / orig_w, avail_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        resample_mode = getattr(Image, "Resampling", Image).LANCZOS
        preview_img = self.original_image.resize((new_w, new_h), resample=resample_mode)
        self.tk_image = ImageTk.PhotoImage(preview_img)
        self.image_label.configure(image=self.tk_image, text="")

    def _update_text_stats(self, _event=None) -> None:
        """Calculates and updates character, word, and line count."""
        raw_text = self.text_result.get("1.0", tk.END)
        # tk.Text always appends a trailing newline
        if raw_text.endswith("\n"):
            raw_text = raw_text[:-1]
        chars = len(raw_text)
        words = len(raw_text.split()) if raw_text.strip() else 0
        lines = len(raw_text.splitlines()) if raw_text.strip() else 0
        self.lbl_stats.config(text=language("stats_info", chars=chars, words=words, lines=lines))

    def _run_ocr(self) -> None:
        """Triggers asynchronous OCR recognition."""
        if self._is_processing:
            return

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
        self._is_processing = True
        self.lbl_status.config(text=f"⏳ {language('status_processing', method=engine_display_name)}")
        self.btn_run.config(state=tk.DISABLED)
        self.btn_clear.config(state=tk.DISABLED)

        self.progress_bar.pack(side=tk.RIGHT, padx=6)
        self.progress_bar.start(10)
        self.root.update_idletasks()

        image_path = self.selected_image_path
        start_time = time.perf_counter()

        def _worker() -> None:
            try:
                image_name = image_path.name
                if Image is not None:
                    import numpy as np
                    img_np = np.array(Image.open(image_path))
                    try:
                        result = engine.recognize(img_np, image_name=image_name)
                    except TypeError:
                        result = engine.recognize(img_np)
                else:
                    try:
                        result = engine.recognize(image_path, image_name=image_name)
                    except TypeError:
                        result = engine.recognize(image_path)

                elapsed = time.perf_counter() - start_time
                self._ocr_queue.put(("success", result, elapsed))
            except Exception as e:
                self._ocr_queue.put(("error", str(e), 0.0))

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(40, self._poll_ocr_queue)

    def _poll_ocr_queue(self) -> None:
        """Polls the queue for OCR completion from the background thread."""
        try:
            status, payload, elapsed = self._ocr_queue.get_nowait()
            if status == "success":
                self.text_result.delete("1.0", tk.END)
                self.text_result.insert(tk.END, payload)
                self._update_text_stats()
                self.lbl_status.config(text=f"✓ {language('status_done_time', time=elapsed)}")
            else:
                self.lbl_status.config(text=f"✗ {language('status_error')}")
                messagebox.showerror("Error", language("error_ocr_failed", error=payload))
            self._finish_processing()
        except queue.Empty:
            if self._is_processing:
                self.root.after(40, self._poll_ocr_queue)

    def _finish_processing(self) -> None:
        """Resets execution state and control widgets."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run.config(state=tk.NORMAL)
        self.btn_clear.config(state=tk.NORMAL)
        self._is_processing = False

    def _copy_to_clipboard(self) -> None:
        """Copies recognized text to OS clipboard with brief button animation."""
        text = self.text_result.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Clipboard", language("info_empty_clipboard"))
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.lbl_status.config(text=f"✓ {language('status_copied')}")

        # Temporarily update copy button label for tactile visual feedback
        current_text = self.btn_copy.cget("text")
        self.btn_copy.config(text=f"✓  {language('copied_badge')}")

        def _restore_button() -> None:
            if self.btn_copy.winfo_exists():
                self.btn_copy.config(text=f"📋  {language('copy_button')}")

        self.root.after(1600, _restore_button)

    def _save_to_file(self) -> None:
        """Exports recognized text to a chosen .txt file."""
        text = self.text_result.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo(language("save_button"), language("info_empty_clipboard"))
            return

        default_name = f"ocr_{self.selected_image_path.stem}.txt" if self.selected_image_path else "ocr_result.txt"
        file_path = filedialog.asksaveasfilename(
            title=language("save_button"),
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[
                (language("file_dialog_txt"), "*.txt"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.lbl_status.config(text=f"✓ {language('status_saved', filename=Path(file_path).name)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def _clear_all(self) -> None:
        """Clears text result area and stats."""
        self.text_result.delete("1.0", tk.END)
        self._update_text_stats()
        self.lbl_status.config(text=f"● {language('status_ready')}")


def run() -> None:
    """Entry point for launching the GUI application."""
    root = tk.Tk()
    _app = OCRAppGUI(root)
    root.mainloop()
