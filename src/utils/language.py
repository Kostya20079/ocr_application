import json
from typing import Any
from src.utils.helpers import get_resource_path

class Language:
    """Manager supporting Polish (default) and English."""

    DEFAULT_LOCALE = "pl"
    SUPPORTED_LOCALES = ["pl", "en"]

    def __init__(self, default_locale: str = DEFAULT_LOCALE) -> None:
        self.locale = default_locale if default_locale in self.SUPPORTED_LOCALES else self.DEFAULT_LOCALE
        self._translations: dict[str, dict[str, Any]] = {}
        self._load_translations()

    def _load_translations(self) -> None:
        locales_dir = get_resource_path("locales")
        for loc in self.SUPPORTED_LOCALES:
            file_path = locales_dir / loc / "messages.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self._translations[loc] = json.load(f)
            else:
                self._translations[loc] = {}

    def set_locale(self, locale: str) -> None:
        if locale in self.SUPPORTED_LOCALES:
            self.locale = locale

    def get(self, key: str, **kwargs: Any) -> str:
        """Fetch translated text by key (dot-notation supported), with optional formatting parameters."""
        parts = key.split(".")
        data: Any = self._translations.get(self.locale, {})

        for part in parts:
            if isinstance(data, dict):
                data = data.get(part)
            else:
                data = None
                break

        # Fallback to default locale if missing in current locale
        if data is None and self.locale != self.DEFAULT_LOCALE:
            data = self._translations.get(self.DEFAULT_LOCALE, {})
            for part in parts:
                if isinstance(data, dict):
                    data = data.get(part)
                else:
                    data = None
                    break

        if data is None:
            return key

        if kwargs and isinstance(data, str):
            try:
                return data.format(**kwargs)
            except KeyError:
                return data

        return str(data)

    def __call__(self, key: str, **kwargs: Any) -> str:
        return self.get(key, **kwargs)

# Global default instance
language = Language()

