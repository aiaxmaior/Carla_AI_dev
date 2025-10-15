# IconLibrary.py
# A simple library to manage and load icons and fonts for applications.

import os
import json
import logging
import pygame
from pathlib import Path
from typing import Dict, Optional
import re

# Define paths relative to the script's location
_ASSETS_PATH = os.path.join(os.path.dirname(__file__), "custom_assets")
_ASSETS_ICON_PATH = os.path.join(_ASSETS_PATH, "icons")
_FONTS_PATH = os.path.join(_ASSETS_PATH, "fonts")


pygame.init()

# Minimal-change IconLibrary for your JSON


class IconLibrary:
    """
    Expects JSON like:
    {
      "Defaults": {"default": "▪"},
      "status_alerts": {"default":"ℹ️","s":"✅","f":"❌","wn":"⚠️","critical":"💣","done":"🏁","query":"❓","skull":"☠️"},
      "file_data": {...},
      "sysUI": {...},
      "nav": {...},
      "items": {...}
    }
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, str]] = {}
        self._global_default: str = "•"
        self._load()

        # very small alias table so you can pass readable names
        self._aliases: Dict[str, Dict[str, str]] = {
            "status_alerts": {
                "ok": "s",
                "pass": "s",
                "success": "s",
                "fail": "f",
                "error": "f",
                "warn": "wn",
                "warning": "wn",
                "crit": "critical",
                "fatal": "critical",
                "completed": "done",
                "finish": "done",
                "question": "query",
                "help": "query",
                "info": "default",
                "i": "default",
            },
            "sysui": {"info": "i", "debug": "debug", "display": "dis", "screen": "dis"},
            "file_data": {"dir": "folder", "db": "database"},
            "nav": {
                "left": "l",
                "right": "r",
                "tri_left": "tril",
                "tri_right": "trir",
                "hand_left": "hl",
                "hand_right": "hr",
            },
            "items": {"gamepad": "gpad", "joystick": "js"},
        }

    def _load(self) -> None:
        for p in (Path("./custom_assets/icons/icons.json"), Path("./icons.json")):
            try:
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                    # global default
                    gd = (raw.get("Defaults") or {}).get("default")
                    if isinstance(gd, str) and gd:
                        self._global_default = gd
                    # keep categories as-is (including "sysUI"), but also build a lowercased view
                    self._data = {}
                    for cat, mapping in raw.items():
                        if cat == "Defaults" or not isinstance(mapping, dict):
                            continue
                        self._data[cat] = mapping
                        # also allow lowercase alias of the category key for case-insensitive lookup
                        lcat = cat.lower()
                        if lcat not in self._data:
                            self._data[lcat] = mapping
                    return
            except Exception as e:
                logging.warning(f"[IconLibrary] Failed loading {p}: {e}")
        # nothing found
        self._data = {}

    def _find_category(self, category: str) -> Optional[str]:
        # try exact, then lowercase; otherwise None
        if category in self._data:
            return category
        lc = category.lower()
        if lc in self._data:
            return lc
        # final pass: case/underscore-insensitive scan
        canon = self._canon(category)
        for k in self._data.keys():
            if self._canon(k) == canon:
                return k
        return None

    @staticmethod
    def _canon(s: str) -> str:
        return "".join(
            ch for ch in s.lower().replace("_", "").replace("-", "") if not ch.isspace()
        )

    def _alias(self, category_key: str, name: str) -> str:
        # map human words to your shortcodes (per-category)
        amap = self._aliases.get(category_key.lower(), {})
        return amap.get(name.lower(), name)

    def get_icon(self, category: str, name: str) -> str:
        if not category or not name:
            return self._global_default
        cat_key = self._find_category(category)
        if not cat_key:
            return self._global_default

        mapping = self._data.get(cat_key, {})
        # direct hit
        if name in mapping:
            return mapping[name]
        # case-insensitive hit
        for k, glyph in mapping.items():
            if k.lower() == name.lower():
                return glyph
        # alias hit
        alias = self._alias(cat_key, name)
        if alias in mapping:
            return mapping[alias]
        for k, glyph in mapping.items():
            if k.lower() == alias.lower():
                return glyph
        # category default
        if "default" in mapping:
            return mapping["default"]
        # global default
        return self._global_default

    # logging helpers (keep your calling style)
    def _to_level(self, level: str) -> int:
        return getattr(logging, str(level).upper(), logging.INFO)

    def ilog(self, level="info", message="", category="", name="", num_icon=1) -> None:
        icon = self.get_icon(category, name)
        logging.log(self._to_level(level), f"{icon * num_icon} {message}")

    def log(self, level: str, message: str, category: str, name: str) -> None:
        icon = self.get_icon(category, name)
        logging.log(self._to_level(level), f"{icon} {message}")


class FontLibrary:
    """
    Discovers fonts under ./custom_assets/fonts and returns pygame.font.Font objects
    via get_loaded_fonts(font=<stem or path or family>, type=<schema>, scale=<float>,
                         styles={<key>: {"bold":bool, "italic":bool}}).

    - If multiple faces exist (regular/bold/italic/bold_italic), it chooses the best face.
    - Otherwise it falls back to set_bold/set_italic on the loaded face.
    """

    def __init__(self):
        # stems -> absolute path
        self._stems: Dict[str, str] = {}
        # families -> { "regular": path, "bold": path, "italic": path, "bold_italic": path }
        self._families: Dict[str, Dict[str, str]] = {}
        # last build cache
        self._loaded: Dict[str, pygame.font.Font] = {}

        self._discover_fonts()

        # ONLY sizes live here; actual pygame font objects are created on demand
        self._schemas: Dict[str, Dict[str, int]] = {
            "default": {
                "title": 12,
                "main_score": 40,
                "sub_label": 40,
                "sub_value": 56,
                "large_val": 28,
                "small_label": 9,
            },
            "welcome_screen": {
                "title": 12,
                "main_score": 40,
                "sub_label": 40,
                "sub_value": 56,
                "large_val": 28,
                "small_label": 9,
            },
            "hud": {
                "title": 12,
                "main_score": 40,
                "sub_label": 40,
                "sub_value": 56,
                "large_val": 28,
                "small_label": 9,
            },
            # Schema your HUD uses
            "panel_fonts": {
                "title": 28,
                "main_score": 64,
                "sub_label": 24,
                "sub_value": 24,
                "large_val": 48,
                "small_label": 24,
                "critical_center": 48,
            },
            "mapping_screen": {
                "title": 28,
                "main": 48,
                "instructions": 72,
                "sub": 36,
                "sub_value": 24,
                "detected": 36,
            },
            "select_screen": {
                "title": 64,
                "subtitle": 32,
                "credits": 22,
                "prompt": 42,
            },
            "end_screen": {
                "title": 40,
                "sub_label": 24,
                "sub_value": 32
            }
        }       

        logging.info("[FontLibrary] initialized")

    # ---------- internals ----------

    def _discover_fonts(self) -> None:
        search_dir = Path("./custom_assets/fonts")
        if not search_dir.exists():
            logging.info(
                "[FontLibrary] No custom font directory; using pygame defaults"
            )
            return
        for pat in ("*.ttf", "*.otf", "*.ttc"):
            for f in search_dir.glob(pat):
                try:
                    stem = (
                        f.stem
                    )  # filename without ext, e.g. 'tt-supermolot-neue-trl.bd-it'
                    path = str(f.resolve())
                    self._stems[stem] = path

                    base, style = self._split_family_and_style(stem)
                    fam = self._families.setdefault(base, {})
                    fam_style_key = self._style_key(style["bold"], style["italic"])
                    fam[fam_style_key] = path
                except Exception as e:
                    logging.warning(f"[FontLibrary] Failed indexing {f}: {e}")

    def _split_family_and_style(self, stem: str):
        """
        Split a stem into (base_family, style_flags).
        We only treat tokens separated by non-alnum as style markers, so 'tt' doesn't trigger.
        Recognized tokens: bold: ['bd','bold','b'], italic: ['it','italic','i'].
        """
        tokens = re.split(r"[^A-Za-z0-9]+", stem.lower())
        bold = any(t in ("bd", "bold", "b") for t in tokens)
        italic = any(t in ("it", "italic", "i") for t in tokens)

        # Remove style tokens when reconstructing base family
        base_tokens = [
            t for t in tokens if t and t not in ("bd", "bold", "b", "it", "italic", "i")
        ]
        # Try to preserve major separators by reusing '-'
        base = "-".join(base_tokens) if base_tokens else stem
        return base, {"bold": bold, "italic": italic}

    @staticmethod
    def _style_key(bold: bool, italic: bool) -> str:
        if bold and italic:
            return "bold_italic"
        if bold:
            return "bold"
        if italic:
            return "italic"
        return "regular"

    def _resolve_font_file(
        self,
        font: Optional[str],
        bold: bool,
        italic: bool,
    ) -> Optional[str]:
        """
        Pick the best font file. Priority:
        1) If 'font' is a full path and exists -> use it (can't swap face).
        2) If 'font' matches a family base -> choose matching face (bold/italic/bold_italic/regular).
        3) If 'font' matches an indexed stem -> use that exact file.
        4) None -> use pygame default.
        """
        # Full path
        if font and os.path.exists(font):
            return font

        # Family by base name
        if font and font in self._families:
            fam = self._families[font]
            want = self._style_key(bold, italic)
            if want in fam:
                return fam[want]
            # fallback order within family
            for k in ("regular", "bold", "italic", "bold_italic"):
                if k in fam:
                    return fam[k]

        # Exact stem
        if font and font in self._stems:
            return self._stems[font]

        # No matching file -> pygame default
        return None

    def _get_schema(self, type_name: str) -> Dict[str, int]:
        if type_name in self._schemas:
            return dict(self._schemas[type_name])  # copy
        logging.info(f"[FontLibrary] Unknown schema '{type_name}', using 'default'")
        return dict(self._schemas["default"])

    # ---------- public API ----------

    def get_font_size_list(self, name: str = "default") -> Dict[str, int]:
        """Return a copy of the size schema (for inspection/testing)."""
        return self._get_schema(name)

    def get_loaded_fonts(
        self,
        font: Optional[str] = "tt-supermolot-neue-trl.bd-it",
        type: str = "default",
        scale: Optional[float] = None,
        styles: Optional[Dict[str, Dict[str, bool]]] = None,
    ) -> Dict[str, pygame.font.Font]:
        """
        Build and return dict[name -> pygame.font.Font] using a schema and optional scale/styles.

        - font: family base (recommended), stem, or full path.
          * If you have separate files, use the base family (e.g., 'tt-supermolot-neue-trl').
          * If you only have one file like '...bd-it.ttf', you can pass that stem or path.
        - styles: per-key style flags, e.g. {"main_score":{"bold":True}, "title":{"italic":True}}
        """
        styles = styles or {}
        sizes = self._get_schema(type)

        # scale sizes
        if scale is not None:
            try:
                s = float(scale)
                sizes = {k: max(1, int(v * s)) for k, v in sizes.items()}
            except Exception as e:
                logging.warning(f"[FontLibrary] Bad scale '{scale}': {e}")

        self._loaded = {}

        for key, px in sizes.items():
            want_bold = bool(styles.get(key, {}).get("bold", False))
            want_italic = bool(styles.get(key, {}).get("italic", False))

            chosen_path = self._resolve_font_file(font, want_bold, want_italic)

            try:
                if chosen_path:
                    fnt = pygame.font.Font(chosen_path, px)
                    # If the chosen file isn't the exact style we want, we can still nudge
                    # with set_bold/set_italic (harmless if already that style).
                    try:
                        fnt.set_bold(want_bold)
                        fnt.set_italic(want_italic)
                    except Exception:
                        pass
                else:
                    # pygame default font; simulate styles
                    fnt = pygame.font.Font(None, px)
                    try:
                        fnt.set_bold(want_bold)
                        fnt.set_italic(want_italic)
                    except Exception:
                        pass

                self._loaded[key] = fnt
            except Exception as e:
                logging.warning(
                    f"[FontLibrary] Fallback font for '{key}' ({px}px): {e}"
                )
                f = pygame.font.Font(None, max(1, px))
                try:
                    f.set_bold(want_bold)
                    f.set_italic(want_italic)
                except Exception:
                    pass
                self._loaded[key] = f

        return self._loaded
    