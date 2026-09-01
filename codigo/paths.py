from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CODIGO_DIR = ROOT_DIR / "codigo"
DATOS_DIR = ROOT_DIR / "datos"
RAW_DIR = DATOS_DIR / "raw"
FIG_DIR = ROOT_DIR / "figuras"
FIG_HTML_DIR = FIG_DIR / "html"
INFORME_DIR = ROOT_DIR / "informe"

