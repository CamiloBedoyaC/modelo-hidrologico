from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from plotly.offline import get_plotlyjs

from paths import ROOT_DIR


IFRAME_OPEN_RE = re.compile(r"<iframe\b(?P<attrs>[^>]*)>", re.IGNORECASE | re.DOTALL)
SRC_RE = re.compile(r"""\bsrc\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
DATA_SRC_RE = re.compile(r"""\bdata-src\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
SRCDOC_RE = re.compile(r"""\bsrcdoc\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
PLOTLY_CDN_RE = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*([\"'])https://cdn\.plot\.ly/plotly-[^\"']+\1[^>]*>\s*</script>",
    re.IGNORECASE,
)


def _strip_attr(attrs: str, name: str) -> str:
    pattern = re.compile(
        rf"""(?i)(?:^|\s){re.escape(name)}\s*=\s*(["'])(.*?)\1""",
        re.DOTALL,
    )
    return pattern.sub("", attrs)


def _is_external(url: str) -> bool:
    u = url.strip()
    if not u:
        return True
    if u.startswith(("#", "data:", "javascript:", "about:")):
        return True
    parts = urlsplit(u)
    return bool(parts.scheme or parts.netloc)


def _resolve_local_path(url: str, base_dir: Path) -> Path | None:
    if _is_external(url):
        return None
    parts = urlsplit(url)
    rel_path = unquote(parts.path)
    if not rel_path:
        return None
    return (base_dir / rel_path).resolve()


def _replace_iframes(html_text: str, base_dir: Path, cache: dict[Path, str], stack: set[Path]) -> str:
    def repl(match: re.Match[str]) -> str:
        attrs = match.group("attrs")

        src_match = SRC_RE.search(attrs)
        data_src_match = DATA_SRC_RE.search(attrs)

        source_url = None
        source_from_data_src = False
        if src_match:
            source_url = src_match.group(2)
        elif data_src_match:
            source_url = data_src_match.group(2)
            source_from_data_src = True

        if not source_url:
            return match.group(0)

        local_path = _resolve_local_path(source_url, base_dir)
        if local_path is None or not local_path.exists() or local_path.suffix.lower() != ".html":
            return match.group(0)

        inlined = _inline_file(local_path, cache, stack)
        inlined_escaped = html.escape(inlined, quote=True)

        new_attrs = attrs
        new_attrs = _strip_attr(new_attrs, "src")
        new_attrs = _strip_attr(new_attrs, "srcdoc")
        new_attrs = re.sub(r"\s+", " ", new_attrs).rstrip()

        if source_from_data_src:
            # Evita que scripts internos cambien a src externo al seleccionar opción.
            new_attrs += ' src="about:blank"'

        new_attrs += f' srcdoc="{inlined_escaped}"'
        return f"<iframe{new_attrs}>"

    return IFRAME_OPEN_RE.sub(repl, html_text)


def _inline_file(path: Path, cache: dict[Path, str], stack: set[Path]) -> str:
    if path in cache:
        return cache[path]
    if path in stack:
        return path.read_text(encoding="utf-8", errors="replace")

    stack.add(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    # En el standalone, cada srcdoc comparte origen con el documento superior.
    # Reutilizar una única copia de Plotly evita 22 descargas CDN y mantiene el
    # archivo por debajo del límite de advertencia de GitHub (50 MiB).
    text = PLOTLY_CDN_RE.sub("<script>window.Plotly = window.top.Plotly;</script>", text)
    text = _replace_iframes(text, path.parent, cache, stack)
    stack.remove(path)
    cache[path] = text
    return text


def build_standalone(input_html: Path, output_html: Path) -> None:
    cache: dict[Path, str] = {}
    inlined = _inline_file(input_html.resolve(), cache, set())
    plotly_script = f"<script>{get_plotlyjs()}</script>"
    if "</head>" not in inlined.lower():
        raise ValueError(f"El HTML base no contiene </head>: {input_html}")
    inlined = re.sub(
        r"</head>",
        lambda match: f"{plotly_script}\n{match.group(0)}",
        inlined,
        count=1,
        flags=re.IGNORECASE,
    )
    output_html.write_text(inlined, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un reporte HTML standalone incrustando iframes locales como srcdoc."
    )
    parser.add_argument(
        "--input",
        default="informe/reporte_tarea1_interactivo.html",
        help="Ruta al HTML base.",
    )
    parser.add_argument(
        "--output",
        default="informe/reporte_tarea1_interactivo_standalone.html",
        help="Ruta del HTML standalone de salida.",
    )
    args = parser.parse_args()

    input_html = Path(args.input)
    output_html = Path(args.output)
    if not input_html.is_absolute():
        input_html = ROOT_DIR / input_html
    if not output_html.is_absolute():
        output_html = ROOT_DIR / output_html

    if not input_html.exists():
        raise FileNotFoundError(f"No se encontró el archivo base: {input_html}")

    output_html.parent.mkdir(parents=True, exist_ok=True)
    build_standalone(input_html, output_html)
    print(f"[ok] Standalone generado: {output_html}")


if __name__ == "__main__":
    main()
