"""Compila el informe LaTeX de forma portable en Windows, macOS y Linux."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from paths import INFORME_DIR


def compile_pdf(tex_path: Path, runs: int = 2, engine: str = "auto") -> Path:
    tex_path = tex_path.resolve()
    if not tex_path.exists():
        raise FileNotFoundError(tex_path)

    pdflatex = shutil.which("pdflatex") if engine in {"auto", "pdflatex"} else None
    tectonic = shutil.which("tectonic") if engine in {"auto", "tectonic"} else None
    if pdflatex is None and tectonic is None:
        raise RuntimeError(
            "No se encontro un motor TeX compatible. Instala MiKTeX/TeX Live "
            "(pdflatex) o Tectonic y agrega el ejecutable al PATH."
        )

    if pdflatex is not None:
        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ]
        total_runs = runs
    else:
        command = [tectonic, "--keep-logs", tex_path.name]
        total_runs = 1  # Tectonic repite internamente hasta estabilizar referencias.

    for run in range(1, total_runs + 1):
        print(f"[pdf] pasada {run}/{total_runs}: {' '.join(command)}")
        subprocess.run(command, cwd=tex_path.parent, check=True)

    output = tex_path.with_suffix(".pdf")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"No se genero {output}")
    print(f"[ok] {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tex",
        type=Path,
        default=INFORME_DIR / "reporte_tarea1.tex",
        help="Ruta al archivo .tex (por defecto: informe/reporte_tarea1.tex).",
    )
    parser.add_argument("--runs", type=int, default=2, help="Numero de pasadas de pdflatex.")
    parser.add_argument(
        "--engine",
        choices=["auto", "pdflatex", "tectonic"],
        default="auto",
        help="Motor TeX; auto usa pdflatex y, si no existe, Tectonic.",
    )
    args = parser.parse_args()
    compile_pdf(args.tex, runs=args.runs, engine=args.engine)


if __name__ == "__main__":
    main()
