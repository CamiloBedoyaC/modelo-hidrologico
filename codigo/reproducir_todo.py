"""Orquesta la reproduccion completa usando el mismo Python del entorno activo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paths import ROOT_DIR


def _run(script: str, *arguments: str) -> None:
    command = [sys.executable, "-X", "utf8", str(ROOT_DIR / "codigo" / script), *arguments]
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--desde-cero",
        action="store_true",
        help="Descarga CAMELS-US (aprox. 3.4 GB) y regenera los datos procesados.",
    )
    parser.add_argument(
        "--omitir-calibracion",
        action="store_true",
        help="Usa las tablas calibradas incluidas; util para una verificacion rapida.",
    )
    parser.add_argument("--omitir-pdf", action="store_true", help="No ejecuta pdflatex.")
    args = parser.parse_args()

    if args.desde_cero:
        _run("descargar_camels.py", "--attributes-only", "--shapes")
        _run("explorar_camels.py", "--gauge", "02327100")
        _run("mapa_cuenca_leaflet.py", "--gauge", "02327100")
        _run("descargar_camels.py", "--all")
        _run("preprocesar_02327100.py", "--gauge", "02327100")

    _run("explorar_datos.py", "--gauge", "02327100", "--precip", "prcp_daymet_mm_day")
    _run("ejecutar_modelo.py", "--gauge", "02327100", "--sources", "daymet", "nldas")
    _run("dashboard_balance.py", "--gauge", "02327100")

    if not args.omitir_calibracion:
        _run(
            "calibrar.py",
            "--gauge", "02327100",
            "--sources", "daymet", "nldas",
            "--methods", "de", "da",
            "--n-seeds", "3",
            "--maxiter", "40",
            "--popsize", "12",
        )
        _run(
            "calibrar_extremos.py",
            "--gauge", "02327100",
            "--sources", "daymet", "nldas",
            "--methods", "de", "da",
            "--n-seeds", "1",
            "--maxiter", "8",
            "--popsize", "8",
        )

    _run("replot_calibrados.py")
    _run("diagnostico_validacion.py", "--gauge", "02327100")
    _run("generar_reporte_standalone.py")
    if not args.omitir_pdf:
        _run("compilar_pdf.py")
    _run("verificar_reproducibilidad.py", "--write-audits")

    print("\n[OK] Reproduccion terminada sin errores.")


if __name__ == "__main__":
    main()
