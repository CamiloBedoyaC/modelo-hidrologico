from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from paths import DATOS_DIR, FIG_DIR, INFORME_DIR, ROOT_DIR


ROOT_FILES = [
    ".gitattributes",
    ".gitignore",
    ".nojekyll",
    "CITATION.cff",
    "DATA_SOURCES.md",
    "LICENSE",
    "README.md",
    "index.html",
    "requirements.txt",
]


def _is_temporary(rel: Path) -> bool:
    name = rel.name.lower()
    return (
        rel.suffix.lower() in {".aux", ".log", ".out", ".toc", ".pyc", ".zip"}
        or name.endswith(".synctex.gz")
        or "_tmp." in name
        or name.endswith((".tar.gz", ".tgz"))
    )


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT_DIR)
    # Datos crudos, geometrías globales y extracciones CAMELS pesadas.
    if rel.parts[:2] == ("datos", "raw"):
        return False
    if rel.parts[:2] == ("datos", "shapes"):
        return False
    if rel.parts[:2] == ("datos", "02327100") and rel.name.endswith(
        ("_forcing_leap.txt", "_streamflow_qc.txt")
    ):
        return False
    if "__pycache__" in rel.parts:
        return False
    if _is_temporary(rel):
        return False
    return True


def add_path(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and should_include(child):
                tar.add(child, arcname=str(Path(arcname) / child.relative_to(path)))
    elif path.is_file() and should_include(path):
        tar.add(path, arcname=arcname)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea una entrega reproducible sin datos CAMELS crudos ni temporales."
    )
    parser.add_argument("--out", default="entrega_tarea1.tar.gz")
    parser.add_argument("--prefix", default="tarea1")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    prefix = args.prefix

    with tarfile.open(out, "w:gz") as tar:
        add_path(tar, ROOT_DIR / "codigo", f"{prefix}/codigo")
        add_path(tar, ROOT_DIR / "tests", f"{prefix}/tests")
        add_path(tar, DATOS_DIR, f"{prefix}/datos")
        add_path(tar, FIG_DIR, f"{prefix}/figuras")
        add_path(tar, INFORME_DIR, f"{prefix}/informe")
        add_path(tar, ROOT_DIR / ".github", f"{prefix}/.github")
        for filename in ROOT_FILES:
            add_path(tar, ROOT_DIR / filename, f"{prefix}/{filename}")

    size_mb = out.stat().st_size / 1024**2
    print(f"[ok] Empaquetado en {out} ({size_mb:.2f} MB).")
    print("[ok] Excluidos: datos crudos, shapes globales, caches, archivos comprimidos y temporales.")


if __name__ == "__main__":
    main()
