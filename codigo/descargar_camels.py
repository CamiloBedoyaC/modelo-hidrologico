"""\
Descarga archivos principales del dataset CAMELS (Zenodo record 15529996).

- Atributos (xlsx) siempre se descargan a `datos/raw/`.
- El zip grande (~3.4 GB) con forzamientos y caudal observado es opcional.

Uso:
    python codigo/descargar_camels.py --attributes-only
    python codigo/descargar_camels.py --all

"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

from paths import RAW_DIR
# Este es el zip grande (forzamientos meteorológicos + caudal observado).
# Tiene ~3.4 GB, por eso lo dejamos como “opcional”.
FORCING_URL = (
    "https://zenodo.org/records/15529996/files/"
    "basin_timeseries_v1p2_metForcing_obsFlow.zip?download=1"
)

KNOWN_MD5 = {
    "readme.txt": "b37d64950e9d4c5c10a8b4ef82bc6219",
    "camels_name.txt": "c96491b32c4df55a31bead7ceca7d64b",
    "camels_topo.txt": "0f6267838c40b1507b64582433bc0b8e",
    "camels_clim.txt": "67f22592f3fb72c57df81358ce68458b",
    "camels_soil.txt": "8edb46a363a20b466a4b7105ba633767",
    "camels_geol.txt": "f5ce5de53eb1ea2532cda7e3b4813993",
    "camels_vege.txt": "f40e843defc1e654a800be9fe5fd5090",
    "camels_hydro.txt": "55ebdeb36c42ee7acdb998229c3edb3a",
    "camels_attributes_v2.0.xlsx": "714c68bd5bb3314ca39b14f9467bd609",
    "camels_attributes_v2.0.pdf": "77a6c084c798a31fbd05594ee58a90c7",
    "basin_timeseries_v1p2_metForcing_obsFlow.zip": "8e9a466710e8270b58f01d332a87184f",
    "basin_set_full_res.zip": "958fe520f6c4062dbddbbb67cfc28985",
}

# Diccionario con archivos pequeños/medianos que siempre conviene descargar:
# textos con nombres, atributos por categoría, y el Excel+PDF principales.
# Clave = nombre final del archivo en la carpeta, valor = link de descarga.
ATTR_FILES: dict[str, str] = {
    "readme.txt": "https://zenodo.org/records/15529996/files/readme.txt?download=1",
    "camels_name.txt": "https://zenodo.org/records/15529996/files/camels_name.txt?download=1",
    "camels_topo.txt": "https://zenodo.org/records/15529996/files/camels_topo.txt?download=1",
    "camels_clim.txt": "https://zenodo.org/records/15529996/files/camels_clim.txt?download=1",
    "camels_soil.txt": "https://zenodo.org/records/15529996/files/camels_soil.txt?download=1",
    "camels_geol.txt": "https://zenodo.org/records/15529996/files/camels_geol.txt?download=1",
    "camels_vege.txt": "https://zenodo.org/records/15529996/files/camels_vege.txt?download=1",
    "camels_hydro.txt": "https://zenodo.org/records/15529996/files/camels_hydro.txt?download=1",
    "camels_attributes_v2.0.xlsx": (
        "https://zenodo.org/records/15529996/files/camels_attributes_v2.0.xlsx?download=1"
    ),
    "camels_attributes_v2.0.pdf": (
        "https://zenodo.org/records/15529996/files/camels_attributes_v2.0.pdf?download=1"
    ),
}


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324: checksum de integridad publicado por Zenodo
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    chunk_size: int = 1024 * 1024,
    expected_md5: str | None = None,
) -> None:
    #Descarga un archivo desde `url` y lo guarda en `dest`.
    # Se descarga en “pedacitos” (chunks) para no saturar memoria.
    # Muestra una barra de progreso con tqdm.
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    if partial.exists():
        partial.unlink()

    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        # Creamos la barra de progreso.
        # unit="B" = bytes, unit_scale=True = lo convierte a KB/MB/GB automáticamente.
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        downloaded = 0
        with partial.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
        pbar.close()

    if total and downloaded != total:
        partial.unlink(missing_ok=True)
        raise IOError(f"Descarga incompleta: {downloaded} de {total} bytes para {dest.name}")
    if expected_md5 and _md5(partial).lower() != expected_md5.lower():
        partial.unlink(missing_ok=True)
        raise IOError(f"Checksum MD5 incorrecto para {dest.name}")
    partial.replace(dest)


def ensure_file(url: str, dest: Path) -> Path:
    """
    Verifica si `dest` ya existe y tiene tamaño > 0.
    - Si existe, lo salta.
    - Si no, lo descarga.
    """
    # Si el archivo existe y no está vacío, no tiene sentido bajarlo otra vez.
    expected_md5 = KNOWN_MD5.get(dest.name)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_md5 and _md5(dest).lower() != expected_md5.lower():
            raise IOError(
                f"El archivo existente no coincide con el MD5 publicado: {dest}. "
                "Eliminalo y vuelve a ejecutar la descarga."
            )
        print(f"[skip] Ya existe y es valido: {dest}")
        return dest
    print(f"[downloading] {dest.name}")
    download(url, dest, expected_md5=expected_md5)
    print(f"[ok] Guardado en {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Descarga también el zip grande (~3.4 GB).")
     # --shapes descarga un zip adicional con shapefiles (más pesado pero útil para mapas).
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="Descarga el shapefile full-res (~45 MB) con todas las cuencas.",
    )
    parser.add_argument(
        "--attributes-only",
        action="store_true",
        help="Descarga solo atributos (rápido).",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    do_all = args.all or (not args.attributes_only)
    if do_all:
        ensure_file(FORCING_URL, RAW_DIR / "basin_timeseries_v1p2_metForcing_obsFlow.zip")

    for fname, url in ATTR_FILES.items():
        ensure_file(url, RAW_DIR / fname)

    if args.shapes:
        ensure_file(
            "https://zenodo.org/records/15529996/files/basin_set_full_res.zip?download=1",
            RAW_DIR / "basin_set_full_res.zip",
        )

    print("\nListo.")


if __name__ == "__main__":
    main()
