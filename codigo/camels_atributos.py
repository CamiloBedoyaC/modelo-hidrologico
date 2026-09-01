from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import RAW_DIR

# ---------------------------------------------------------
# 1) Diccionario con los archivos de atributos CAMELS
# ---------------------------------------------------------

# Aquí organizamos todos los archivos de atributos en un diccionario.
# La clave es un nombre corto y el valor es la ruta completa del archivo.
ATTR_FILES = {
    "name": RAW_DIR / "camels_name.txt",
    "topo": RAW_DIR / "camels_topo.txt",
    "clim": RAW_DIR / "camels_clim.txt",
    "hydro": RAW_DIR / "camels_hydro.txt",
    "soil": RAW_DIR / "camels_soil.txt",
    "geol": RAW_DIR / "camels_geol.txt",
    "vege": RAW_DIR / "camels_vege.txt",
}

# ---------------------------------------------------------
# 2) Función interna para leer un archivo CAMELS
# ---------------------------------------------------------
def _read_camels_txt(path: Path) -> pd.DataFrame:
    """
    Lee un archivo .txt de atributos CAMELS y lo devuelve como DataFrame.
    """

    # Primero verifica que el archivo exista.
    # Si no existe, probablemente no se descargó el dataset.
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta `python codigo/descargar_camels.py --attributes-only`."
        )
    return pd.read_csv(path, sep=";", dtype={"gauge_id": "string"})

# ---------------------------------------------------------
# 3) Función principal: cargar todos los atributos juntos
# ---------------------------------------------------------
def load_attributes() -> pd.DataFrame:
    """
    Carga todos los archivos de atributos CAMELS y los une en un solo DataFrame.
    """

    # Empieza leyendo el archivo base (name),
    # que contiene el gauge_id y la información básica.
    df = _read_camels_txt(ATTR_FILES["name"])
    # Luego va recorriendo el resto de archivos
    # y los va uniendo uno por uno usando merge.
    for key in ["topo", "clim", "hydro", "soil", "geol", "vege"]:
        df = df.merge(_read_camels_txt(ATTR_FILES[key]), on="gauge_id", how="left")
    return df

# ---------------------------------------------------------
# 4) Función para obtener atributos de una sola estación
# ---------------------------------------------------------

def attributes_for_gauge(gauge_id: str) -> pd.Series:
    df = load_attributes()
    # Busca la fila donde gauge_id coincida.
    # Convierte ambos a string por seguridad (evita problemas de tipo).
    row = df.loc[df["gauge_id"].astype(str) == str(gauge_id)]
    # Si no encontró nada, significa que el ID no existe en CAMELS.
    if row.empty:
        raise ValueError(f"No existe gauge_id={gauge_id} en atributos CAMELS.")
    return row.iloc[0]
