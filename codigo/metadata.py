"""Acceso portable a metadatos de la cuenca.

Los scripts de modelacion pueden ejecutarse con los datos procesados incluidos
en el repositorio, aunque ``datos/raw`` no este disponible. Cuando existen los
archivos originales CAMELS se usan como fuente primaria; en caso contrario se
consulta ``datos/atributos.csv``.
"""

from __future__ import annotations

import pandas as pd

from paths import DATOS_DIR, RAW_DIR


def _normalized_gauge(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.zfill(8)


def _row_from_processed(gauge_id: str) -> pd.Series | None:
    path = DATOS_DIR / "atributos.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"gauge_id": "string"})
    if "gauge_id" not in frame.columns:
        return None
    row = frame.loc[_normalized_gauge(frame["gauge_id"]) == str(gauge_id).zfill(8)]
    return None if row.empty else row.iloc[0]


def gauge_name(gauge_id: str) -> str:
    """Devuelve el nombre oficial; funciona con o sin ``datos/raw``."""
    name_path = RAW_DIR / "camels_name.txt"
    if name_path.exists():
        names = pd.read_csv(name_path, sep=";", dtype={"gauge_id": "string"})
        row = names.loc[_normalized_gauge(names["gauge_id"]) == str(gauge_id).zfill(8)]
        if not row.empty:
            return str(row.iloc[0]["gauge_name"])

    row = _row_from_processed(gauge_id)
    if row is not None and pd.notna(row.get("gauge_name")):
        return str(row["gauge_name"])
    return f"Gauge {str(gauge_id).zfill(8)}"


def area_km2(gauge_id: str) -> float:
    """Devuelve ``area_gages2`` en km2; funciona con o sin datos crudos."""
    topo_path = RAW_DIR / "camels_topo.txt"
    if topo_path.exists():
        topo = pd.read_csv(topo_path, sep=";", dtype={"gauge_id": "string"})
        row = topo.loc[_normalized_gauge(topo["gauge_id"]) == str(gauge_id).zfill(8)]
        if not row.empty:
            return float(row.iloc[0]["area_gages2"])

    row = _row_from_processed(gauge_id)
    if row is not None and pd.notna(row.get("area_gages2")):
        return float(row["area_gages2"])

    raise FileNotFoundError(
        "No se encontro el area de la cuenca. Se requiere datos/atributos.csv "
        "o datos/raw/camels_topo.txt."
    )
