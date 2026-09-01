"""\
Preprocesa la cuenca seleccionada (por defecto 02327100) usando CAMELS.

Requiere:
- Zip grande con forzamientos y caudal observado
- Archivos de atributos básicos (name y topo)

Genera archivos listos para usar en el modelo,
pero solo para una cuenca (para no superar 50 MB).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATOS_DIR, RAW_DIR
# ------------------------------
# Constantes de conversión
# ------------------------------
FT3S_TO_CMS = 0.0283168  #Conversión de pies cúbicos por segundo (cfs) a metros cúbicos por segundo (m³/s)
MMDAY_TO_CMS_PER_KM2 = 0.0115740741  
# Conversión: 1 mm/día sobre 1 km² equivale a 0.011574 m³/s
# Esto permite pasar de lámina de agua (mm/d) a caudal volumétrico.
# --------------------------------------------------
# 1) Obtener HUC02 de la cuenca
# --------------------------------------------------
def _huc02_for_gauge(gauge_id: str) -> str:
    name_path = RAW_DIR / 'camels_name.txt'
    if not name_path.exists():
        raise FileNotFoundError(
            f'No se encontró {name_path}. Ejecuta `python codigo/descargar_camels.py --attributes-only`.'
        )
    df = pd.read_csv(name_path, sep=';', dtype={'gauge_id': 'string', 'huc_02': 'string'})
    row = df.loc[df['gauge_id'].astype(str) == str(gauge_id)]
    if row.empty:
        raise ValueError(f'No existe gauge_id={gauge_id} en camels_name.txt')
    return str(row.iloc[0]['huc_02']).zfill(2)

# --------------------------------------------------
# 2) Obtener área de la cuenca
# --------------------------------------------------
def _area_km2_for_gauge(gauge_id: str) -> float:
    topo_path = RAW_DIR / 'camels_topo.txt'
    if not topo_path.exists():
        raise FileNotFoundError(
            f'No se encontró {topo_path}. Ejecuta `python codigo/descargar_camels.py --attributes-only`.'
        )
    df = pd.read_csv(topo_path, sep=';', dtype={'gauge_id': 'string'})
    row = df.loc[df['gauge_id'].astype(str) == str(gauge_id)]
    if row.empty:
        raise ValueError(f'No existe gauge_id={gauge_id} en camels_topo.txt')
    return float(row.iloc[0]['area_gages2'])

# --------------------------------------------------
# 3) Leer archivos de forzamiento (Daymet, Maurer, NLDAS)
# --------------------------------------------------
def _read_forcing_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r'\s+', skiprows=3)  # Lee archivo de forzamiento climático.
    # Los archivos CAMELS tienen 3 líneas iniciales que se saltan.
    df.columns = [
        'year', 'month', 'day', 'hour', 'dayl_s', 'prcp_mm_day', 'srad_w_m2',
        'swe_mm', 'tmax_c', 'tmin_c', 'vp_pa'
    ]
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    return df[['date', 'prcp_mm_day']]

# --------------------------------------------------
# 4) Leer caudal observado
# --------------------------------------------------
def _read_streamflow_txt(path: Path) -> pd.DataFrame:
    # En CAMELS el streamflow suele venir en ft3/s (cfs)
    cols = ['gauge_id', 'year', 'month', 'day', 'q_cfs', 'flag']
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols, dtype={'gauge_id': 'string'})
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df.loc[df['q_cfs'] < 0, 'q_cfs'] = np.nan
    return df[['date', 'q_cfs']]

# --------------------------------------------------
# 5) Extraer solo archivos necesarios del zip
# --------------------------------------------------
def extract_needed(zip_path: Path, gauge_id: str) -> dict[str, Path]:
    huc = _huc02_for_gauge(gauge_id)
    out_dir = DATOS_DIR / gauge_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rutas internas del zip para esa cuenca específica.
    members = {
        'streamflow': f'basin_dataset_public_v1p2/usgs_streamflow/{huc}/{gauge_id}_streamflow_qc.txt',
        'daymet': f'basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gauge_id}_lump_cida_forcing_leap.txt',
        'maurer': f'basin_dataset_public_v1p2/basin_mean_forcing/maurer/{huc}/{gauge_id}_lump_maurer_forcing_leap.txt',
        'nldas': f'basin_dataset_public_v1p2/basin_mean_forcing/nldas/{huc}/{gauge_id}_lump_nldas_forcing_leap.txt',
    }

    if not zip_path.exists():
        raise FileNotFoundError(
            f'No se encontró {zip_path}. Descárgalo con `python codigo/descargar_camels.py --all`.'
        )

    out_paths: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for key, member in members.items():
            if member not in names:
                raise FileNotFoundError(f'No se encontró {member} dentro del zip. (gauge={gauge_id}, huc={huc})')
            target = out_dir / Path(member).name
            out_paths[key] = target
            if target.exists() and target.stat().st_size > 0:
                continue
            with zf.open(member) as src, open(target, 'wb') as dst:
                dst.write(src.read())

    return out_paths

# --------------------------------------------------
# 6) Programa principal
# --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gauge', default='02327100')
    args = parser.parse_args()

    gauge_id = str(args.gauge)
    zip_path = RAW_DIR / 'basin_timeseries_v1p2_metForcing_obsFlow.zip'

    paths = extract_needed(zip_path, gauge_id)

    daymet = _read_forcing_txt(paths['daymet']).rename(columns={'prcp_mm_day': 'prcp_daymet_mm_day'})
    maurer = _read_forcing_txt(paths['maurer']).rename(columns={'prcp_mm_day': 'prcp_maurer_mm_day'})
    nldas = _read_forcing_txt(paths['nldas']).rename(columns={'prcp_mm_day': 'prcp_nldas_mm_day'})
    q = _read_streamflow_txt(paths['streamflow'])

    # Daymet define el calendario base (1980-2014). Maurer queda como NaN tras 2008.
    df = (
        daymet
        .merge(nldas, on='date', how='left')
        .merge(maurer, on='date', how='left')
        .merge(q, on='date', how='left')
    )
    df = df.sort_values('date').reset_index(drop=True)
    # Conversión de unidades
    area_km2 = _area_km2_for_gauge(gauge_id)
    coef_cms_per_mmday = area_km2 * MMDAY_TO_CMS_PER_KM2

    df['q_cms'] = df['q_cfs'] * FT3S_TO_CMS
    df['q_obs_mm_day'] = df['q_cms'] / coef_cms_per_mmday

    out_dir = DATOS_DIR / gauge_id
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / f'{gauge_id}_merged.csv'
    df.to_csv(merged_path, index=False)

    forcing = df[['date', 'prcp_daymet_mm_day', 'prcp_maurer_mm_day', 'prcp_nldas_mm_day']]
    forcing.to_csv(DATOS_DIR / 'forzamiento.csv', index=False)

    flow = df[['date', 'q_cfs', 'q_cms', 'q_obs_mm_day']]
    flow.to_csv(DATOS_DIR / 'caudal.csv', index=False)

    print(f'[ok] {merged_path}')
    print(f'[ok] {DATOS_DIR / "forzamiento.csv"}')
    print(f'[ok] {DATOS_DIR / "caudal.csv"}')


if __name__ == '__main__':
    main()
