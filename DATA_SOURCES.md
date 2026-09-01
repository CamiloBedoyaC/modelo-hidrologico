# Fuentes y trazabilidad de los datos

Este repositorio **no versiona los datos crudos**. Los archivos grandes se
descargan de la fuente pública mediante `codigo/descargar_camels.py` y quedan
en `datos/raw/`, carpeta excluida por `.gitignore`.

## Fuente principal: CAMELS-US v1.2

- Registro oficial de distribución usado por el código: [CAMELS: Catchment
  Attributes and MEteorology for Large-sample
  Studies](https://zenodo.org/records/15529996), versión 1.2, publicado por
  NSF NCAR. El registro conserva el DOI del conjunto de series:
  [10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D).
- Licencia declarada por el registro: [Creative Commons Attribution
  4.0](https://creativecommons.org/licenses/by/4.0/) (`CC BY 4.0`).
- Series meteorologicas y caudal: `basin_timeseries_v1p2_metForcing_obsFlow.zip`
  (MD5 publicado: `8e9a466710e8270b58f01d332a87184f`).
- Límites de cuencas: `basin_set_full_res.zip`
  (MD5 publicado: `958fe520f6c4062dbddbbb67cfc28985`).
- Atributos estáticos: archivos `camels_*.txt` y
  `camels_attributes_v2.0.xlsx` del mismo registro.
- Cuenca analizada: USGS `02327100`, Sopchoppy River near Sopchoppy, Florida.
- Ventana de las series: 1980-01-01 a 2014-12-31; Maurer termina en 2008.

El script `codigo/descargar_camels.py` usa los enlaces de archivo de ese
registro y valida todos los MD5 publicados antes de aceptar una descarga.

## Referencias científicas

1. Newman, A. J., et al. (2015). *Development of a large-sample
   watershed-scale hydrometeorological data set for the contiguous USA*.
   Hydrology and Earth System Sciences, 19, 209-223.
   <https://doi.org/10.5194/hess-19-209-2015>
2. Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). *The CAMELS
   data set: catchment attributes and meteorology for large-sample studies*.
   Hydrology and Earth System Sciences, 21, 5293-5313.
   <https://doi.org/10.5194/hess-21-5293-2017>
3. Atributos CAMELS-US: <https://doi.org/10.5065/D6G73C3Q>
4. Series hidrometeorologicas CAMELS-US: <https://doi.org/10.5065/D6MW2F4D>

## Archivos procesados incluidos

Los CSV pequeños de `datos/` y `datos/02327100/` son productos derivados para
permitir la verificación rápida sin descargar 3.4 GB. Se regeneran con:

```bash
python -X utf8 codigo/descargar_camels.py --all --shapes
python -X utf8 codigo/preprocesar_02327100.py --gauge 02327100
python -X utf8 codigo/explorar_camels.py --gauge 02327100
```

La licencia MIT del repositorio cubre únicamente el código propio. Los datos y
productos derivados de CAMELS-US deben conservar su atribución y se rigen por
`CC BY 4.0`.
