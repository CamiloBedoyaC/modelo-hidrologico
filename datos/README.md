# Datos del proyecto

- `forzamiento.csv`, `caudal.csv` y `atributos.csv`: insumos procesados de la
  cuenca 02327100 que permiten ejecutar el modelo sin los archivos crudos.
- `02327100/02327100_merged.csv`: serie diaria integrada usada en calibración.
- `resultados_calibracion*.csv` y `tabla_source_x_method*.csv`: resultados y
  mejores corridas por fuente y método.
- `two_tank_*`: salidas diarias y auditoría del balance hídrico.
- `auditoria_*`: recálculo independiente de las métricas reportadas.

Los datos crudos se descargan en `datos/raw/` y no se publican en GitHub. La
procedencia, DOI y checksums están documentados en
[`../DATA_SOURCES.md`](../DATA_SOURCES.md).
