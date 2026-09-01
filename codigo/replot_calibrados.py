"""Re-genera hidrogramas calibrados (HTML) sin re-optimizar y crea HTML combinados por fuente.

Usa tablas existentes:
- datos/tabla_source_x_method.csv (calibracion estandar)
- datos/tabla_source_x_method_extremos.csv (calibracion extremos)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import calibrar
import calibrar_extremos
from paths import DATOS_DIR, FIG_DIR, FIG_HTML_DIR


def _force_dark_body(html_path: Path) -> None:
    """Normalize Plotly standalone HTML margins/background for iframe embedding."""
    if not html_path.exists():
        return
    txt = html_path.read_text(encoding="utf-8", errors="ignore")
    # Plotly standalone pages usually write plain <html>/<body>; force dark chrome.
    txt2 = txt
    txt2 = txt2.replace("<html>", '<html style="background:#000;">', 1)
    txt2 = txt2.replace(
        '<body style="margin:0;background:#000;">',
        '<body style="margin:0;background:#000;overflow:hidden;">',
        1,
    )
    txt2 = txt2.replace("<body>", '<body style="margin:0;background:#000;overflow:hidden;">', 1)
    if txt2 != txt:
        html_path.write_text(txt2, encoding="utf-8")


def _write_toggle_html(out_path: Path, options: list[tuple[str, str]], height: int = 680) -> None:
    select_options = "\n".join([f'<option value="{fname}">{label}</option>' for label, fname in options])
    frames = "\n".join(
        [
            f'<iframe class="plot-frame" data-src="{fname}" style="display:none"></iframe>'
            for _, fname in options
        ]
    )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Selector de hidrogramas</title>
  <style>
    html, body {{ margin:0; background:#000; color:#fff; font-family:Arial, sans-serif; }}
    .toolbar {{ display:flex; gap:8px; align-items:center; padding:10px 12px; border-bottom:1px solid #222; }}
    select {{ background:#111826; color:#fff; border:1px solid #2a3140; border-radius:6px; padding:6px 10px; }}
    .container {{ padding:0; background:#000; }}
    .plot-frame {{ width:100%; height:{height}px; border:0; display:block; background:#000; }}
  </style>
</head>
<body>
  <div class="toolbar">
    <span>Selecciona método:</span>
    <select id="sel">
      {select_options}
    </select>
  </div>
  <div class="container">
    {frames}
  </div>
  <script>
    const sel = document.getElementById('sel');
    const frames = Array.from(document.querySelectorAll('.plot-frame'));
    function showSelected() {{
      const value = sel.value;
      frames.forEach((f) => {{
        if (f.getAttribute('data-src') === value) {{
          f.style.display = 'block';
          if (!f.src) f.src = value;
        }} else {{
          f.style.display = 'none';
        }}
      }});
    }}
    sel.addEventListener('change', showSelected);
    showSelected();
  </script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def _replot_standard(gauge_id: str) -> None:
    table_path = DATOS_DIR / "tabla_source_x_method.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    df_tab = pd.read_csv(table_path, dtype={"gauge_id": "string"})
    df_tab = df_tab[df_tab["gauge_id"].astype(str).str.zfill(8) == str(gauge_id).zfill(8)]
    if df_tab.empty:
        raise RuntimeError("No hay filas en tabla_source_x_method.csv para la cuenca.")

    gauge_name = calibrar._gauge_name(gauge_id)
    gauge_label = f"{gauge_id} ({gauge_name})"
    area_km2 = calibrar._area_km2_for_gauge(gauge_id)
    df_merged = calibrar._load_merged(gauge_id)
    source_frames = {s: calibrar._dataset_for_source(df_merged, s) for s in ["daymet", "nldas"]}

    for _, row in df_tab.iterrows():
        source = str(row["source"])
        method = str(row["method"])
        params = calibrar._params_from_row(row)
        df_eval, met_cal, met_val, _bal, _peaks = calibrar.evaluate(source_frames[source], params, area_km2)
        title = (
            f"Modelo calibrado "
            f"({source}, método={method}) | NSE cal={met_cal['NSE']:.3f}, NSE val={met_val['NSE']:.3f}"
        )
        out_html = FIG_HTML_DIR / f"hidrograma_calibrado_{source}_{method}.html"
        out_png = FIG_DIR / f"hidrograma_calibrado_{source}_{method}.png"
        calibrar._plot_hydrograph_png(
            df_eval,
            title=title,
            out_png=out_png,
            precip_label=calibrar.SOURCE_TO_LABEL[source],
        )
        calibrar._plot_hydrograph_html(
            df_eval,
            title=title,
            out_html=out_html,
            precip_label=calibrar.SOURCE_TO_LABEL[source],
        )
        _force_dark_body(out_html)


def _replot_extremos(gauge_id: str) -> None:
    table_path = DATOS_DIR / "tabla_source_x_method_extremos.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    df_tab = pd.read_csv(table_path, dtype={"gauge_id": "string"})
    df_tab = df_tab[df_tab["gauge_id"].astype(str).str.zfill(8) == str(gauge_id).zfill(8)]
    if df_tab.empty:
        raise RuntimeError("No hay filas en tabla_source_x_method_extremos.csv para la cuenca.")

    gauge_name = calibrar_extremos._gauge_name(gauge_id)
    gauge_label = f"{gauge_id} ({gauge_name})"
    area_km2 = calibrar_extremos._area_km2_for_gauge(gauge_id)
    df_merged = calibrar_extremos._load_merged(gauge_id)
    source_frames = {s: calibrar_extremos._dataset_for_source(df_merged, s) for s in ["daymet", "nldas"]}

    for _, row in df_tab.iterrows():
        source = str(row["source"])
        method = str(row["method"])
        params = calibrar_extremos._params_from_row(row)
        df_eval, metrics, _extras = calibrar_extremos.evaluate(source_frames[source], params, area_km2)
        title = (
            f"Modelo calibrado extremos "
            f"({source}, método={method}) | J={metrics['J_cal']:.3f}, NSE cal={metrics['NSE_cal']:.3f}, "
            f"NSE val={metrics['NSE_val']:.3f}"
        )
        out_html = FIG_HTML_DIR / f"hidrograma_calibrado_extremos_{source}_{method}.html"
        out_png = FIG_DIR / f"hidrograma_calibrado_extremos_{source}_{method}.png"
        calibrar_extremos._plot_hydrograph_png(
            df_eval,
            title=title,
            out_png=str(out_png),
            precip_label=calibrar_extremos.SOURCE_TO_LABEL[source],
        )
        calibrar_extremos._plot_hydrograph_html(
            df_eval,
            title=title,
            out_html=str(out_html),
            precip_label=calibrar_extremos.SOURCE_TO_LABEL[source],
        )
        _force_dark_body(out_html)


def main() -> None:
    gauge_id = "02327100"
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)

    _replot_standard(gauge_id)
    _replot_extremos(gauge_id)

    _write_toggle_html(
        FIG_HTML_DIR / "hidrograma_calibrado_daymet.html",
        [
            ("Daymet - DA", "hidrograma_calibrado_daymet_da.html"),
            ("Daymet - DE", "hidrograma_calibrado_daymet_de.html"),
        ],
    )
    _write_toggle_html(
        FIG_HTML_DIR / "hidrograma_calibrado_nldas.html",
        [
            ("NLDAS - DA", "hidrograma_calibrado_nldas_da.html"),
            ("NLDAS - DE", "hidrograma_calibrado_nldas_de.html"),
        ],
    )
    _write_toggle_html(
        FIG_HTML_DIR / "hidrograma_calibrado_extremos_daymet.html",
        [
            ("Extremos Daymet - DA", "hidrograma_calibrado_extremos_daymet_da.html"),
            ("Extremos Daymet - DE", "hidrograma_calibrado_extremos_daymet_de.html"),
        ],
    )
    _write_toggle_html(
        FIG_HTML_DIR / "hidrograma_calibrado_extremos_nldas.html",
        [
            ("Extremos NLDAS - DA", "hidrograma_calibrado_extremos_nldas_da.html"),
            ("Extremos NLDAS - DE", "hidrograma_calibrado_extremos_nldas_de.html"),
        ],
    )

    # Validación (HTML interactivo) usando el mejor combo global por NSE_val
    std_table = pd.read_csv(DATOS_DIR / "tabla_source_x_method.csv", dtype={"gauge_id": "string"})
    std_table = std_table[std_table["gauge_id"].astype(str).str.zfill(8) == str(gauge_id).zfill(8)]
    if not std_table.empty:
        best_source = std_table.loc[std_table.groupby("source")["NSE_val"].idxmax()].sort_values("source")
        overall_best = best_source.sort_values("NSE_val", ascending=False).iloc[0]
        source = str(overall_best["source"])
        method = str(overall_best["method"])
        params = calibrar._params_from_row(overall_best)

        gauge_name = calibrar._gauge_name(gauge_id)
        gauge_label = f"{gauge_id} ({gauge_name})"
        area_km2 = calibrar._area_km2_for_gauge(gauge_id)
        df_merged = calibrar._load_merged(gauge_id)
        source_frames = {s: calibrar._dataset_for_source(df_merged, s) for s in ["daymet", "nldas"]}

        df_eval, _met_cal, met_val, _bal, _peaks = calibrar.evaluate(source_frames[source], params, area_km2)
        df_val = df_eval[(df_eval["date"] >= calibrar.VAL_START) & (df_eval["date"] <= calibrar.VAL_END)].copy()

        title = f"Validación - {source.upper()} ({method.upper()}) | NSE={met_val['NSE']:.3f}"
        calibrar._plot_hydrograph_html(
            df_val,
            title=title,
            out_html=FIG_HTML_DIR / "validacion.html",
            precip_label=calibrar.SOURCE_TO_LABEL[source],
            show_period_lines=False,
        )
        _force_dark_body(FIG_HTML_DIR / "validacion.html")

        # El PNG se conserva; aquí se actualiza el HTML interactivo.
        print("[ok] figuras/html/validacion.html")

    print("[ok] Hidrogramas calibrados re-generados y combinados.")


if __name__ == "__main__":
    main()

