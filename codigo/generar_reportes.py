from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datos"
FIG_DIR = ROOT / "figuras"
FIG_HTML_DIR = FIG_DIR / "html"
REPORT_DIR = ROOT / "informe"


def parse_key_value_file(path: Path) -> dict[str, str]:
    content: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        content[key.strip()] = value.strip()
    return content


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def format_numeric_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    formatted = dataframe.copy()
    for column_name in formatted.columns:
        series = formatted[column_name]
        if pd.api.types.is_numeric_dtype(series):
            formatted[column_name] = series.map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            )
    return formatted


def build_latex_table(dataframe: pd.DataFrame, caption: str, label: str) -> str:
    table_latex = format_numeric_table(dataframe).to_latex(index=False, escape=True)
    return dedent(
        f"""
        \\begin{{table}}[H]
        \\centering
        \\scriptsize
        \\caption{{{latex_escape(caption)}}}
        \\label{{{label}}}
        \\resizebox{{\\textwidth}}{{!}}{{%
        {table_latex}
        }}
        \\end{{table}}
        """
    ).strip()


def html_table(dataframe: pd.DataFrame) -> str:
    return format_numeric_table(dataframe).to_html(
        index=False, classes="styled-table", border=0
    )


def figure_block_html(
    title: str,
    html_path: str,
    height: int = 600,
    class_name: str | None = None,
) -> str:
    class_attr = f"figure-card {class_name}".strip() if class_name else "figure-card"
    return dedent(
        f"""
        <div class="{class_attr}">
          <h4>{title}</h4>
          <iframe src="{html_path}" loading="lazy" style="height:{height}px;"></iframe>
        </div>
        """
    ).strip()


def repair_mojibake(text: str) -> str:
    repaired = text
    try:
        repaired = repaired.encode("latin-1").decode("utf-8")
    except UnicodeError:
        pass

    replacements = {
        "Ã¡": "á",
        "Ã©": "é",
        "Ã­": "í",
        "Ã³": "ó",
        "Ãº": "ú",
        "Ã±": "ñ",
        "Ã": "Á",
        "Ã‰": "É",
        "Ã": "Í",
        "Ã“": "Ó",
        "Ãš": "Ú",
        "Ã‘": "Ñ",
        "Ã¼": "ü",
        "Ãœ": "Ü",
        "Â²": "²",
        "Â°": "°",
        "Â·": "·",
        "Â ": " ",
    }
    for src, dst in replacements.items():
        repaired = repaired.replace(src, dst)
    return repaired


def generate_reports() -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    attributes = pd.read_csv(DATA_DIR / "atributos.csv", dtype={"gauge_id": str}).iloc[0]
    stats = pd.read_csv(DATA_DIR / "estadisticas_basicas.csv")
    calibration_standard = pd.read_csv(
        DATA_DIR / "tabla_source_x_method.csv", dtype={"gauge_id": str}
    )
    calibration_extremes = pd.read_csv(
        DATA_DIR / "tabla_source_x_method_extremos.csv", dtype={"gauge_id": str}
    )
    balance_daymet = parse_key_value_file(DATA_DIR / "two_tank_balance_summary_daymet.txt")
    balance_nldas = parse_key_value_file(DATA_DIR / "two_tank_balance_summary_nldas.txt")

    best_standard = calibration_standard.sort_values(by="NSE_val", ascending=False).iloc[0]
    best_extremes = calibration_extremes.sort_values(by="J_val", ascending=True).iloc[0]

    comparison_standard = calibration_standard[
        [
            "source",
            "method",
            "NSE_cal",
            "NSE_val",
            "KGE_val",
            "peak_ratio_val",
            "p95_ratio_val",
            "ETc",
            "beta",
            "alpha1",
            "D1",
            "k1",
            "alpha2",
            "D2",
            "k2",
        ]
    ].copy()
    comparison_standard.columns = [
        "Fuente",
        "Metodo",
        "NSE_cal",
        "NSE_val",
        "KGE_val",
        "PeakRatio_val",
        "P95Ratio_val",
        "ETc",
        "beta",
        "alpha1",
        "D1",
        "k1",
        "alpha2",
        "D2",
        "k2",
    ]

    comparison_extremes = calibration_extremes[
        [
            "source",
            "method",
            "J_cal",
            "J_val",
            "NSE_cal",
            "NSE_val",
            "NSE_high_val",
            "PeakBias_val",
            "UnderHigh_val",
            "ETc",
            "beta",
            "alpha1",
            "D1",
            "k1",
            "alpha2",
            "D2",
            "k2",
        ]
    ].copy()
    comparison_extremes.columns = [
        "Fuente",
        "Metodo",
        "J_cal",
        "J_val",
        "NSE_cal",
        "NSE_val",
        "NSE_high_val",
        "PeakBias_val",
        "UnderHigh_val",
        "ETc",
        "beta",
        "alpha1",
        "D1",
        "k1",
        "alpha2",
        "D2",
        "k2",
    ]

    stats_table = stats[
        ["variable", "units", "count", "missing", "mean", "std", "max", "p25", "p75"]
    ].copy()
    stats_table.columns = [
        "Variable",
        "Unidades",
        "N",
        "Missing",
        "Media",
        "DesvStd",
        "Max",
        "P25",
        "P75",
    ]

    balance_summary_table = pd.DataFrame(
        [
            {
                "Fuente": "Daymet",
                "Periodo": f"{balance_daymet.get('period_start', '')} a {balance_daymet.get('period_end', '')}",
                "NSE (Q)": float(balance_daymet.get("nse_m3s", "nan")),
                "KGE (Q)": float(balance_daymet.get("kge_m3s", "nan")),
                "Residual acum. (mm)": float(
                    balance_daymet.get("residual_cumulative_mm", "nan")
                ),
                "Error relativo final": float(
                    balance_daymet.get("relative_error_final", "nan")
                ),
            },
            {
                "Fuente": "NLDAS",
                "Periodo": f"{balance_nldas.get('period_start', '')} a {balance_nldas.get('period_end', '')}",
                "NSE (Q)": float(balance_nldas.get("nse_m3s", "nan")),
                "KGE (Q)": float(balance_nldas.get("kge_m3s", "nan")),
                "Residual acum. (mm)": float(
                    balance_nldas.get("residual_cumulative_mm", "nan")
                ),
                "Error relativo final": float(
                    balance_nldas.get("relative_error_final", "nan")
                ),
            },
        ]
    )

    latex_stats = build_latex_table(
        stats_table,
        "Estadísticas básicas de variables hidrometeorológicas",
        "tab:stats",
    )
    latex_balance = build_latex_table(
        balance_summary_table, "Resumen balance hídrico modelo no calibrado", "tab:balance"
    )
    latex_cal_std = build_latex_table(
        comparison_standard, "Resultados calibración estándar (NSE)", "tab:cal_std"
    )
    latex_cal_ext = build_latex_table(
        comparison_extremes, "Resultados calibración orientada a extremos", "tab:cal_ext"
    )

    html_stats = html_table(stats_table)
    html_balance = html_table(balance_summary_table)
    html_cal_std = html_table(comparison_standard)
    html_cal_ext = html_table(comparison_extremes)

    tex_output_path = REPORT_DIR / "reporte_tarea1.tex"
    html_output_path = REPORT_DIR / "reporte_tarea1_interactivo.html"

    latex_text = dedent(
        f"""
        \\documentclass[12pt]{{article}}
        \\usepackage[utf8]{{inputenc}}
        \\usepackage[T1]{{fontenc}}
        \\usepackage[spanish, es-noquoting]{{babel}}
        \\usepackage{{lmodern}}
        \\usepackage{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{booktabs}}
        \\usepackage{{float}}
        \\usepackage{{xcolor}}
        \\usepackage{{amsmath}}
        \\usepackage{{caption}}
        \\geometry{{margin=2.2cm}}
        \\pagecolor{{black}}
        \\color{{white}}
        \\captionsetup{{labelfont={{color=white}},textfont={{color=white}}}}
        \\setlength{{\\parindent}}{{0pt}}
        \\setlength{{\\parskip}}{{6pt}}

        \\begin{{document}}

        \\begin{{center}}
        {{\\LARGE \\textbf{{Modelo hidrológico lluvia-escorrentía en la cuenca Sopchoppy river, FL}}}}\\\\[0.4cm]
        \\textbf{{Autores:}} Linda Catalina Correa Lozano y Juan Camilo Bedoya Carmona
        \\end{{center}}

        \\section*{{Objetivo}}
        Implementar, diagnosticar y calibrar un modelo conceptual de dos tanques para simular la respuesta lluvia-escorrentía de la cuenca CAMELS 02327100, evaluando desempeño hidrológico global y en extremos.

        \\section*{{Datos}}
        Cuenca seleccionada: \\textbf{{{latex_escape(str(attributes["gauge_id"]))}}} ({latex_escape(str(attributes["gauge_name"]))}).
        Atributos clave: área = {float(attributes["area_gages2"]):.2f} km$^2$, fracción de nieve = {100*float(attributes["frac_snow"]):.3f}\\%, aridity = {float(attributes["aridity"]):.3f}, runoff ratio = {float(attributes["runoff_ratio"]):.3f}, frac\\_forest = {float(attributes["frac_forest"]):.3f}.

        Los datos se tomaron de CAMELS (atributos + series diarias de forzamiento y caudal observado). Las precipitaciones usadas son Daymet y NLDAS (1980-01-01 a 2014-12-31), y Maurer disponible hasta 2008-12-31.

        Mapa interactivo de cuenca y coordenadas (Leaflet): \\texttt{{\\detokenize{{figuras/html/02327100_mapa_leaflet.html}}}}.

        \\section*{{Metodología}}
        \\subsection*{{Parte 1 - Exploración de CAMELS}}
        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.48\\textwidth]{{../figuras/camels_area_vs_q_aridity.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/camels_area_vs_q_frac_forest.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/camels_area_vs_q_runoff_ratio.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/camels_area_vs_q_slope_fdc.png}}
        \\caption{{Dispersión área vs caudal medio (escala log-log) coloreada por distintos indicadores.}}
        \\end{{figure}}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.85\\textwidth]{{../figuras/camels_correlacion_heatmap.png}}
        \\caption{{Matriz de correlación entre variables de clima, suelo, geología e hidrología.}}
        \\end{{figure}}
        """
    ).strip() + "\n"

    latex_text += dedent(
        f"""
        \\subsection*{{Parte 2 - Exploración de datos}}
        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.95\\textwidth]{{../figuras/series_tiempo.png}}
        \\caption{{Serie diaria P-Q con Daymet (P invertida en panel superior).}}
        \\end{{figure}}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.95\\textwidth]{{../figuras/series_tiempo_nldas.png}}
        \\caption{{Serie diaria P-Q con NLDAS (P invertida en panel superior).}}
        \\end{{figure}}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.80\\textwidth]{{../figuras/ciclo_anual.png}}
        \\caption{{Ciclo anual mensual de precipitación y caudal (media y banda P25-P75).}}
        \\end{{figure}}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.90\\textwidth]{{../figuras/comparacion_precip.png}}
        \\caption{{Comparación de precipitación mensual Daymet, NLDAS y Maurer.}}
        \\end{{figure}}

        {latex_stats}

        Resumen breve: Daymet y NLDAS cubren el periodo completo hasta 2014, Maurer termina en 2008. Los picos de caudal son episódicos y altos respecto al caudal base, lo que exige calibraciones robustas para extremos.

        \\subsection*{{Parte 3 - Implementación del modelo}}
        Se implementó el modelo de dos tanques en serie conforme al enunciado, con balance diario:
        \\[
        ET_t = ET_c + \\beta P_t,\\quad
        P_{{neta,t}}=\\max(P_t-ET_t,0),\\quad
        Q_t = Q_{{directa,t}} + Q_{{desborde1,t}}+Q_{{rapido2,t}}+Q_{{desborde2,t}}+Q_{{lento2,t}}
        \\]
        incluyendo estados $S_1, S_2$, desbordes por $D_1, D_2$ y liberaciones por $k_1, k_2$.

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.95\\textwidth]{{../figuras/two_tank_uncalibrated_hydrograph_daymet.png}}
        \\caption{{Modelo no calibrado con Daymet.}}
        \\end{{figure}}

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.95\\textwidth]{{../figuras/two_tank_uncalibrated_hydrograph_nldas.png}}
        \\caption{{Modelo no calibrado con NLDAS.}}
        \\end{{figure}}

        Verificación interactiva de balance hídrico: \\texttt{{\\detokenize{{figuras/html/balance_dashboard_interactivo.html}}}}.
        Incluye: (i) Sankey de entradas/salidas/acumulación, (ii) curva de masa acumulada, (iii) heatmap de residuales mensuales.

        {latex_balance}

        Interpretación del balance: el residual acumulado es numéricamente cercano a cero (orden 10$^{{-13}}$ mm), confirmando cierre del balance en la implementación.
        """
    ).strip() + "\n"

    latex_text += dedent(
        f"""
        \\subsection*{{Parte 4 - Calibración}}
        \\textbf{{Función objetivo base (NSE):}}
        \\[
        NSE = 1 - \\frac{{\\sum_{{t=1}}^n (Q_{{obs,t}}-Q_{{sim,t}})^2}}{{\\sum_{{t=1}}^n (Q_{{obs,t}}-\\overline{{Q_{{obs}}}})^2}}
        \\]
        con calibración global por dos métodos: Differential Evolution (DE) y Dual Annealing (DA), para Daymet y NLDAS.

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_daymet_da.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_daymet_de.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_nldas_da.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_nldas_de.png}}
        \\caption{{Hidrogramas calibrados con objetivo NSE estándar (Daymet/NLDAS; métodos DA/DE).}}
        \\end{{figure}}

        {latex_cal_std}

        Conclusión calibración estándar: el mejor desempeño en validación se logra con Daymet (NSE\\_val $\\approx$ {best_standard["NSE_val"]:.3f}); NLDAS muestra menor capacidad de generalización.

        Luego se ejecutó una calibración adicional orientada a extremos (objetivo multi-criterio con penalización por subestimación de altos caudales), para reducir sesgo en picos.

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_extremos_daymet_da.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_extremos_daymet_de.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_extremos_nldas_da.png}}
        \\includegraphics[width=0.48\\textwidth]{{../figuras/hidrograma_calibrado_extremos_nldas_de.png}}
        \\caption{{Hidrogramas calibrados con objetivo orientado a extremos (Daymet/NLDAS; métodos DA/DE).}}
        \\end{{figure}}

        {latex_cal_ext}

        Conclusión calibración extremos: mejora la representación relativa de picos (peak ratio y métricas de altos caudales), pero reduce NSE global en validación. El mínimo J en validación se obtuvo con {best_extremes["source"]}-{best_extremes["method"]} (J\\_val={best_extremes["J_val"]:.3f}).

        Parámetros óptimos reportados en tablas anteriores para cada fuente y método.

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.95\\textwidth]{{../figuras/validacion.png}}
        \\caption{{Hidrograma de validación (ejemplo de desempeño en periodo de validación).}}
        \\end{{figure}}

        Gráficas interactivas de validación adicional:
        \\texttt{{\\detokenize{{figuras/html/validacion_fdc_obs_vs_sim.html}}}} (FDC obs vs sim) y
        \\texttt{{\\detokenize{{figuras/html/validacion_scatter_obs_vs_sim.html}}}} (dispersión obs vs sim).

        \\section*{{Análisis y discusión}}
        \\textbf{{Interpretación física de parámetros calibrados:}}
        \\begin{{itemize}}
        \\item $ET_c$ alto sugiere pérdidas atmosféricas relevantes incluso en días de baja lluvia.
        \\item $\\beta$ modula la fracción de lluvia que se pierde por ET adicional; valores bajos-medios implican respuesta relativamente eficiente a lluvia efectiva.
        \\item $\\alpha_1$ cercano a 0 indica poca escorrentía instantánea y mayor partición a almacenamiento del tanque superficial.
        \\item $D_1$ y $D_2$ altos representan alta capacidad de almacenamiento (suelo + subsuelo), coherente con respuesta amortiguada en parte de los eventos.
        \\item $k_1$ y $k_2$ controlan vaciado de tanques; combinaciones intermedias generan recesión gradual, pero aún con tendencia a subestimar máximos extremos.
        \\item $\\alpha_2$ cercano a 0 limita flujo rápido subterráneo explícito, lo que puede contribuir a subestimación de picos.
        \\end{{itemize}}

        \\textbf{{Limitaciones del modelo:}} estructura lumped simple, parámetros constantes en todo el periodo, sin representación explícita de no linealidades fuertes de eventos extremos ni variabilidad intra-cuenca.
        \\textbf{{Calibración vs validación:}} Daymet mantiene mejor transferencia (NSE\\_val mayor), mientras NLDAS cae en validación. La calibración de extremos mejora métricas de altos caudales pero sacrifica ajuste global.
        \\textbf{{Mejoras propuestas:}} usar objetivo multiobjetivo con pesos ajustados por gestión de riesgo, incorporar transformación log/Box-Cox de caudal en objetivo, regionalizar o estacionalizar parámetros, y evaluar estructuras con reservorio rápido adicional para picos.

        \\section*{{Conclusión general}}
        El modelo de dos tanques reproduce de forma razonable la dinámica lluvia-escorrentía de la cuenca 02327100, con cierre de balance hídrico robusto. Daymet ofrece mejor desempeño global de validación. Para gestión de riesgo, la calibración orientada a extremos es más útil para aproximar picos, aunque requiere balancear explícitamente la pérdida de ajuste global. La siguiente mejora clave es una calibración multiobjetivo formal con mayor énfasis en caudales altos y posible ajuste estructural del modelo.

        \\end{{document}}
        """
    ).strip() + "\n"

    html_report_text = dedent(
        f"""
        <!doctype html>
        <html lang="es">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Reporte interactivo - Modelo hidrológico Sopchoppy</title>
          <style>
            :root {{
              --bg: #000000;
              --panel: #0e0e0f;
              --panel2: #151518;
              --text: #f5f7fa;
              --muted: #b5bcc9;
              --accent: #44b3ff;
              --accent2: #38d39f;
              --border: #2d3340;
            }}
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 28px 80px; }}
            h1, h2, h3, h4 {{ margin: 0 0 12px; line-height: 1.25; }}
            h1 {{ font-size: 2rem; color: var(--accent); }}
            h2 {{ margin-top: 34px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--accent2); }}
            h3 {{ margin-top: 24px; color: #9fd6ff; }}
            p, li {{ color: var(--text); font-size: 1.03rem; }}
            .muted {{ color: var(--muted); }}
            .card {{ background: linear-gradient(180deg, var(--panel), var(--panel2)); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin: 14px 0; }}
            .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; margin-top: 14px; }}
            .figure-card {{ background: #08090b; border: 1px solid var(--border); border-radius: 10px; padding: 10px; }}
            .figure-wide {{ grid-column: 1 / -1; }}
            .figure-card h4 {{ font-size: 1rem; color: #cfe9ff; margin-bottom: 8px; }}
            iframe {{ width: 100%; border: 1px solid #2a3140; border-radius: 8px; background: #000; }}
            .styled-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.95rem; overflow: hidden; }}
            .styled-table th, .styled-table td {{ border: 1px solid #2a3140; padding: 8px 10px; text-align: center; white-space: nowrap; }}
            .styled-table th {{ background: #1b2230; color: #d2e7ff; }}
            .styled-table tr:nth-child(even) td {{ background: #10131b; }}
            .styled-table tr:nth-child(odd) td {{ background: #0b0f16; }}
            .toc a {{ color: #8dc7ff; text-decoration: none; margin-right: 12px; }}
            .toc a:hover {{ text-decoration: underline; }}
            .pill {{ display: inline-block; padding: 4px 9px; border-radius: 999px; background: #1f2f4c; color: #cde7ff; border: 1px solid #37588a; margin-right: 8px; margin-bottom: 8px; font-size: 0.9rem; }}
            img.static-figure {{ width: 100%; border-radius: 8px; border: 1px solid #2a3140; margin-top: 8px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <h1>Modelo hidrológico lluvia-escorrentía en la cuenca Sopchoppy river, FL</h1>
            <p><strong>Autores:</strong> Linda Catalina Correa Lozano y Juan Camilo Bedoya Carmona</p>
            <div class="card toc">
              <a href="#objetivo">Objetivo</a><a href="#datos">Datos</a><a href="#parte1">Parte 1</a><a href="#parte2">Parte 2</a><a href="#parte3">Parte 3</a><a href="#parte4">Parte 4</a><a href="#analisis">Análisis</a><a href="#conclusion">Conclusión</a>
            </div>

            <h2 id="objetivo">Objetivo</h2>
            <div class="card"><p>Implementar, diagnosticar y calibrar un modelo conceptual de dos tanques para simular la respuesta lluvia-escorrentía de la cuenca CAMELS 02327100, evaluando desempeño hidrológico global y en extremos.</p></div>

            <h2 id="datos">Datos</h2>
            <div class="card">
              <p>
                <span class="pill">Cuenca: {attributes["gauge_id"]}</span>
                <span class="pill">Nombre: {attributes["gauge_name"]}</span>
                <span class="pill">Área: {float(attributes["area_gages2"]):.2f} km²</span>
                <span class="pill">Fracción nieve: {100*float(attributes["frac_snow"]):.3f}%</span>
                <span class="pill">Aridity: {float(attributes["aridity"]):.3f}</span>
                <span class="pill">Runoff ratio: {float(attributes["runoff_ratio"]):.3f}</span>
                <span class="pill">Frac forest: {float(attributes["frac_forest"]):.3f}</span>
              </p>
              <p>Datos CAMELS: atributos de 671 cuencas + series hidrometeorológicas diarias. Para esta cuenca se usaron Daymet y NLDAS (1980-2014) y Maurer (1980-2008).</p>
            </div>
            <div class="figure-grid">{figure_block_html("Mapa Leaflet de la cuenca y coordenadas", "../figuras/html/02327100_mapa_leaflet.html", 670)}</div>

            <h2 id="metodologia">Metodología</h2>
            <h3 id="parte1">Parte 1 - Exploración de CAMELS</h3>
            <div class="figure-grid">
              {figure_block_html("Área vs Q medio (color: aridity)", "../figuras/html/camels_area_vs_q_aridity.html")}
              {figure_block_html("Área vs Q medio (color: frac_forest)", "../figuras/html/camels_area_vs_q_frac_forest.html")}
              {figure_block_html("Área vs Q medio (color: runoff_ratio)", "../figuras/html/camels_area_vs_q_runoff_ratio.html")}
              {figure_block_html("Área vs Q medio (color: slope_fdc)", "../figuras/html/camels_area_vs_q_slope_fdc.html")}
              {figure_block_html("Heatmap de correlación CAMELS", "../figuras/html/camels_correlacion_heatmap.html")}
            </div>

            <h3 id="parte2">Parte 2 - Exploración de datos</h3>
            <div class="figure-grid">
              {figure_block_html("Serie diaria P-Q (Daymet)", "../figuras/html/series_tiempo_P_Q.html", class_name="figure-wide")}
            </div>
            <div class="figure-grid">
              {figure_block_html("Ciclo anual mensual P-Q", "../figuras/html/ciclo_anual_P_Q.html")}
              {figure_block_html("Comparación precipitación (Daymet/NLDAS/Maurer)", "../figuras/html/comparacion_precip.html")}
            </div>
            <div class="card"><h4>Estadísticas básicas</h4>{html_stats}<p class="muted">Resumen: Daymet y NLDAS tienen cobertura completa hasta 2014; Maurer se detiene en 2008. Los picos de caudal son episódicos y altos respecto al caudal base.</p></div>

            <h3 id="parte3">Parte 3 - Implementación del modelo</h3>
            <div class="card"><p>Se implementó el modelo de dos tanques en serie conforme al enunciado, con estados de almacenamiento (S1, S2), desbordes controlados por (D1, D2) y descargas lentas (k1, k2). Se verificó el balance hídrico diario y acumulado.</p></div>
            <div class="figure-grid">
              {figure_block_html("No calibrado - Daymet", "../figuras/html/two_tank_uncalibrated_hydrograph_daymet.html")}
              {figure_block_html("No calibrado - NLDAS", "../figuras/html/two_tank_uncalibrated_hydrograph_nldas.html")}
              {figure_block_html("Dashboard balance hídrico", "../figuras/html/balance_dashboard_interactivo.html", 720, class_name="figure-wide")}
            </div>
            <div class="card"><h4>Resumen de balance hídrico (no calibrado)</h4>{html_balance}<p class="muted">El residual acumulado es del orden de 10^-13 mm (numéricamente cero), lo cual confirma cierre del balance. El Sankey muestra partición entre ET, caudal simulado y cambio de almacenamiento; la curva de masa verifica consistencia acumulada; y el heatmap permite detectar sesgos residuales por mes-año.</p></div>

            <h3 id="parte4">Parte 4 - Calibración</h3>
            <div class="card"><p><strong>Función objetivo base:</strong> NSE, con optimización global mediante DE y DA para Daymet y NLDAS.</p><p class="muted">NSE = 1 - Σ(Qobs - Qsim)² / Σ(Qobs - Qobs̄)²</p></div>
            <div class="figure-grid">
              {figure_block_html("Calibrado estándar - Daymet (DA/DE)", "../figuras/html/hidrograma_calibrado_daymet.html", class_name="figure-wide")}
              {figure_block_html("Calibrado estándar - NLDAS (DA/DE)", "../figuras/html/hidrograma_calibrado_nldas.html", class_name="figure-wide")}
            </div>
            <div class="card"><h4>Resultados calibración estándar</h4>{html_cal_std}<p class="muted">Mejor desempeño global en validación: <strong>{best_standard["source"].upper()}-{best_standard["method"].upper()}</strong> con NSE_val={best_standard["NSE_val"]:.3f}.</p></div>

            <div class="card"><p>Para reducir subestimación de picos se aplicó una calibración adicional orientada a extremos (objetivo multi-criterio con penalización por subestimación en altos caudales).</p></div>
            <div class="figure-grid">
              {figure_block_html("Calibrado extremos - Daymet (DA/DE)", "../figuras/html/hidrograma_calibrado_extremos_daymet.html", class_name="figure-wide")}
              {figure_block_html("Calibrado extremos - NLDAS (DA/DE)", "../figuras/html/hidrograma_calibrado_extremos_nldas.html", class_name="figure-wide")}
            </div>
            <div class="card"><h4>Resultados calibración orientada a extremos</h4>{html_cal_ext}<p class="muted">Menor J en validación: <strong>{best_extremes["source"].upper()}-{best_extremes["method"].upper()}</strong> con J_val={best_extremes["J_val"]:.3f}. Esta calibración mejora métricas de altos caudales, a costa de menor NSE global.</p></div>

            <div class="figure-grid">
              {figure_block_html("Validación (hidrograma interactivo)", "../figuras/html/validacion.html", class_name="figure-wide")}
              {figure_block_html("Validación FDC observada vs simulada", "../figuras/html/validacion_fdc_obs_vs_sim.html")}
              {figure_block_html("Validación dispersión Qobs vs Qsim", "../figuras/html/validacion_scatter_obs_vs_sim.html")}
            </div>
            <div class="card"><p>Las gráficas de validación muestran que Daymet conserva mejor habilidad de generalización. Las curvas FDC revelan diferencias sistemáticas en caudales altos y bajos, mientras la dispersión frente a línea 1:1 evidencia la persistencia de subestimación en picos más extremos.</p></div>

            <h2 id="analisis">Análisis y discusión</h2>
            <div class="card">
              <p><strong>Interpretación física de parámetros calibrados:</strong></p>
              <ul>
                <li><strong>ETc:</strong> pérdida basal por evapotranspiración; valores altos indican fuerte demanda atmosférica.</li>
                <li><strong>beta:</strong> fracción adicional de P que evapora; controla cuánto reduce la lluvia efectiva en eventos húmedos.</li>
                <li><strong>alpha1:</strong> escorrentía directa rápida; valores muy bajos indican predominio de almacenamiento previo al drenaje.</li>
                <li><strong>D1, D2:</strong> capacidades de almacenamiento de suelo y subsuelo; altos valores amortiguan respuesta instantánea.</li>
                <li><strong>k1, k2:</strong> tasas de liberación lenta; regulan recesión y persistencia del caudal.</li>
                <li><strong>alpha2:</strong> fracción rápida desde tanque 2; cercana a cero limita componente de respuesta súbita subterránea.</li>
              </ul>
              <p><strong>Limitaciones:</strong> estructura lumped simple, parámetros constantes en el tiempo, ausencia de no linealidades adicionales para crecidas extremas.</p>
              <p><strong>Calibración vs validación:</strong> la calibración estándar maximiza ajuste global (NSE), mientras la de extremos prioriza picos y reduce sesgo alto, con sacrificio parcial en NSE_val.</p>
              <p><strong>Mejoras propuestas:</strong> calibración multiobjetivo formal con pesos ajustables, objetivo híbrido (NSE + NSE en altos + PeakBias), parámetros estacionales o dependientes de humedad antecedente, y estructura con reservorio rápido adicional para riesgos de inundación.</p>
            </div>

            <h2 id="conclusion">Conclusión general</h2>
            <div class="card"><p>El modelo de dos tanques implementado para la cuenca Sopchoppy logra cierre robusto del balance hídrico y reproduce razonablemente la dinámica lluvia-escorrentía. Daymet ofrece mejor desempeño global de validación. Para aplicaciones de gestión de riesgos, la calibración orientada a extremos es útil para mejorar representación de crecidas, pero debe combinarse con estrategias multiobjetivo y posibles mejoras estructurales del modelo para evitar pérdida excesiva de ajuste global.</p></div>
          </div>
        </body>
        </html>
        """
    ).strip() + "\n"

    latex_text = repair_mojibake(latex_text)
    html_report_text = repair_mojibake(html_report_text)

    tex_output_path.write_text(latex_text, encoding="utf-8")
    html_output_path.write_text(html_report_text, encoding="utf-8")

    return tex_output_path, html_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generador historico de la primera version del informe. Por seguridad no "
            "sobrescribe los informes curados salvo autorizacion explicita."
        )
    )
    parser.add_argument(
        "--legacy-overwrite",
        action="store_true",
        help="Sobrescribe HTML/TeX con la plantilla historica (no recomendado).",
    )
    args = parser.parse_args()
    if not args.legacy_overwrite:
        print("[safe] No se modifico ningun informe.")
        print("Los archivos informe/reporte_tarea1_interactivo.html y .tex son las fuentes curadas.")
        print("Para reconstruir salidas usa codigo/generar_reporte_standalone.py y codigo/compilar_pdf.py.")
    else:
        tex_path, html_path = generate_reports()
        print(f"Generado LaTeX: {tex_path}")
        print(f"Generado HTML: {html_path}")
