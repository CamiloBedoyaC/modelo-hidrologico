from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metadata import gauge_name
from paths import DATOS_DIR, FIG_DIR, FIG_HTML_DIR
from plotly_theme import apply_dark_theme

# ----------------------------
# Configuración general
# ----------------------------

MONTH_LABELS_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
XMIN = pd.Timestamp("1980-01-01")
XMAX = pd.Timestamp("2015-01-01")
# Rango fijo del eje X para mantener comparabilidad en todas las gráficas.
# Se usa 2015-01-01 como “límite superior” para cubrir 2014 completo.

def _load() -> pd.DataFrame:
    """
    Carga los datos ya preprocesados (forzamiento y caudal) y los une por fecha.

    - forzamiento.csv contiene precipitación diaria de varias fuentes.
    - caudal.csv contiene caudal observado en varias unidades.
    """
    forcing = pd.read_csv(DATOS_DIR / "forzamiento.csv", parse_dates=["date"])
    flow = pd.read_csv(DATOS_DIR / "caudal.csv", parse_dates=["date"])
    df = forcing.merge(flow, on="date", how="inner").sort_values("date")
    return df


def _gauge_name(gauge_id: str) -> str:
    """
    Busca el nombre de la cuenca en camels_name.txt para mostrarlo en títulos/etiquetas.

    Si no se encuentra el archivo o el gauge_id, devuelve un texto genérico.
    """
    return gauge_name(gauge_id)


def _basic_stats(df: pd.DataFrame, out_csv: Path) -> None:
    """
    Calcula estadísticas básicas de variables clave, en el mismo formato del reporte.

    La idea es dejar una tabla compacta con:
    - cantidad de datos
    - datos faltantes
    - media, desviación, máximo
    - percentiles 25 y 75 (para describir variabilidad)
    """
    
    # Definimos qué columnas queremos resumir y cómo se llamarán en la tabla.
    # units se guarda explícito para que quede claro en el informe.
    specs = [
        ("prcp_daymet_mm_day", "prcp_daymet_mm_day", "mm/d"),
        ("prcp_nldas_mm_day", "prcp_nldas_mm_day", "mm/d"),
        ("prcp_maurer_mm_day", "prcp_maurer_mm_day", "mm/d"),
        ("q_cms", "q_m3s", "m3/s"),
    ]

    rows = []
    n = len(df) # total de filas (días) del DataFrame
    for col, var_name, units in specs:
        # Convertimos a numérico por seguridad; valores raros se vuelven NaN.
        s = pd.to_numeric(df[col], errors="coerce")
        # Contamos cuántos datos válidos hay y cuántos faltan.
        cnt = int(s.notna().sum())
        miss = int(n - cnt)
        rows.append(
            {
                "variable": var_name,
                "units": units,
                "count": cnt,
                "missing": miss,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "max": float(s.max()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
            }
        )
     # Convertimos lista de diccionarios en una tabla final y la exportamos.
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def _daily_hydro_png(
    df: pd.DataFrame,
    precip_col: str,
    precip_legend: str,
    gauge_label: str,
    out_png: Path,
) -> None:
    """
    Grafica hidrológica diaria en PNG:
    - Arriba: precipitación (invertida, como en hidrología: “llueve hacia abajo”).
    - Abajo: caudal observado.

    Esto permite ver respuesta de caudal frente a eventos de lluvia.
    """
    plt.style.use("dark_background")
    # Dos paneles: precipitación arriba (más pequeño), caudal abajo (más grande).
    fig, (ax_p, ax_q) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.0], "hspace": 0.05},
    )

    ax_p.bar(df["date"], df[precip_col], color="#4ea1ff", width=1.0, alpha=0.92, label=precip_legend)
    ax_p.set_ylabel("Precipitación (mm/d)", color="white", fontsize=12)
    # Invertimos el eje para que las barras crezcan hacia abajo.
    ax_p.invert_yaxis()
    ax_p.grid(color="white", alpha=0.10, linewidth=0.8)
    ax_p.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)

    ax_q.plot(df["date"], df["q_cms"], color="#ff4d5a", linewidth=1.0, label="Caudal observado (m3/s)")
    ax_q.set_ylabel("Caudal (m³/s)", color="white", fontsize=12)
    ax_q.set_xlabel("Fecha", color="white", fontsize=12)
    ax_q.grid(color="white", alpha=0.10, linewidth=0.8)
    ax_q.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)

    # Formato de fechas: ticks cada 5 años para que no quede ilegible.
    ax_q.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_q.set_xlim(XMIN, XMAX)

    fig.suptitle(f"Serie diaria P–Q | Daymet", color="white", fontsize=18, y=0.98)

    for ax in (ax_p, ax_q):
        ax.tick_params(colors="white", labelsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _daily_hydro_html(
    df: pd.DataFrame,
    precip_col: str,
    precip_legend: str,
    gauge_label: str,
    out_html: Path,
) -> None:
    """
    Versión interactiva (HTML) de la gráfica diaria P–Q.

    Misma idea que el PNG, pero con zoom, hover y exportación interactiva.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.65],
    )

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df[precip_col],
            marker=dict(color="#4ea1ff"),
            opacity=0.90,
            name=precip_legend,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Precipitación (mm/d)", autorange="reversed", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["q_cms"],
            mode="lines",
            line=dict(color="#ff4d5a", width=1.2),
            name="Caudal observado (m3/s)",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Caudal (m3/s)", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"Serie diaria P–Q | Daymet",
            y=0.98,
            yanchor="top",
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.93,
            xanchor="left",
            x=0.0,
            font=dict(size=12),
        ),
        margin=dict(t=170),
    )
    fig.update_xaxes(range=[XMIN, XMAX], row=2, col=1)
    apply_dark_theme(fig)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def _monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte series diarias a series mensuales.

    - Precipitación mensual: se suma (mm/mes)
    - Caudal mensual: se promedia (m³/s)

    Esto permite comparar fuentes y mirar estacionalidad de forma más clara.
    """
    mon = df.set_index("date")[["prcp_daymet_mm_day", "prcp_maurer_mm_day", "prcp_nldas_mm_day", "q_cms"]].copy()
    # Creamos un DataFrame mensual con fechas tipo “inicio de mes” (MS = month start).
    out = pd.DataFrame(index=mon.resample("MS").sum().index)
    # Sumamos precipitación diaria para obtener precipitación acumulada mensual.
    # min_count=1 evita que un mes entero quede 0 si todo es NaN.
    out["P_daymet_mm_mon"] = mon["prcp_daymet_mm_day"].resample("MS").sum(min_count=1)
    out["P_maurer_mm_mon"] = mon["prcp_maurer_mm_day"].resample("MS").sum(min_count=1)
    out["P_nldas_mm_mon"] = mon["prcp_nldas_mm_day"].resample("MS").sum(min_count=1)
    # Caudal mensual como promedio (porque está en m³/s, no se “suma”).
    out["Q_cms_mon_mean"] = mon["q_cms"].resample("MS").mean()
    return out.reset_index().rename(columns={"index": "date"})


def _annual_cycle(df: pd.DataFrame, precip_mon_col: str) -> pd.DataFrame:
    """
    Calcula ciclo anual mensual:
    - Para cada mes (1..12) calcula media, p25 y p75 de precipitación y caudal.

    Esto sirve para ver estacionalidad y variabilidad interanual.
    """
    tmp = df[["date", precip_mon_col, "Q_cms_mon_mean"]].dropna().copy()
    tmp["month"] = tmp["date"].dt.month
    grp = tmp.groupby("month", as_index=False)
     # Agregación mensual: promedio y banda intercuartílica (p25–p75)
    out = grp.agg(
        P_mean=(precip_mon_col, "mean"),
        P_p25=(precip_mon_col, lambda s: s.quantile(0.25)),
        P_p75=(precip_mon_col, lambda s: s.quantile(0.75)),
        Q_mean=("Q_cms_mon_mean", "mean"),
        Q_p25=("Q_cms_mon_mean", lambda s: s.quantile(0.25)),
        Q_p75=("Q_cms_mon_mean", lambda s: s.quantile(0.75)),
    )
    out["month_name"] = out["month"].map(lambda m: MONTH_LABELS_ES[m - 1])
    return out


def _annual_cycle_png(cyc: pd.DataFrame, gauge_label: str, out_png: Path) -> None:
    """
    Grafica el ciclo anual en PNG con:
    - Precipitación arriba invertida (mm/mes)
    - Caudal abajo (m³/s)
    - Línea de media + banda p25–p75 en sombreado
    """
    plt.style.use("dark_background")
    fig, (ax_p, ax_q) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.0], "hspace": 0.05},
    )

    x = cyc["month"].to_numpy()

    p_fill = ax_p.fill_between(x, cyc["P_p25"], cyc["P_p75"], color="#7aa2ff", alpha=0.25, label="Precipitación p25-p75")
    p_line = ax_p.plot(x, cyc["P_mean"], color="#4ea1ff", marker="o", linewidth=2.0, markersize=4, label="Precipitación media")[0]
    ax_p.invert_yaxis()
    ax_p.set_ylabel("Precipitación (mm/mes)", color="white", fontsize=12)
    ax_p.grid(color="white", alpha=0.10, linewidth=0.8)

    q_fill = ax_q.fill_between(x, cyc["Q_p25"], cyc["Q_p75"], color="#ff7f8a", alpha=0.25, label="Caudal p25-p75")
    q_line = ax_q.plot(x, cyc["Q_mean"], color="#ff4d5a", marker="o", linewidth=2.0, markersize=4, label="Caudal medio")[0]
    ax_q.set_ylabel("Caudal medio (m³/s)", color="white", fontsize=12)
    ax_q.set_xlabel("Mes", color="white", fontsize=12)
    ax_q.grid(color="white", alpha=0.10, linewidth=0.8)

    ax_q.set_xticks(np.arange(1, 13))
    ax_q.set_xticklabels(MONTH_LABELS_ES)

    handles = [q_line, q_fill, p_line, p_fill]
    labels = ["Caudal medio", "Caudal p25-p75", "Precipitación media", "Precipitación p25-p75"]
    ax_p.legend(handles, labels, loc="upper left", facecolor="black", framealpha=0.45, fontsize=10, ncol=2)

    fig.suptitle(f"Ciclo anual mensual P–Q", color="white", fontsize=18, y=0.98)
    for ax in (ax_p, ax_q):
        ax.tick_params(colors="white", labelsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _annual_cycle_html(cyc: pd.DataFrame, gauge_label: str, out_html: Path) -> None:
    """
    Versión interactiva (HTML) del ciclo anual mensual:
    - incluye sombreado p25–p75
    - incluye media con marcadores
    - precipitación arriba invertida, caudal abajo
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.65],
    )

    x = cyc["month"].to_numpy()
    p_band_x = np.r_[x, x[::-1]]

    fig.add_trace(
        go.Scatter(
            x=p_band_x,
            y=np.r_[cyc["P_p75"].to_numpy(), cyc["P_p25"].to_numpy()[::-1]],
            fill="toself",
            fillcolor="rgba(122,162,255,0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Precipitación p25-p75",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cyc["P_mean"],
            mode="lines+markers",
            line=dict(color="#4ea1ff", width=2.2),
            marker=dict(size=6),
            name="Precipitación media",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Precipitación (mm/mes)", autorange="reversed", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=p_band_x,
            y=np.r_[cyc["Q_p75"].to_numpy(), cyc["Q_p25"].to_numpy()[::-1]],
            fill="toself",
            fillcolor="rgba(255,127,138,0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Caudal p25-p75",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cyc["Q_mean"],
            mode="lines+markers",
            line=dict(color="#ff4d5a", width=2.2),
            marker=dict(size=6),
            name="Caudal medio",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Caudal medio (m³/s)", row=2, col=1)
    fig.update_xaxes(
        title_text="Mes",
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=MONTH_LABELS_ES,
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"Ciclo anual mensual P–Q",
            y=0.98,
            yanchor="top",
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.93,
            xanchor="left",
            x=0.0,
            font=dict(size=12),
        ),
        margin=dict(t=170),
    )
    apply_dark_theme(fig)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def _single_source_monthly_png(dfm: pd.DataFrame, col: str, label: str, color: str, gauge_label: str, out_png: Path) -> None:
    """
    Grafica precipitación mensual de UNA fuente (PNG).

    Esto sirve para ver cómo cambia la precipitación mes a mes,
    sin mezclar fuentes en la misma figura.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(dfm["date"], dfm[col], color=color, linewidth=1.3, label=label)
    ax.set_ylabel("Precipitación mensual (mm/mes)", color="white", fontsize=12)
    ax.set_xlabel("Fecha", color="white", fontsize=12)
    ax.grid(color="white", alpha=0.10)
    ax.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(XMIN, XMAX)
    ax.set_title(f"Cuenca {gauge_label}: Precipitación mensual ({label})", color="white", fontsize=16)
    ax.tick_params(colors="white", labelsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _compare_monthly_png(dfm: pd.DataFrame, gauge_label: str, out_png: Path) -> None:
    """
    Grafica comparación de precipitación mensual de las 3 fuentes en un solo PNG.

    Aquí se busca ver:
    - si las fuentes coinciden en años húmedos/secos
    - si alguna fuente es sistemáticamente más alta o más baja
    - si hay diferencias en picos mensuales
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.plot(dfm["date"], dfm["P_daymet_mm_mon"], color="#00bfff", linewidth=1.3, label="Daymet")
    ax.plot(dfm["date"], dfm["P_nldas_mm_mon"], color="#8a5cff", linewidth=1.3, label="NLDAS")
    ax.plot(dfm["date"], dfm["P_maurer_mm_mon"], color="#38d27a", linewidth=1.3, label="Maurer")
    ax.set_ylabel("Precipitación mensual (mm/mes)", color="white", fontsize=12)
    ax.set_xlabel("Fecha", color="white", fontsize=12)
    ax.grid(color="white", alpha=0.10)
    ax.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10, ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(XMIN, XMAX)
    ax.set_title(f"Comparación precipitación mensual (Daymet | NLDAS | Maurer)", color="white", fontsize=17)
    ax.tick_params(colors="white", labelsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _compare_monthly_html(dfm: pd.DataFrame, gauge_label: str, out_html: Path) -> None:
    """
    Versión interactiva (HTML) de comparación mensual de precipitación.

    Es útil porque permite hacer zoom en periodos específicos y comparar
    visualmente diferencias en eventos.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfm["date"], y=dfm["P_daymet_mm_mon"], mode="lines", line=dict(color="#00bfff", width=1.5), name="Daymet"))
    fig.add_trace(go.Scatter(x=dfm["date"], y=dfm["P_nldas_mm_mon"], mode="lines", line=dict(color="#8a5cff", width=1.5), name="NLDAS"))
    fig.add_trace(go.Scatter(x=dfm["date"], y=dfm["P_maurer_mm_mon"], mode="lines", line=dict(color="#38d27a", width=1.5), name="Maurer"))
    fig.update_layout(
        title=dict(
            text=f"Comparación precipitación mensual (Daymet | NLDAS | Maurer)",
            y=0.98,
            yanchor="top",
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.93,
            xanchor="left",
            x=0.0,
            font=dict(size=12),
        ),
        margin=dict(t=170),
    )
    fig.update_xaxes(title_text="Fecha")
    fig.update_xaxes(range=[XMIN, XMAX])
    fig.update_yaxes(title_text="Precipitación mensual (mm/mes)")
    apply_dark_theme(fig)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def _cleanup_old_outputs() -> None:
    """
    Borra archivos viejos que podrían quedar de versiones anteriores del script,
    para evitar confusión en la carpeta de figuras.
    """
    old_files = [
        FIG_DIR / "series_tiempo_P_Q.png",
        FIG_DIR / "comparacion_precip_mensual.png",
        FIG_HTML_DIR / "comparacion_precip_mensual.html",
    ]
    for p in old_files:
        if p.exists():
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    parser.add_argument(
        "--precip",
        default="prcp_daymet_mm_day",
        choices=["prcp_daymet_mm_day", "prcp_maurer_mm_day", "prcp_nldas_mm_day"],
    )
    args = parser.parse_args()

    df = _load()
    gauge_label = f"{args.gauge} ({_gauge_name(args.gauge)})"
    precip_legend = {
        "prcp_daymet_mm_day": "Precipitacion Daymet (mm/d)",
        "prcp_nldas_mm_day": "Precipitacion NLDAS (mm/d)",
        "prcp_maurer_mm_day": "Precipitacion Maurer (mm/d)",
    }[args.precip]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)

    # (b) Estadisticas basicas
    _basic_stats(df, DATOS_DIR / "estadisticas_basicas.csv")

    # (a) Serie diaria hidrologica (P arriba invertida, Q abajo)
    _daily_hydro_png(df, args.precip, precip_legend, gauge_label, FIG_DIR / "series_tiempo.png")
    _daily_hydro_html(df, args.precip, precip_legend, gauge_label, FIG_HTML_DIR / "series_tiempo_P_Q.html")

    # Grafica adicional solicitada: serie diaria con NLDAS (sin tocar las salidas actuales).
    _daily_hydro_png(
        df,
        "prcp_nldas_mm_day",
        "Precipitacion NLDAS (mm/d)",
        gauge_label,
        FIG_DIR / "series_tiempo_nldas.png",
    )
    _daily_hydro_html(
        df,
        "prcp_nldas_mm_day",
        "Precipitacion NLDAS (mm/d)",
        gauge_label,
        FIG_HTML_DIR / "series_tiempo_P_Q_nldas.html",
    )

    # (b) Ciclo anual mensual con media + banda p25-p75
    dfm = _monthly_aggregates(df)
    precip_mon_col = {
        "prcp_daymet_mm_day": "P_daymet_mm_mon",
        "prcp_maurer_mm_day": "P_maurer_mm_mon",
        "prcp_nldas_mm_day": "P_nldas_mm_mon",
    }[args.precip]
    cyc = _annual_cycle(dfm, precip_mon_col)
    _annual_cycle_png(cyc, gauge_label, FIG_DIR / "ciclo_anual.png")
    _annual_cycle_html(cyc, gauge_label, FIG_HTML_DIR / "ciclo_anual_P_Q.html")

    # (c) Series mensuales por fuente + comparacion unica
    _single_source_monthly_png(dfm, "P_daymet_mm_mon", "Daymet", "#00bfff", gauge_label, FIG_DIR / "precip_mensual_daymet.png")
    _single_source_monthly_png(dfm, "P_maurer_mm_mon", "Maurer", "#ff00ff", gauge_label, FIG_DIR / "precip_mensual_maurer.png")
    _single_source_monthly_png(dfm, "P_nldas_mm_mon", "NLDAS", "#ffd700", gauge_label, FIG_DIR / "precip_mensual_nldas.png")

    _compare_monthly_png(dfm, gauge_label, FIG_DIR / "comparacion_precip.png")
    _compare_monthly_html(dfm, gauge_label, FIG_HTML_DIR / "comparacion_precip.html")

    # Evita dejar graficas viejas no solicitadas
    _cleanup_old_outputs()

    print("[ok] Parte 2 lista: series_tiempo, ciclo_anual, comparacion_precip y estadisticas_basicas.")


if __name__ == "__main__":
    main()
