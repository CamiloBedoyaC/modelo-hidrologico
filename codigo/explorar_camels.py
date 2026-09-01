from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from camels_atributos import attributes_for_gauge, load_attributes
# load_attributes(): carga la tabla completa de atributos CAMELS (671 cuencas).
# attributes_for_gauge(): devuelve los atributos de una cuenca específica (una fila).
from paths import DATOS_DIR, FIG_DIR, FIG_HTML_DIR
from plotly_theme import apply_dark_theme # Función que aplica el tema oscuro de forma consistente en Plotly (fondo negro, letras claras).

# Conversión útil: 1 mm/día sobre 1 km² equivale a 0.011574 m³/s
# (esto se usa para convertir caudal normalizado (mm/d) a caudal volumétrico (m³/s))
MMDAY_TO_CMS_PER_KM2 = 0.0115740741  


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convierte a numérico una lista de columnas.

    Esto es importante porque CAMELS a veces puede venir leído como texto.
    errors="coerce" convierte valores raros a NaN en vez de romper el código.
    """
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def scatter_area_vs_q(
    df: pd.DataFrame,
    color_col: str,
    out_png: Path,
    out_html: Path,
) -> None:
    """
    Hace la gráfica área vs caudal medio (escala log-log), coloreando por algún indicador.

    - x: área de la cuenca (km²)
    - y: caudal medio (m³/s)
    - color: variable escogida (ej: aridity, runoff_ratio, frac_snow, etc.)

    Además ajusta una regresión log-log y la dibuja como línea.
    """
    
    df = df.copy()
    df["area_km2"] = pd.to_numeric(df["area_gages2"], errors="coerce")
     # q_mean en CAMELS suele estar en mm/d (caudal normalizado por área).
    df["q_mean_mmday"] = pd.to_numeric(df["q_mean"], errors="coerce")
    # Convertimos q_mean de mm/d a m³/s usando:
    # q_mean_cms = q_mean_mmday * area_km2 * (mm/d sobre 1 km² a m³/s)
    df["q_mean_cms"] = df["q_mean_mmday"] * df["area_km2"] * MMDAY_TO_CMS_PER_KM2
    df[color_col] = pd.to_numeric(df[color_col], errors="coerce")
    df = df.dropna(subset=["area_km2", "q_mean_cms", color_col])

    # ----------------------------
    # Ajuste log-log (regresión)
    # ----------------------------
    # Para la línea de regresión usamos:
    # log10(Q) = intercept + slope * log10(A)
    x = df["area_km2"].to_numpy()
    y = df["q_mean_cms"].to_numpy()
     # En logaritmos no pueden entrar ceros o negativos, por eso filtramos.
    valid = (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]
    n = int(x.size)# número de puntos válidos para el ajuste
    slope = np.nan
    intercept = np.nan
    r2 = np.nan
     # Un ajuste lineal mínimo requiere al menos 3 puntos para tener sentido.
    if n >= 3:
        lx = np.log10(x)
        ly = np.log10(y)
        slope, intercept = np.polyfit(lx, ly, 1)
        # Calculamos el R² para ver qué tan bien explica la línea los datos.
        yhat = slope * lx + intercept
        denom = np.sum((ly - ly.mean()) ** 2)
        r2 = 1.0 - np.sum((ly - yhat) ** 2) / denom if denom > 0 else np.nan
         # Creamos una línea suave para graficar la regresión en escala real.
        x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y_line = 10 ** (intercept + slope * np.log10(x_line))
    else:
        x_line = np.array([])
        y_line = np.array([])

    # ----------------------------
    # Plotly (HTML interactivo)
    # ----------------------------
    # Scatter interactivo: permite hover y zoom.
    fig = px.scatter(
        df,
        x="area_km2",
        y="q_mean_cms",
        color=color_col,
        color_continuous_scale="Turbo",
        hover_data=["gauge_id", "gauge_name", "frac_snow", "aridity", "runoff_ratio"],
        labels={"area_km2": "Área (km²)", "q_mean_cms": "Caudal medio (m³/s)"},
    )
    # Agregamos la línea de regresión si existe.
    if x_line.size > 0:
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="white", width=2),
                # En la leyenda muestro b (pendiente), R² y n para transparencia del ajuste.
                name=f"Regresión log-log: b={slope:.2f}, R²={r2:.2f}, n={n}",
            )
        )
    # Escalas logarítmicas: es lo típico para relaciones área–caudal.
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(
        title=dict(
            text=f"Indicador: {color_col}",
            y=0.98,
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
        margin=dict(t=140),
    )
    apply_dark_theme(fig)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")

    # ----------------------------
    # Matplotlib (PNG estático)
    # ----------------------------
    # Esto es útil para entregas en PDF o documentos sin interactividad.
    plt.style.use("dark_background")
    figm, ax = plt.subplots(figsize=(9, 6))
    # Scatter con color según la variable elegida.
    sc = ax.scatter(
        df["area_km2"],
        df["q_mean_cms"],
        c=df[color_col],
        s=18,
        cmap="turbo",
        alpha=0.85,
    )
    if x_line.size > 0:
        ax.plot(
            x_line,
            y_line,
            color="white",
            linewidth=2,
            label=f"Regresión log-log: b={slope:.2f}, R²={r2:.2f}, n={n}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Área (km²)")
    ax.set_ylabel("Caudal medio (m³/s)")
    ax.set_title(f"Indicador: {color_col}", pad=22)
    ax.grid(color="w", alpha=0.10)
    if x_line.size > 0:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            facecolor="black",
            framealpha=0.45,
            fontsize=9,
        )
    # Barra de color para interpretar el indicador usado.
    cb = figm.colorbar(sc, ax=ax)
    cb.set_label(color_col)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figm.tight_layout(rect=[0, 0, 1, 0.94])
    figm.savefig(out_png, dpi=200, facecolor="black")
    plt.close(figm)
    
def correlation_heatmap(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    """
    Calcula y grafica la matriz de correlación (heatmap) usando un grupo de variables
    climáticas, topográficas, hidrológicas, de suelo, geología y vegetación.

    Objetivo:
    - ver relaciones globales entre variables
    - identificar colinealidad (variables que se parecen mucho)
    """
    numeric_cols = [
        # topo
        "elev_mean",
        "slope_mean",
        "area_gages2",
        # clim
        "p_mean",
        "pet_mean",
        "p_seasonality",
        "frac_snow",
        "aridity",
        "high_prec_freq",
        "high_prec_dur",
        "low_prec_freq",
        "low_prec_dur",
        # hydro
        "q_mean",
        "runoff_ratio",
        "slope_fdc",
        "baseflow_index",
        "stream_elas",
        "q5",
        "q95",
        "high_q_freq",
        "high_q_dur",
        "low_q_freq",
        "low_q_dur",
        "zero_q_freq",
        # soil
        "soil_depth_pelletier",
        "soil_depth_statsgo",
        "soil_porosity",
        "soil_conductivity",
        "max_water_content",
        "sand_frac",
        "silt_frac",
        "clay_frac",
        "water_frac",
        "organic_frac",
        "other_frac",
        # geol (numéricos)
        "carbonate_rocks_frac",
        "geol_porostiy",
        "geol_permeability",
        # vege
        "frac_forest",
        "lai_max",
        "lai_diff",
        "gvf_max",
        "gvf_diff",
        "root_depth_50",
        "root_depth_99",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    work = _to_numeric(df, available).dropna(subset=available)
    corr = work[available].corr()

    # ----------------------------
    # Plotly (heatmap interactivo)
    # ----------------------------
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0, # centra en 0 para ver mejor correlaciones positivas/negativas
            colorbar=dict(title="r"),
        )
    )
    #fig.update_layout(title="Matriz de correlación")
    fig.update_layout(title=dict(text=f"Matriz de correlación de variables", y=0.98, x=0.5, xanchor="center"))
    apply_dark_theme(fig)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")

    # Matplotlib (PNG)
    plt.style.use("dark_background")
    figm, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Matriz de correlación de variables")
    cbar = figm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("r")
    figm.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figm.savefig(out_png, dpi=220, facecolor="black")
    plt.close(figm)


def export_selected_basin(gauge_id: str, out_csv: Path) -> None:
    """
    Exporta a CSV los atributos de la cuenca elegida.
    Esto es útil para el informe: dejas trazable qué cuenca seleccionaste
    y cuáles son sus atributos principales.
    """
    row = attributes_for_gauge(gauge_id)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, index=False)


def check_selection(gauge_id: str) -> dict[str, object]:
    """
    Verifica automáticamente si la cuenca cumple los criterios del enunciado.

    Criterios usados:
    - Área < 500 km²
    - Fracción de nieve < 10%
    """
    row = attributes_for_gauge(gauge_id)
    area = float(row["area_gages2"])
    frac_snow = float(row["frac_snow"])
    ok_area = area < 500.0
    ok_snow = frac_snow < 0.10
    return {
        "gauge_id": gauge_id,
        "gauge_name": row.get("gauge_name"),
        "area_km2": area,
        "frac_snow": frac_snow,
        "ok_area_lt_500": ok_area,
        "ok_frac_snow_lt_0p10": ok_snow,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    args = parser.parse_args()

    df = load_attributes()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)

    scatter_area_vs_q(
        df,
        "aridity",
        FIG_DIR / "camels_area_vs_q_aridity.png",
        FIG_HTML_DIR / "camels_area_vs_q_aridity.html",
    )
    scatter_area_vs_q(
        df,
        "runoff_ratio",
        FIG_DIR / "camels_area_vs_q_runoff_ratio.png",
        FIG_HTML_DIR / "camels_area_vs_q_runoff_ratio.html",
    )
    scatter_area_vs_q(
        df,
        "frac_snow",
        FIG_DIR / "camels_area_vs_q_frac_snow.png",
        FIG_HTML_DIR / "camels_area_vs_q_frac_snow.html",
    )
    scatter_area_vs_q(
        df,
        "frac_forest",
        FIG_DIR / "camels_area_vs_q_frac_forest.png",
        FIG_HTML_DIR / "camels_area_vs_q_frac_forest.html",
    )
    scatter_area_vs_q(
        df,
        "slope_fdc",
        FIG_DIR / "camels_area_vs_q_slope_fdc.png",
        FIG_HTML_DIR / "camels_area_vs_q_slope_fdc.html",
    )

    correlation_heatmap(
        df,
        FIG_DIR / "camels_correlacion_heatmap.png",
        FIG_HTML_DIR / "camels_correlacion_heatmap.html",
    )

    export_selected_basin(args.gauge, DATOS_DIR / "atributos.csv")
    rep = check_selection(args.gauge)
    print("\nSelección de cuenca (criterios PDF):")
    for k, v in rep.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
