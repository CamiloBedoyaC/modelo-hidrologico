"""
Calibración del modelo de 2 tanques (Parte 4).

Qué hace este script (en palabras simples):
- Toma los datos diarios de precipitación (Daymet o NLDAS) y el caudal observado (m³/s).
- Ejecuta una calibración del modelo conceptual de dos tanques buscando los parámetros que mejor ajusten el caudal.
- Usa dos métodos de optimización global (DE y Dual Annealing).
- Evalúa desempeño en dos periodos: calibración (1990–1999) y validación (2000–2009).
- Genera tablas y gráficas (PNG y HTML) para comparar resultados por fuente y método.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, dual_annealing

from modelo import TwoTankParams, two_tank_run
from metadata import area_km2, gauge_name
from paths import DATOS_DIR, FIG_DIR, FIG_HTML_DIR
from plotly_theme import apply_dark_theme
from utils import kge, nse

# -----------------------------
# 1) Configuración general
# -----------------------------

# Periodos (según enunciado / decisión del proyecto)
CAL_START = pd.Timestamp("1990-01-01")
CAL_END = pd.Timestamp("1999-12-31")
VAL_START = pd.Timestamp("2000-01-01")
VAL_END = pd.Timestamp("2009-12-31")
WARMUP_DAYS = 365 # Warm-up: se ignoran los primeros 365 días para reducir efecto de condiciones iniciales
XMIN = pd.Timestamp("1980-01-01")# Warm-up: se ignoran los primeros 365 días para reducir efecto de condiciones iniciales
XMAX = pd.Timestamp("2015-01-01")
# Conversión: 1 mm/d sobre 1 km² equivale a 0.011574 m³/s
# (sirve para pasar de caudal en “lámina” mm/d a caudal en m³/s)
MMDAY_TO_CMS_PER_KM2 = 0.0115740741

SOURCE_TO_COL = {"daymet": "prcp_daymet_mm_day", "nldas": "prcp_nldas_mm_day"}
SOURCE_TO_LABEL = {"daymet": "Daymet", "nldas": "NLDAS"}
METHOD_TO_LABEL = {"de": "DE", "da": "Dual Annealing"}

# -----------------------------
# 2) Parámetros del modelo y rangos de búsqueda (Tabla 5.2)
# -----------------------------
# Cada par (min, max) corresponde a un parámetro del modelo:
# ETc, beta, alpha1, D1, k1, alpha2, D2, k2
BOUNDS = [
    (0.0, 10.0),    # ETc: evapotranspiración base (mm/d)
    (0.0, 1.0),     # beta: cuánto aumenta ET con la lluvia (adimensional)
    (0.0, 1.0),     # alpha1: fracción de Pneta que va a escorrentía directa
    (10.0, 500.0),  # D1: capacidad tanque 1 (mm)
    (0.001, 0.99),  # k1: fracción de salida lenta del tanque 1 por día
    (0.0, 1.0),     # alpha2: fracción de salida del tanque 1 que va por vía “rápida”
    (10.0, 1000.0), # D2: capacidad tanque 2 (mm)
    (0.001, 0.99),  # k2: fracción de salida lenta del tanque 2 por día
    ]
PARAM_NAMES = ["ETc", "beta", "alpha1", "D1", "k1", "alpha2", "D2", "k2"]

# -----------------------------
# 3) Funciones auxiliares de lectura / metadatos
# -----------------------------
def _gauge_name(gauge_id: str) -> str:
    """
    Busca el nombre de la estación (gauge) en camels_name.txt.
    Si el archivo no existe o no encuentra el gauge_id, devuelve un nombre genérico.
    """
    return gauge_name(gauge_id)


def _area_km2_for_gauge(gauge_id: str) -> float:
    """
    Lee el área de la cuenca (km²) desde camels_topo.txt.
    Esto se usa para convertir Q_sim de mm/d a m³/s.
    """
    return area_km2(gauge_id)


def _load_merged(gauge_id: str) -> pd.DataFrame:
    """
    Carga el archivo merged de la cuenca, que contiene:
    - precipitación diaria por fuente (Daymet/NLDAS/Maurer)
    - caudal observado en m³/s (q_cms)
    """
    path = DATOS_DIR / gauge_id / f"{gauge_id}_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta preprocesado primero.")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def _dataset_for_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Construye un DataFrame “limpio” con:
    - date
    - prcp_mm_day (columna elegida según la fuente)
    - q_obs_m3s (caudal observado en m³/s)

    También elimina infinitos y filas sin precipitación. Los días sin caudal
    observado se conservan para no romper la continuidad del balance y de los
    estados internos; las métricas ignoran esos valores faltantes.
    """
    col = SOURCE_TO_COL[source]
    out = (
        df[["date", col, "q_cms"]]
        .rename(columns={col: "prcp_mm_day", "q_cms": "q_obs_m3s"})
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["prcp_mm_day"])
        .reset_index(drop=True)
    )
    return out


def _split_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide la serie en:
    - valid: después del warm-up (se usa para revisar balance/consistencia general)
    - cal: periodo de calibración (1990–1999)
    - val: periodo de validación (2000–2009)

    Nota: el warm-up se corta contando desde el primer día disponible en df.
    """
    warmup_cut = df["date"].min() + pd.Timedelta(days=WARMUP_DAYS)
    valid = df[df["date"] >= warmup_cut]
    cal = valid[(valid["date"] >= CAL_START) & (valid["date"] <= CAL_END)]
    val = valid[(valid["date"] >= VAL_START) & (valid["date"] <= VAL_END)]
    return valid, cal, val


def _calibration_inputs(
    df_src: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepara la simulación continua usada por la función objetivo.

    Se simula desde el inicio del registro hasta el final de calibración para
    que los estados de los tanques en 1990 sean exactamente los mismos que en
    :func:`evaluate`. Solo las observaciones de 1990-1999 participan en NSE.
    """
    train = df_src.loc[df_src["date"] <= CAL_END].copy()
    calibration_mask = (
        train["date"].between(CAL_START, CAL_END)
        & train["q_obs_m3s"].notna()
    ).to_numpy(dtype=bool)
    if not calibration_mask.any():
        raise ValueError("No hay observaciones válidas en el periodo de calibración.")
    return (
        train["prcp_mm_day"].to_numpy(dtype=float),
        train["q_obs_m3s"].to_numpy(dtype=float),
        calibration_mask,
    )

# -----------------------------
# 4) Conversión parámetros y simulación
# -----------------------------

def _params_from_vector(x: np.ndarray) -> TwoTankParams:
    """
    Convierte un vector numérico (numpy) en el dataclass TwoTankParams.
    El orden debe coincidir con PARAM_NAMES.
    """
    return TwoTankParams(*[float(v) for v in x])


def _simulate_mm(precip_mm_day: np.ndarray, params: TwoTankParams):
    """
    Corre el modelo con precipitación en mm/d y devuelve:
    - Q_sim (mm/d)
    - ET_usada (mm/d)
    - S1, S2 (mm)
    - dS (mm/d)
    """
    return two_tank_run(precip_mm_day, params, return_states=True)

# -----------------------------
# 5) Función objetivo (lo que optimizan DE/DA)
# -----------------------------

def objective_nse(
    x: np.ndarray,
    precip: np.ndarray,
    q_obs_m3s: np.ndarray,
    coef_cms: float,
    calibration_mask: np.ndarray,
) -> float:
    """
    Función objetivo para calibración estándar:
    - Queremos MAXIMIZAR NSE, pero los optimizadores MINIMIZAN.
    - Por eso devolvemos -NSE.

    Flujo:
    1) Convierto vector x a parámetros del modelo.
    2) Simulo Q en mm/d.
    3) Convierto Q a m³/s con coef_cms.
    4) Calculo NSE solo en las fechas marcadas como calibración.
    """
    params = _params_from_vector(x)
    q_sim_mm_day = two_tank_run(precip, params, return_states=False)
    q_sim_m3s = q_sim_mm_day * coef_cms
    mask = np.asarray(calibration_mask, dtype=bool)
    return -nse(q_obs_m3s[mask], q_sim_m3s[mask])

# -----------------------------
# 6) Calibración por métodos: DE y Dual Annealing
# -----------------------------
def _calibrate_de(
    precip: np.ndarray,
    q_obs_m3s: np.ndarray,
    coef_cms: float,
    calibration_mask: np.ndarray,
    seed: int,
    maxiter: int,
    popsize: int,
) -> TwoTankParams:
    """
    Calibración con Differential Evolution (DE).
    - Explora el espacio de parámetros con una población de soluciones.
    - Al final aplica “polish=True” para refinar la mejor solución.

    workers=1 para mantenerlo reproducible y evitar líos de paralelismo.
    """
    res = differential_evolution(
        objective_nse,
        bounds=BOUNDS,
        args=(precip, q_obs_m3s, coef_cms, calibration_mask),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        updating="deferred",
        workers=1,
    )
    return _params_from_vector(res.x)


def _calibrate_da(
    precip: np.ndarray,
    q_obs_m3s: np.ndarray,
    coef_cms: float,
    calibration_mask: np.ndarray,
    seed: int,
    maxiter: int,
) -> TwoTankParams:
    """
    Calibración con Dual Annealing.
    - Es un método global estilo “recocido simulado” + búsqueda local.
    - no_local_search=False permite que haga un refinamiento local al final.
    """
    res = dual_annealing(
        objective_nse,
        bounds=BOUNDS,
        args=(precip, q_obs_m3s, coef_cms, calibration_mask),
        seed=seed,
        maxiter=maxiter,
        no_local_search=False,
    )
    return _params_from_vector(res.x)

# -----------------------------
# 7) Evaluación completa de un set de parámetros
# -----------------------------
def evaluate(df_src: pd.DataFrame, params: TwoTankParams, area_km2: float) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    """
    Ejecuta el modelo sobre toda la serie df_src y calcula:
    - Series simuladas (Q, ET, estados y residual de balance)
    - Métricas en calibración (NSE, KGE)
    - Métricas en validación (NSE, KGE)
    - Estadísticas de balance hídrico (residual acumulado y error relativo)
    - Indicadores simples de picos (peak_ratio, p95_ratio, mean_ratio) en validación

    Devuelve:
    (df_con_simulacion, metrics_cal, metrics_val, balance_stats, peak_stats)
    """
    # Coeficiente para convertir mm/d a m³/s: área * (m³/s por mm/d en 1 km²)
    coef_cms = area_km2 * MMDAY_TO_CMS_PER_KM2
    # Simulación en “lámina” mm/d y estados
    q_sim_mm_day, et_used, s1, s2, ds = _simulate_mm(df_src["prcp_mm_day"].to_numpy(dtype=float), params)
    # ET “potencial” según fórmula ETc + beta*P (solo para registrar/diagnóstico)
    et_pot = params.ETc + params.beta * df_src["prcp_mm_day"].to_numpy(dtype=float)

    out = df_src.copy()
    out["q_sim_mm_day"] = q_sim_mm_day
    out["q_sim_m3s"] = out["q_sim_mm_day"] * coef_cms
    out["ET_potencial_mm_day"] = et_pot
    out["ET_usada_mm_day"] = et_used
    out["S1_mm"] = s1
    out["S2_mm"] = s2
    out["dS_mm"] = ds
    # Residual de balance (en mm/d): debería ser ~0 (solo error numérico)
    out["balance_residual_mm"] = out["prcp_mm_day"] - out["ET_usada_mm_day"] - out["q_sim_mm_day"] - out["dS_mm"]
    # Cortes temporal (warm-up, cal, val)
    valid_out, cal_out, val_out = _split_periods(out)
    # ---- Métricas hidrológicas (comparando Q_obs vs Q_sim en m³/s)
    def metrics(sub: pd.DataFrame) -> dict:
        return {
            "NSE": float(nse(sub["q_obs_m3s"].to_numpy(dtype=float), sub["q_sim_m3s"].to_numpy(dtype=float))),
            "KGE": float(kge(sub["q_obs_m3s"].to_numpy(dtype=float), sub["q_sim_m3s"].to_numpy(dtype=float))),
        }
    # ---- Resumen de cierre del balance (en mm)
    def balance_stats(sub: pd.DataFrame) -> dict:
        total_p = float(sub["prcp_mm_day"].sum())
        resid = float(sub["balance_residual_mm"].sum())
        return {
            "balance_resid_sum_mm": resid,
            "balance_rel_err": float(resid / total_p) if total_p != 0 else np.nan,
        }
    # ---- Indicadores rápidos de extremos (en validación)
    def peak_stats(sub: pd.DataFrame) -> dict:
        obs = sub["q_obs_m3s"].to_numpy(dtype=float)
        sim = sub["q_sim_m3s"].to_numpy(dtype=float)
        if obs.size == 0 or not np.isfinite(obs).any() or not np.isfinite(sim).any():
            return {"peak_ratio": np.nan, "p95_ratio": np.nan, "mean_ratio": np.nan}
        # peak_ratio: Qmax_sim / Qmax_obs
        peak_ratio = float(np.nanmax(sim) / np.nanmax(obs)) if np.nanmax(obs) != 0 else np.nan
        p95o = float(np.nanquantile(obs, 0.95))
        p95s = float(np.nanquantile(sim, 0.95))
        p95_ratio = float(p95s / p95o) if p95o != 0 else np.nan
        # mean_ratio: media sim / media obs (sesgo global)
        mean_ratio = float(np.nanmean(sim) / np.nanmean(obs)) if np.nanmean(obs) != 0 else np.nan
        return {"peak_ratio": peak_ratio, "p95_ratio": p95_ratio, "mean_ratio": mean_ratio}

    return out, metrics(cal_out), metrics(val_out), balance_stats(valid_out), peak_stats(val_out)

# -----------------------------
# 8) Gráficas: hidrogramas (PNG y HTML)
# -----------------------------
def _plot_hydrograph_png(
    df: pd.DataFrame,
    title: str,
    out_png: Path,
    precip_label: str,
    show_period_lines: bool = True,
) -> None:
    """
    Genera hidrograma en PNG:
    - Arriba: precipitación diaria (invertida, para que “llueva hacia abajo”)
    - Abajo: Q sim y Q obs (m³/s)
    - Líneas verticales opcionales para marcar: inicio cal, inicio val, fin val
    """
    plt.style.use("dark_background")
    fig, (ax_p, ax_q) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.0], "hspace": 0.05},
    )
    # Precipitación (barra) en el panel superior
    ax_p.bar(df["date"], df["prcp_mm_day"], color="#4ea1ff", width=1.0, alpha=0.90, label=f"P {precip_label} (mm/d)")
    ax_p.set_ylabel("Precipitación (mm/d)", color="white", fontsize=12)
    ax_p.invert_yaxis()
    ax_p.grid(color="white", alpha=0.10, linewidth=0.8)
    ax_p.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)
    # Caudales en el panel inferior
    ax_q.plot(df["date"], df["q_sim_m3s"], color="#ff4d5a", linewidth=1.0, label="Q sim (calibrado)")
    ax_q.plot(df["date"], df["q_obs_m3s"], color="#c8c8c8", linewidth=1.0, label="Q observado")
    ax_q.set_ylabel("Caudal (m3/s)", color="white", fontsize=12)
    ax_q.set_xlabel("Fecha", color="white", fontsize=12)
    ax_q.grid(color="white", alpha=0.10, linewidth=0.8)
    ax_q.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)

    if show_period_lines:
        for x, col in [(CAL_START, "#8ac6ff"), (VAL_START, "#ffd166"), (VAL_END, "#ffd166")]:
            ax_q.axvline(x, color=col, linestyle="--", linewidth=0.9, alpha=0.75)
            ax_p.axvline(x, color=col, linestyle="--", linewidth=0.9, alpha=0.35)

    ax_q.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_q.set_xlim(XMIN, XMAX)

    fig.suptitle(title, color="white", fontsize=17, y=0.98)
    for axis in (ax_p, ax_q):
        axis.tick_params(colors="white", labelsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _plot_hydrograph_html(
    df: pd.DataFrame,
    title: str,
    out_html: Path,
    precip_label: str,
    show_period_lines: bool = True,
) -> None:
    """
    Versión interactiva del hidrograma (HTML, Plotly):
    - mismo contenido que el PNG
    - incluye zoom, hover y exportación desde Plotly
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
            y=df["prcp_mm_day"],
            marker=dict(color="#4ea1ff"),
            opacity=0.90,
            name=f"P {precip_label} (mm/d)",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Precipitación (mm/d)", autorange="reversed", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["q_sim_m3s"],
            mode="lines",
            line=dict(color="#ff4d5a", width=1.2),
            name="Q sim (calibrado)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["q_obs_m3s"],
            mode="lines",
            line=dict(color="#c8c8c8", width=1.2),
            name="Q observado",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Caudal (m3/s)", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_xaxes(range=[XMIN, XMAX], row=2, col=1)

    if show_period_lines:
        for x, col in [(CAL_START, "#8ac6ff"), (VAL_START, "#ffd166"), (VAL_END, "#ffd166")]:
            fig.add_vline(x=x, line_width=1, line_dash="dash", line_color=col, opacity=0.75)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.98, yanchor="top")
    )
    apply_dark_theme(fig)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.92,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=140, b=60),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def _hist_params_png(rows: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()
    for axis, param in zip(axes, PARAM_NAMES):
        axis.hist(rows[param].astype(float), bins=10, color="#00bfff", alpha=0.75)
        axis.set_title(param, color="white")
        axis.grid(color="white", alpha=0.10)
        axis.tick_params(colors="white")
    fig.suptitle(title, color="white", y=0.98)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _hist_metrics_png(rows: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(rows["NSE_val"].astype(float), bins=10, color="#38d27a", alpha=0.75)
    axes[0].set_title("NSE validación", color="white")
    axes[0].grid(color="white", alpha=0.10)
    axes[0].tick_params(colors="white")

    axes[1].hist(rows["KGE_val"].astype(float), bins=10, color="#ffd166", alpha=0.75)
    axes[1].set_title("KGE validación", color="white")
    axes[1].grid(color="white", alpha=0.10)
    axes[1].tick_params(colors="white")

    fig.suptitle(title, color="white", y=0.98)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)

# -----------------------------
# 9) Limpieza de salidas anteriores (para no mezclar resultados viejos)
# -----------------------------
def _cleanup_old_outputs() -> None:
    """
    Borra archivos antiguos de calibración estándar (figuras, html y tablas).
    Importante: aquí se intenta NO borrar salidas de calibración por extremos.
    """
    patterns_fig = [
        "calibrado_*.png",
        # Limpiar solo salidas de calibracion estandar para no borrar extremos.
        "hidrograma_calibrado_daymet*.png",
        "hidrograma_calibrado_nldas*.png",
        "hist_params_*.png",
        "hist_metricas_*.png",
        "default_daymet.png",
    ]
    patterns_html = [
        "calibrado_*.html",
        # Limpiar solo salidas de calibracion estandar para no borrar extremos.
        "hidrograma_calibrado_daymet*.html",
        "hidrograma_calibrado_nldas*.html",
    ]
    files_exact = [FIG_DIR / "calibracion.png", FIG_DIR / "validacion.png"]
    data_files = [
        DATOS_DIR / "resultados_calibracion.csv",
        DATOS_DIR / "tabla_source_x_method.csv",
        DATOS_DIR / "tabla_source_x_method.md",
        DATOS_DIR / "resumen_calibracion.json",
    ]

    for pattern in patterns_fig:
        for path in FIG_DIR.glob(pattern):
            path.unlink(missing_ok=True)
    for pattern in patterns_html:
        for path in FIG_HTML_DIR.glob(pattern):
            path.unlink(missing_ok=True)
    for path in files_exact + data_files:
        path.unlink(missing_ok=True)

# -----------------------------
# 10) Utilidades para reconstruir parámetros desde una fila de tabla
# -----------------------------
def _params_from_row(row: pd.Series) -> TwoTankParams:
    """
    Convierte una fila (row) con columnas ETc, beta, alpha1, ... en TwoTankParams.
    Se usa cuando ya tenemos una tabla de “mejores resultados” y queremos graficar.
    """
    return TwoTankParams(
        ETc=float(row["ETc"]),
        beta=float(row["beta"]),
        alpha1=float(row["alpha1"]),
        D1=float(row["D1"]),
        k1=float(row["k1"]),
        alpha2=float(row["alpha2"]),
        D2=float(row["D2"]),
        k2=float(row["k2"]),
    )


def _write_md_table(df_table: pd.DataFrame, out_md: Path) -> None:
    """
    Escribe una tabla Markdown (para pegarla fácil en reporte) con:
    - fuente, método, seed
    - métricas (NSE/KGE cal y val)
    - ratios de picos
    - parámetros calibrados

    También formatea números para que se vean “limpios” (3 decimales métricas, 4 decimales parámetros).
    """
    cols = [
        "source",
        "method",
        "seed",
        "NSE_cal",
        "NSE_val",
        "KGE_cal",
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
    md = df_table[cols].copy()
    for col in ["NSE_cal", "NSE_val", "KGE_cal", "KGE_val", "peak_ratio_val", "p95_ratio_val"]:
        md[col] = md[col].map(lambda value: f"{float(value):.3f}")
    for col in PARAM_NAMES:
        md[col] = md[col].map(lambda value: f"{float(value):.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in md.astype(str).to_numpy().tolist()]
    out_md.write_text("\n".join([header, sep] + rows), encoding="utf-8")

# -----------------------------
# 11) Programa principal
# -----------------------------
def main() -> None:
    """
    Punto de entrada.

    Qué controla el usuario por consola:
    - gauge: id de cuenca
    - sources: daymet/nldas
    - methods: de/da
    - n-seeds: cuántas semillas (corridas) por combinación
    - maxiter, popsize: parámetros del optimizador (más alto = más tiempo)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    parser.add_argument("--sources", nargs="+", default=["daymet", "nldas"], choices=["daymet", "nldas"])
    parser.add_argument("--methods", nargs="+", default=["de", "da"], choices=["de", "da"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=12)
    args = parser.parse_args()

    _cleanup_old_outputs()

    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)

    gauge_name = _gauge_name(args.gauge)
    gauge_label = f"{args.gauge} ({gauge_name})"
    area_km2 = _area_km2_for_gauge(args.gauge)
    coef_cms = area_km2 * MMDAY_TO_CMS_PER_KM2

    df_merged = _load_merged(args.gauge)
    source_frames = {source: _dataset_for_source(df_merged, source) for source in args.sources}

    results: list[dict] = []
    for source in args.sources:
        df_src = source_frames[source]
        precip, q_obs, calibration_mask = _calibration_inputs(df_src)

        for method in args.methods:
            for i in range(args.n_seeds):
                seed = 100 + i
                if method == "de":
                    params = _calibrate_de(
                        precip,
                        q_obs,
                        coef_cms,
                        calibration_mask,
                        seed=seed,
                        maxiter=args.maxiter,
                        popsize=args.popsize,
                    )
                else:
                    params = _calibrate_da(
                        precip,
                        q_obs,
                        coef_cms,
                        calibration_mask,
                        seed=seed,
                        maxiter=args.maxiter,
                    )

                df_eval, met_cal, met_val, bal, peaks = evaluate(df_src, params, area_km2)
                # Guardo un registro plano (una fila) con:
                # - identificación de corrida (gauge, source, method, seed)
                # - parámetros
                # - métricas
                # - balance y picos
                results.append(
                    {
                        "gauge_id": args.gauge,
                        "source": source,
                        "method": method,
                        "seed": seed,
                        **asdict(params),
                        "NSE_cal": met_cal["NSE"],
                        "KGE_cal": met_cal["KGE"],
                        "NSE_val": met_val["NSE"],
                        "KGE_val": met_val["KGE"],
                        **bal,
                        **{f"{k}_val": v for k, v in peaks.items()},
                    }
                )

    df_res = pd.DataFrame(results)
    if df_res.empty:
        raise RuntimeError("No se generaron resultados de calibración.")
    df_res.to_csv(DATOS_DIR / "resultados_calibracion.csv", index=False)

    idx_best_combo = df_res.groupby(["source", "method"])["NSE_cal"].idxmax()
    best_combo = df_res.loc[idx_best_combo].sort_values(["source", "method"]).reset_index(drop=True)
    best_combo.to_csv(DATOS_DIR / "tabla_source_x_method.csv", index=False)
    _write_md_table(best_combo, DATOS_DIR / "tabla_source_x_method.md")

    # -------------------------
    # Gráficas por combinación (source, method) con sus mejores parámetros
    # -------------------------
    for _, row in best_combo.iterrows():
        source = str(row["source"])
        method = str(row["method"])
        params = _params_from_row(row)
        df_eval, met_cal, met_val, _bal, _peaks = evaluate(source_frames[source], params, area_km2)
        title = (
            f"Cuenca {gauge_label}: modelo 2 tanques calibrado "
            f"({source}, método={method}) | NSE cal={met_cal['NSE']:.3f}, NSE val={met_val['NSE']:.3f}"
        )
        _plot_hydrograph_png(
            df_eval,
            title=title,
            out_png=FIG_DIR / f"hidrograma_calibrado_{source}_{method}.png",
            precip_label=SOURCE_TO_LABEL[source],
            show_period_lines=True,
        )
        _plot_hydrograph_html(
            df_eval,
            title=f"{title} | NSE cal={met_cal['NSE']:.3f}, NSE val={met_val['NSE']:.3f}",
            out_html=FIG_HTML_DIR / f"hidrograma_calibrado_{source}_{method}.html",
            precip_label=SOURCE_TO_LABEL[source],
        )
    # -------------------------
    # “Mejor por fuente” según NSE en validación
    # (elige, para cada fuente, el método que mejor generaliza)
    idx_best_source = best_combo.groupby("source")["NSE_val"].idxmax()
    best_source = best_combo.loc[idx_best_source].sort_values("source").reset_index(drop=True)
    for _, row in best_source.iterrows():
        source = str(row["source"])
        method = str(row["method"])
        params = _params_from_row(row)
        df_eval, met_cal, met_val, _bal, _peaks = evaluate(source_frames[source], params, area_km2)
        title = (
            f"Cuenca {gauge_label}: modelo 2 tanques calibrado "
            f"({source}, método={method})"
        )
        _plot_hydrograph_png(
            df_eval,
            title=title,
            out_png=FIG_DIR / f"calibrado_{source}.png",
            precip_label=SOURCE_TO_LABEL[source],
            show_period_lines=True,
        )
        _plot_hydrograph_html(
            df_eval,
            title=f"{title} | NSE cal={met_cal['NSE']:.3f}, NSE val={met_val['NSE']:.3f}",
            out_html=FIG_HTML_DIR / f"calibrado_{source}.html",
            precip_label=SOURCE_TO_LABEL[source],
        )
    # -------------------------
    # “Mejor global” (para hacer figuras resumen calibración vs validación)
    # -------------------------
    overall_best = best_source.sort_values("NSE_val", ascending=False).iloc[0]
    overall_source = str(overall_best["source"])
    overall_method = str(overall_best["method"])
    overall_params = _params_from_row(overall_best)
    df_overall, met_cal_overall, met_val_overall, _bal_overall, _peaks_overall = evaluate(
        source_frames[overall_source], overall_params, area_km2
    )
    df_cal = df_overall[(df_overall["date"] >= CAL_START) & (df_overall["date"] <= CAL_END)].copy()
    df_val = df_overall[(df_overall["date"] >= VAL_START) & (df_overall["date"] <= VAL_END)].copy()

    _plot_hydrograph_png(
        df_cal,
        title=f"Calibración – {overall_source.upper()} ({overall_method.upper()}) | NSE={met_cal_overall['NSE']:.3f}",
        out_png=FIG_DIR / "calibracion.png",
        precip_label=SOURCE_TO_LABEL[overall_source],
        show_period_lines=False,
    )
    _plot_hydrograph_png(
        df_val,
        title=f"Validación – {overall_source.upper()} ({overall_method.upper()}) | NSE={met_val_overall['NSE']:.3f}",
        out_png=FIG_DIR / "validacion.png",
        precip_label=SOURCE_TO_LABEL[overall_source],
        show_period_lines=False,
    )
    # -------------------------
    # Resumen en JSON (para trazabilidad / auditoría)
    # -------------------------
    summary = {
        "gauge_id": args.gauge,
        "area_km2": area_km2,
        "calibration_period": [str(CAL_START.date()), str(CAL_END.date())],
        "validation_period": [str(VAL_START.date()), str(VAL_END.date())],
        "warmup_days": WARMUP_DAYS,
        "sources": args.sources,
        "methods": args.methods,
        "best_source_x_method": best_combo.to_dict(orient="records"),
        "best_by_source": best_source.to_dict(orient="records"),
        "overall_best_for_calibracion_validacion": {
            "source": overall_source,
            "method": overall_method,
            "NSE_cal": float(met_cal_overall["NSE"]),
            "NSE_val": float(met_val_overall["NSE"]),
        },
    }
    (DATOS_DIR / "resumen_calibracion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] {DATOS_DIR / 'resultados_calibracion.csv'}")
    print(f"[ok] {DATOS_DIR / 'tabla_source_x_method.csv'}")
    print(f"[ok] {DATOS_DIR / 'tabla_source_x_method.md'}")
    print(f"[ok] {DATOS_DIR / 'resumen_calibracion.json'}")
    print(f"[ok] {FIG_DIR / 'calibrado_daymet.png'}")
    print(f"[ok] {FIG_DIR / 'calibrado_nldas.png'}")
    print(f"[ok] {FIG_HTML_DIR / 'calibrado_daymet.html'}")
    print(f"[ok] {FIG_HTML_DIR / 'calibrado_nldas.html'}")


if __name__ == "__main__":
    main()
