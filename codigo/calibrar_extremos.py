"""
Calibración del modelo 2 tanques priorizando extremos (riesgo).

Idea principal (en simple):
- La calibración “normal” suele optimizar un NSE global, pero eso a veces deja mal
  representados los caudales extremos (crecidas).
- Aquí cambiamos la función objetivo para que el algoritmo le ponga más atención
  a los días de caudal alto (por ejemplo, sobre el percentil 95).

Objetivo multiobjetivo (en el periodo de calibración):
    J = 0.5*(1-NSE_all) + 0.3*(1-NSE_high) + 0.2*PeakBias + 0.2*UnderHigh

Donde:
- NSE_all: NSE global (todos los días).
- NSE_high: NSE solo en días altos (Qobs >= p95 de calibración).
- PeakBias: error relativo en el pico máximo: |Qmax_sim - Qmax_obs| / Qmax_obs.
- UnderHigh: penalización extra cuando el modelo SUBestima en días altos:
             mean(((Qobs-Qsim)/Qobs)^2) para Qobs>=p95 y Qsim<Qobs.

IMPORTANTE:
- Aquí MINIMIZAMOS J (más pequeño = mejor).
- No borra salidas previas: todo se guarda con sufijo `_extremos`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

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
# 1) Configuración de periodos y constantes
# -----------------------------

# Periodos usados para calibrar y validar (igual que la calibración estándar)    
CAL_START = pd.Timestamp("1990-01-01")
CAL_END = pd.Timestamp("1999-12-31")
VAL_START = pd.Timestamp("2000-01-01")
VAL_END = pd.Timestamp("2009-12-31")
WARMUP_DAYS = 365 # Warm-up: quitamos 365 días desde el inicio para que el modelo “se estabilice”
XMIN = pd.Timestamp("1980-01-01")
XMAX = pd.Timestamp("2015-01-01")
MMDAY_TO_CMS_PER_KM2 = 0.0115740741 # Warm-up: quitamos 365 días desde el inicio para que el modelo “se estabilice”
# Qué columna de precipitación usar según la fuente
SOURCE_TO_COL = {"daymet": "prcp_daymet_mm_day", "nldas": "prcp_nldas_mm_day"}
# Etiquetas para los métodos de optimización
SOURCE_TO_LABEL = {"daymet": "Daymet", "nldas": "NLDAS"}
METHOD_TO_LABEL = {"de": "DE", "da": "Dual Annealing"}

# -----------------------------
# 2) Parámetros del modelo: nombres y rangos permitidos
# -----------------------------

# Orden de parámetros (debe coincidir con TwoTankParams y con los bounds)
PARAM_NAMES = ["ETc", "beta", "alpha1", "D1", "k1", "alpha2", "D2", "k2"]
# Rangos de búsqueda (misma idea que calibración estándar)
BOUNDS = [
    (0.0, 10.0),    # ETc
    (0.0, 1.0),     # beta
    (0.0, 1.0),     # alpha1
    (10.0, 500.0),  # D1
    (0.001, 0.99),  # k1
    (0.0, 1.0),     # alpha2
    (10.0, 1000.0), # D2
    (0.001, 0.99),  # k2
]

# -----------------------------
# 3) Pesos del objetivo (importancia relativa)
# -----------------------------
# OJO: si sumas estos pesos dan 1.2, no 1.0.
# Eso NO está “mal”, solo significa que J está escalado. Como igual comparamos J
# entre corridas, lo importante es la proporción entre pesos.
W_NSE_ALL = 0.5
W_NSE_HIGH = 0.3
W_PEAK_BIAS = 0.2
W_UNDER_HIGH = 0.2

# -----------------------------
# 4) Lectura de metadatos y datos
# -----------------------------

def _gauge_name(gauge_id: str) -> str:
    """
    Busca el nombre del gauge (estación) en camels_name.txt.
    Si no existe el archivo o no encuentra el ID, devuelve un texto genérico.
    """
    return gauge_name(gauge_id)


def _area_km2_for_gauge(gauge_id: str) -> float:
    """
    Lee el área de la cuenca desde camels_topo.txt (campo area_gages2).
    Se necesita para convertir Qsim (mm/d) a m³/s.
    """
    return area_km2(gauge_id)


def _load_merged(gauge_id: str) -> pd.DataFrame:
    """
    Carga el archivo merged generado en preprocesado:
    datos/<gauge>/<gauge>_merged.csv

    Ahí ya vienen:
    - precipitación diaria (por fuente)
    - caudal observado (q_cms)
    """
    path = DATOS_DIR / gauge_id / f"{gauge_id}_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta preprocesado primero.")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def _dataset_for_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Extrae un dataset simple para una fuente específica:
    - date
    - prcp_mm_day (precipitación de esa fuente)
    - q_obs_m3s (caudal observado, en m³/s)
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
    Divide en:
    - valid: después del warm-up
    - cal: periodo calibración (1990–1999)
    - val: periodo validación (2000–2009)
    """
    warmup_cut = df["date"].min() + pd.Timedelta(days=WARMUP_DAYS)
    valid = df[df["date"] >= warmup_cut]
    cal = valid[(valid["date"] >= CAL_START) & (valid["date"] <= CAL_END)]
    val = valid[(valid["date"] >= VAL_START) & (valid["date"] <= VAL_END)]
    return valid, cal, val


def _calibration_inputs(
    df_src: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construye el registro continuo y la máscara 1990-1999 del objetivo."""
    train = df_src.loc[df_src["date"] <= CAL_END].copy()
    calibration_mask = (
        train["date"].between(CAL_START, CAL_END)
        & train["q_obs_m3s"].notna()
    ).to_numpy(dtype=bool)
    if not calibration_mask.any():
        raise ValueError("No hay observaciones válidas en el periodo de calibración.")
    q_obs = train["q_obs_m3s"].to_numpy(dtype=float)
    q95 = float(np.nanquantile(q_obs[calibration_mask], 0.95))
    return (
        train["prcp_mm_day"].to_numpy(dtype=float),
        q_obs,
        calibration_mask,
        q95,
    )

# -----------------------------
# 5) Conversión vector -> parámetros
# -----------------------------

def _params_from_vector(x: np.ndarray) -> TwoTankParams:
    """
    Los optimizadores trabajan con un vector x (numpy array).
    Esta función lo convierte al dataclass TwoTankParams en el orden correcto.
    """
    return TwoTankParams(*[float(v) for v in x])

# -----------------------------
# 6) Función objetivo “especial para extremos”
# -----------------------------
def _objective_components(q_obs: np.ndarray, q_sim: np.ndarray, q95: float) -> dict[str, float]:
    """
    Calcula los componentes del objetivo y devuelve un diccionario.

    Entradas:
    - q_obs: caudal observado (m³/s)
    - q_sim: caudal simulado (m³/s)
    - q95: umbral de “alto caudal” (percentil 95 del periodo de calibración)

    Salida:
    - J y también los componentes (NSE_all, NSE_high, PeakBias, UnderHigh)

    Detalles importantes:
    - Si hay pocos días altos (menos de 5), NSE_high se iguala a NSE_all
      para no castigar por falta de datos.
    - Si NSE sale NaN/inf, se fuerza a un valor malo (-5) para que no “engañe” al optimizador.
    """
    finite = np.isfinite(q_obs) & np.isfinite(q_sim)
    q_obs = np.asarray(q_obs, dtype=float)[finite]
    q_sim = np.asarray(q_sim, dtype=float)[finite]
    # Caso extremo: si no hay datos, devolvemos un J enorme para que sea mala solución
    if q_obs.size == 0 or q_sim.size == 0:
        return {
            "J": 1e6,
            "NSE_all": -5.0,
            "NSE_high": -5.0,
            "PeakBias": 10.0,
            "UnderHigh": 10.0,
        }
    # 1) NSE global (todos los días)
    nse_all = float(nse(q_obs, q_sim))
    if not np.isfinite(nse_all):
        nse_all = -5.0
    # 2) NSE en “días altos”: definidos con q_obs >= q95
    high_mask = q_obs >= q95
    if np.count_nonzero(high_mask) >= 5:
        nse_high = float(nse(q_obs[high_mask], q_sim[high_mask]))
        if not np.isfinite(nse_high):
            nse_high = -5.0
    else:
        nse_high = nse_all
    # 3) PeakBias: qué tan bien captura el máximo
    qmax_obs = float(np.nanmax(q_obs)) if np.isfinite(q_obs).any() else np.nan
    qmax_sim = float(np.nanmax(q_sim)) if np.isfinite(q_sim).any() else np.nan
    if not np.isfinite(qmax_obs) or qmax_obs <= 0 or not np.isfinite(qmax_sim):
        peak_bias = 10.0
    else:
        peak_bias = abs(qmax_sim - qmax_obs) / qmax_obs
    # 4) UnderHigh: penaliza solo la subestimación en días altos
    #    Si Qsim<Qobs en esos días, medimos el error relativo al cuadrado y promediamos.
    under_mask = high_mask & (q_sim < q_obs) & (q_obs > 0)
    if np.count_nonzero(under_mask) > 0:
        under_high = float(np.mean(((q_obs[under_mask] - q_sim[under_mask]) / q_obs[under_mask]) ** 2))
    else:
        under_high = 0.0
    # 5) Combinación ponderada (J más bajo = mejor)
    j = (
        W_NSE_ALL * (1.0 - nse_all)
        + W_NSE_HIGH * (1.0 - nse_high)
        + W_PEAK_BIAS * peak_bias
        + W_UNDER_HIGH * under_high
    )
    return {
        "J": float(j),
        "NSE_all": float(nse_all),
        "NSE_high": float(nse_high),
        "PeakBias": float(peak_bias),
        "UnderHigh": float(under_high),
    }


def objective_extremos(
    x: np.ndarray,
    precip: np.ndarray,
    q_obs_m3s: np.ndarray,
    coef_cms: float,
    calibration_mask: np.ndarray,
    q95: float,
) -> float:
    """
    Función objetivo que ve el optimizador (DE o Dual Annealing).

    Flujo:
    1) x -> parámetros
    2) simulo Q en mm/d
    3) convierto a m³/s
    4) calculo J con los componentes de extremos
    5) devuelvo solo J (porque el optimizador necesita un número)
    """
    params = _params_from_vector(x)
    q_sim_mm_day = two_tank_run(precip, params, return_states=False)
    q_sim_m3s = q_sim_mm_day * coef_cms
    mask = np.asarray(calibration_mask, dtype=bool)
    return _objective_components(q_obs_m3s[mask], q_sim_m3s[mask], q95)["J"]

# -----------------------------
# 7) Calibración con DE y Dual Annealing (igual idea que estándar, pero objetivo distinto)
# -----------------------------
def _calibrate_de(
    precip: np.ndarray,
    q_obs_m3s: np.ndarray,
    coef_cms: float,
    calibration_mask: np.ndarray,
    q95: float,
    seed: int,
    maxiter: int,
    popsize: int,
) -> TwoTankParams:
    """
    Differential Evolution minimizando J (extremos).
    """
    res = differential_evolution(
        objective_extremos,
        bounds=BOUNDS,
        args=(precip, q_obs_m3s, coef_cms, calibration_mask, q95),
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
    q95: float,
    seed: int,
    maxiter: int,
) -> TwoTankParams:
    """
    Dual Annealing minimizando J (extremos).
    """
    res = dual_annealing(
        objective_extremos,
        bounds=BOUNDS,
        args=(precip, q_obs_m3s, coef_cms, calibration_mask, q95),
        seed=seed,
        maxiter=maxiter,
        no_local_search=False,
    )
    return _params_from_vector(res.x)

# -----------------------------
# 8) Métricas rápidas de picos (para reportar)
# -----------------------------
def _peak_ratios(q_obs: np.ndarray, q_sim: np.ndarray) -> dict[str, float]:
    """
    Devuelve ratios útiles para ver cómo se comporta el modelo en extremos:
    - peak_ratio: (máximo sim) / (máximo obs)
    - p95_ratio: (p95 sim) / (p95 obs)
    - mean_ratio: (media sim) / (media obs)
    """
    if q_obs.size == 0 or q_sim.size == 0 or not np.isfinite(q_obs).any() or not np.isfinite(q_sim).any():
        return {"peak_ratio": np.nan, "p95_ratio": np.nan, "mean_ratio": np.nan}
    qmax_obs = float(np.nanmax(q_obs))
    qmax_sim = float(np.nanmax(q_sim))
    p95_obs = float(np.nanquantile(q_obs, 0.95))
    p95_sim = float(np.nanquantile(q_sim, 0.95))
    mean_obs = float(np.nanmean(q_obs))
    mean_sim = float(np.nanmean(q_sim))
    return {
        "peak_ratio": float(qmax_sim / qmax_obs) if qmax_obs > 0 else np.nan,
        "p95_ratio": float(p95_sim / p95_obs) if p95_obs > 0 else np.nan,
        "mean_ratio": float(mean_sim / mean_obs) if mean_obs > 0 else np.nan,
    }

# -----------------------------
# 9) Evaluación completa con este enfoque de extremos
# -----------------------------

def evaluate(df_src: pd.DataFrame, params: TwoTankParams, area_km2: float) -> tuple[pd.DataFrame, dict, dict]:
    """
    Corre el modelo sobre toda la serie y calcula:
    - Series simuladas (Q, ET, estados, balance)
    - Métricas clásicas (NSE y KGE en cal y val)
    - Métrica objetivo J (en cal y val) y componentes en altos
    - Ratios de picos en validación
    - Umbrales q95 para cal y val (para diagnóstico)

    Devuelve:
    - out: DataFrame con columnas de simulación
    - metrics: diccionario con métricas y componentes
    - extras: cosas adicionales (q95_cal, q95_val)
    """
    coef_cms = area_km2 * MMDAY_TO_CMS_PER_KM2
    # Simulación completa (incluye estados y dS) en mm/d
    q_sim_mm_day, et_used, s1, s2, ds = two_tank_run(df_src["prcp_mm_day"].to_numpy(dtype=float), params, return_states=True)
    # ET potencial “teórica” (solo para ver qué tan alta queda)
    et_pot = params.ETc + params.beta * df_src["prcp_mm_day"].to_numpy(dtype=float)
    # Armo el DataFrame final con todo lo que se quiera exportar
    out = df_src.copy()
    out["q_sim_mm_day"] = q_sim_mm_day
    out["q_sim_m3s"] = out["q_sim_mm_day"] * coef_cms
    out["ET_potencial_mm_day"] = et_pot
    out["ET_usada_mm_day"] = et_used
    out["S1_mm"] = s1
    out["S2_mm"] = s2
    out["dS_mm"] = ds
    # Residual de balance en mm/d (idealmente cercano a 0)
    out["balance_residual_mm"] = out["prcp_mm_day"] - out["ET_usada_mm_day"] - out["q_sim_mm_day"] - out["dS_mm"]

    valid, cal, val = _split_periods(out)
     # Percentil 95 observado para definir “alto caudal” (en cada periodo, para reportar)
    q95_cal = float(np.nanquantile(cal["q_obs_m3s"].to_numpy(dtype=float), 0.95)) if len(cal) else np.nan
    q95_val = float(np.nanquantile(val["q_obs_m3s"].to_numpy(dtype=float), 0.95)) if len(val) else np.nan
     # Percentil 95 observado para definir “alto caudal” (en cada periodo, para reportar)
    cal_obj = _objective_components(cal["q_obs_m3s"].to_numpy(dtype=float), cal["q_sim_m3s"].to_numpy(dtype=float), q95_cal)
    val_obj = _objective_components(val["q_obs_m3s"].to_numpy(dtype=float), val["q_sim_m3s"].to_numpy(dtype=float), q95_val)
    # También calculamos J sobre toda la serie “válida” (post warm-up) como chequeo general
    valid_obj = _objective_components(valid["q_obs_m3s"].to_numpy(dtype=float), valid["q_sim_m3s"].to_numpy(dtype=float), q95_cal)
    
    # Métricas para tabla final (mezcla de métricas tradicionales + objetivo de extremos)
    metrics = {
        "NSE_cal": float(nse(cal["q_obs_m3s"].to_numpy(dtype=float), cal["q_sim_m3s"].to_numpy(dtype=float))),
        "KGE_cal": float(kge(cal["q_obs_m3s"].to_numpy(dtype=float), cal["q_sim_m3s"].to_numpy(dtype=float))),
        "NSE_val": float(nse(val["q_obs_m3s"].to_numpy(dtype=float), val["q_sim_m3s"].to_numpy(dtype=float))),
        "KGE_val": float(kge(val["q_obs_m3s"].to_numpy(dtype=float), val["q_sim_m3s"].to_numpy(dtype=float))),
        # Objetivo J (más bajo = mejor) y componentes relevantes
        "J_cal": cal_obj["J"],
        "J_val": val_obj["J"],
        "NSE_high_cal": cal_obj["NSE_high"],
        "NSE_high_val": val_obj["NSE_high"],
        "PeakBias_cal": cal_obj["PeakBias"],
        "PeakBias_val": val_obj["PeakBias"],
        "UnderHigh_cal": cal_obj["UnderHigh"],
        "UnderHigh_val": val_obj["UnderHigh"],
        # Balance hídrico acumulado (sobre todo el periodo válido)
        "balance_resid_sum_mm": float(valid["balance_residual_mm"].sum()),
        "balance_rel_err": float(valid["balance_residual_mm"].sum() / valid["prcp_mm_day"].sum()) if valid["prcp_mm_day"].sum() != 0 else np.nan,
        # Ratios de picos en validación (para ver extremos)
        "peak_ratio_val": _peak_ratios(val["q_obs_m3s"].to_numpy(dtype=float), val["q_sim_m3s"].to_numpy(dtype=float))["peak_ratio"],
        "p95_ratio_val": _peak_ratios(val["q_obs_m3s"].to_numpy(dtype=float), val["q_sim_m3s"].to_numpy(dtype=float))["p95_ratio"],
        "mean_ratio_val": _peak_ratios(val["q_obs_m3s"].to_numpy(dtype=float), val["q_sim_m3s"].to_numpy(dtype=float))["mean_ratio"],
        "J_valid": valid_obj["J"],
    }
    return out, metrics, {"q95_cal": q95_cal, "q95_val": q95_val}

# -----------------------------
# 10) Gráficas (hidrogramas) para ver comportamiento en toda la serie
# -----------------------------
def _plot_hydrograph_png(df: pd.DataFrame, title: str, out_png: str, precip_label: str) -> None:
    """
    Hidrograma en PNG con:
    - P arriba (invertida)
    - Qobs y Qsim abajo
    - líneas verticales que marcan inicio/final de periodos
    """
    plt.style.use("dark_background")
    fig, (ax_p, ax_q) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.0], "hspace": 0.05},
    )
    ax_p.bar(df["date"], df["prcp_mm_day"], color="#4ea1ff", width=1.0, alpha=0.9, label=f"P {precip_label} (mm/d)")
    ax_p.set_ylabel("Precipitación (mm/d)", color="white", fontsize=12)
    ax_p.invert_yaxis()
    ax_p.grid(color="white", alpha=0.10)
    ax_p.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)

    ax_q.plot(df["date"], df["q_sim_m3s"], color="#ff4d5a", linewidth=1.0, label="Q sim (calibrado)")
    ax_q.plot(df["date"], df["q_obs_m3s"], color="#c8c8c8", linewidth=1.0, label="Q observado")
    ax_q.set_ylabel("Caudal (m3/s)", color="white", fontsize=12)
    ax_q.set_xlabel("Fecha", color="white", fontsize=12)
    ax_q.grid(color="white", alpha=0.10)
    ax_q.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)
    for x, col in [(CAL_START, "#8ac6ff"), (VAL_START, "#ffd166"), (VAL_END, "#ffd166")]:
        ax_p.axvline(x, color=col, linestyle="--", linewidth=0.9, alpha=0.35)
        ax_q.axvline(x, color=col, linestyle="--", linewidth=0.9, alpha=0.75)
    ax_q.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_q.set_xlim(XMIN, XMAX)

    fig.suptitle(title, color="white", fontsize=17, y=0.98)
    for axis in (ax_p, ax_q):
        axis.tick_params(colors="white", labelsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _plot_hydrograph_html(df: pd.DataFrame, title: str, out_html: str, precip_label: str) -> None:
    """
    Versión interactiva del hidrograma (HTML con Plotly).
    Útil para zoom y hover en eventos extremos.
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
            opacity=0.9,
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
    fig.write_html(out_html, include_plotlyjs="cdn")

# -----------------------------
# 11) Utilidades para reconstruir parámetros desde una fila de resultados
# -----------------------------
def _params_from_row(row: pd.Series) -> TwoTankParams:
    """
    Cuando ya escogimos la “mejor” corrida en una tabla (CSV),
    esta función permite reconstruir TwoTankParams para volver a simular/graficar.
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


def _write_md(df_table: pd.DataFrame) -> None:
    """
    Escribe una tabla Markdown con los mejores resultados por combinación.
    Sirve para pegar directo en el reporte.
    """
    cols = [
        "source", "method", "seed",
        "J_cal", "NSE_cal", "NSE_high_cal", "PeakBias_cal", "UnderHigh_cal",
        "J_val", "NSE_val", "KGE_val", "peak_ratio_val", "p95_ratio_val",
        "ETc", "beta", "alpha1", "D1", "k1", "alpha2", "D2", "k2",
    ]
    md = df_table[cols].copy()
    for col in [
        "J_cal", "NSE_cal", "NSE_high_cal", "PeakBias_cal", "UnderHigh_cal",
        "J_val", "NSE_val", "KGE_val", "peak_ratio_val", "p95_ratio_val",
    ]:
        md[col] = md[col].map(lambda v: f"{float(v):.3f}")
    for col in PARAM_NAMES:
        md[col] = md[col].map(lambda v: f"{float(v):.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(r) + " |" for r in md.astype(str).to_numpy().tolist()]
    (DATOS_DIR / "tabla_source_x_method_extremos.md").write_text("\n".join([header, sep] + rows), encoding="utf-8")

# -----------------------------
# 12) Programa principal
# -----------------------------
def main() -> None:
    """
    Corre la calibración enfocada en extremos.

    Por qué n-seeds y maxiter son bajos por defecto:
    - Calibrar “multiobjetivo” puede ser más lento.
    - Aquí se deja liviano para correr rápido, pero puedes subirlo si tienes tiempo.

    Archivos de salida (todos con sufijo _extremos):
    - resultados_calibracion_extremos.csv
    - tabla_source_x_method_extremos.csv y .md
    - resumen_calibracion_extremos.json
    - figuras calibrado_extremos_daymet/nldas + html
    - hidrograma_calibrado_extremos_<source>_<method>.png/html
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    parser.add_argument("--sources", nargs="+", default=["daymet", "nldas"], choices=["daymet", "nldas"])
    parser.add_argument("--methods", nargs="+", default=["de", "da"], choices=["de", "da"])
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=8)
    args = parser.parse_args()

    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)

    gauge_name = _gauge_name(args.gauge)
    gauge_label = f"{args.gauge} ({gauge_name})"
    area_km2 = _area_km2_for_gauge(args.gauge)
    coef_cms = area_km2 * MMDAY_TO_CMS_PER_KM2
    df_merged = _load_merged(args.gauge)
    src_frames = {s: _dataset_for_source(df_merged, s) for s in args.sources}

    results: list[dict] = []
    for source in args.sources:
        df_src = src_frames[source]
        precip, q_obs, calibration_mask, q95 = _calibration_inputs(df_src)

        for method in args.methods:
            for idx in range(args.n_seeds):
                seed = 100 + idx
                if method == "de":
                    params = _calibrate_de(
                        precip,
                        q_obs,
                        coef_cms,
                        calibration_mask,
                        q95,
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
                        q95,
                        seed=seed,
                        maxiter=args.maxiter,
                    )

                df_eval, metrics, extras = evaluate(df_src, params, area_km2)
                row = {
                    "gauge_id": args.gauge,
                    "source": source,
                    "method": method,
                    "seed": seed,
                    **asdict(params),
                    **metrics,
                    **extras,
                }
                results.append(row)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        raise RuntimeError("No se generaron resultados.")

    df_res.to_csv(DATOS_DIR / "resultados_calibracion_extremos.csv", index=False)

    idx_best_combo = df_res.groupby(["source", "method"])["J_cal"].idxmin()
    best_combo = df_res.loc[idx_best_combo].sort_values(["source", "method"]).reset_index(drop=True)
    best_combo.to_csv(DATOS_DIR / "tabla_source_x_method_extremos.csv", index=False)
    _write_md(best_combo)

    # -------------------------
    # Hidrogramas por combinación (source, method)
    # -------------------------
    for _, row in best_combo.iterrows():
        source = str(row["source"])
        method = str(row["method"])
        params = _params_from_row(row)
        df_eval, metrics, _extras = evaluate(src_frames[source], params, area_km2)
        title = (
            f"Cuenca {gauge_label}: modelo 2 tanques calibrado extremos "
            f"({source}, método={method}) | J={metrics['J_cal']:.3f}, NSE cal={metrics['NSE_cal']:.3f}, NSE val={metrics['NSE_val']:.3f}"
        )
        png_path = FIG_DIR / f"hidrograma_calibrado_extremos_{source}_{method}.png"
        html_path = FIG_HTML_DIR / f"hidrograma_calibrado_extremos_{source}_{method}.html"
        _plot_hydrograph_png(df_eval, title, str(png_path), SOURCE_TO_LABEL[source])
        _plot_hydrograph_html(df_eval, title, str(html_path), SOURCE_TO_LABEL[source])

    # -------------------------
    # Mejor por fuente (Daymet vs NLDAS) usando J_cal mínimo
    # (para una comparación directa de fuentes)
    # -------------------------
    idx_best_source = best_combo.groupby("source")["J_cal"].idxmin()
    best_source = best_combo.loc[idx_best_source].sort_values("source").reset_index(drop=True)
    for _, row in best_source.iterrows():
        source = str(row["source"])
        method = str(row["method"])
        params = _params_from_row(row)
        df_eval, metrics, _extras = evaluate(src_frames[source], params, area_km2)
        title = (
            f"Cuenca {gauge_label}: calibración extremos ({source}, método={method}) "
            f"| J={metrics['J_cal']:.3f}, NSE cal={metrics['NSE_cal']:.3f}, NSE val={metrics['NSE_val']:.3f}"
        )
        _plot_hydrograph_png(df_eval, title, str(FIG_DIR / f"calibrado_extremos_{source}.png"), SOURCE_TO_LABEL[source])
        _plot_hydrograph_html(df_eval, title, str(FIG_HTML_DIR / f"calibrado_extremos_{source}.html"), SOURCE_TO_LABEL[source])

    # -------------------------
    # Resumen JSON: deja trazabilidad del objetivo y “mejores” resultados
    # -------------------------
    summary = {
        "gauge_id": args.gauge,
        "objective": {
            "formula": "J = 0.5*(1-NSE_all) + 0.3*(1-NSE_high) + 0.2*PeakBias + 0.2*UnderHigh",
            "weights": {
                "w_nse_all": W_NSE_ALL,
                "w_nse_high": W_NSE_HIGH,
                "w_peak_bias": W_PEAK_BIAS,
                "w_under_high": W_UNDER_HIGH,
            },
        },
        "best_by_source_method": best_combo.to_dict(orient="records"),
        "best_by_source": best_source.to_dict(orient="records"),
    }
    (DATOS_DIR / "resumen_calibracion_extremos.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] {DATOS_DIR / 'resultados_calibracion_extremos.csv'}")
    print(f"[ok] {DATOS_DIR / 'tabla_source_x_method_extremos.csv'}")
    print(f"[ok] {DATOS_DIR / 'tabla_source_x_method_extremos.md'}")
    print(f"[ok] {DATOS_DIR / 'resumen_calibracion_extremos.json'}")
    print(f"[ok] {FIG_DIR / 'calibrado_extremos_daymet.png'}")
    print(f"[ok] {FIG_DIR / 'calibrado_extremos_nldas.png'}")
    print(f"[ok] {FIG_HTML_DIR / 'calibrado_extremos_daymet.html'}")
    print(f"[ok] {FIG_HTML_DIR / 'calibrado_extremos_nldas.html'}")


if __name__ == "__main__":
    main()
