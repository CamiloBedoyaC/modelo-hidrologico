from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modelo import default_params, two_tank_run
from metadata import area_km2, gauge_name
from paths import DATOS_DIR, FIG_DIR, FIG_HTML_DIR
from plotly_theme import apply_dark_theme
from utils import kge, nse

# Conversión: 1 mm/d sobre 1 km2 equivale aproximadamente a 0.011574 m3/s.
# Esto sirve para convertir caudal en lámina (mm/d) a caudal volumétrico (m3/s).
MMDAY_TO_CMS_PER_KM2 = 0.0115740741
# Límites de tiempo para que todas las figuras queden con el mismo rango.
XMIN = pd.Timestamp("1980-01-01")
XMAX = pd.Timestamp("2015-01-01")
# Mapea cada “source” a la columna correcta de precipitación en forzamiento.csv
SOURCE_TO_COL = {
    "daymet": "prcp_daymet_mm_day",
    "nldas": "prcp_nldas_mm_day",
    "maurer": "prcp_maurer_mm_day",
}
SOURCE_TO_LABEL = {"daymet": "Daymet", "nldas": "NLDAS", "maurer": "Maurer"}


def _gauge_name(gauge_id: str) -> str:
    """
    Busca el nombre del gauge (estación USGS) desde camels_name.txt.
    Si no existe el archivo o el gauge no aparece, retorna un nombre genérico.
    """
    return gauge_name(gauge_id)


def _area_km2_for_gauge(gauge_id: str, df: pd.DataFrame) -> float:
    """
    Obtiene el área de la cuenca (km²) para convertir Q simulado de mm/d a m³/s.

    Plan A (preferido):
    - Leer camels_topo.txt y tomar area_gages2, que es el área oficial de CAMELS.

    Plan B (respaldo):
    - Si no está camels_topo.txt, se estima el área “al revés” usando datos observados:
      coef = mediana( q_cms / q_obs_mm_day )
      Como q_cms ≈ q_mmday * area_km2 * 0.011574,
      entonces area_km2 ≈ coef / 0.011574.
    """
    try:
        return area_km2(gauge_id)
    except FileNotFoundError:
        pass

    valid = df[["q_cms", "q_obs_mm_day"]].dropna()
    valid = valid[valid["q_obs_mm_day"] > 0]
    if valid.empty:
        raise ValueError("No se pudo estimar area de cuenca para convertir Qsim de mm/d a m3/s.")
    coef = float((valid["q_cms"] / valid["q_obs_mm_day"]).median())
    return coef / MMDAY_TO_CMS_PER_KM2


def _load(precip_col: str) -> pd.DataFrame:
    """
    Carga datos base (forzamiento y caudal) y deja una tabla lista para correr el modelo.

    - Lee forzamiento.csv (precipitaciones por fuente)
    - Lee caudal.csv (q_cms, q_obs_mm_day, etc.)
    - Los une por fecha.
    - Q puede tener NaNs (faltantes), pero P no: aquí se eliminan filas donde P falte
      porque no se puede simular un día sin precipitación de entrada en este esquema.
    """
    forcing = pd.read_csv(DATOS_DIR / "forzamiento.csv", parse_dates=["date"])
    flow = pd.read_csv(DATOS_DIR / "caudal.csv", parse_dates=["date"])
    df = forcing.merge(flow, on="date", how="left").sort_values("date")
    df = df.dropna(subset=[precip_col]).reset_index(drop=True)
    return df


def _run_default(df: pd.DataFrame, precip_col: str, area_km2: float) -> pd.DataFrame:
    """
    Corre el modelo NO calibrado (parámetros por defecto) y construye una tabla con:
    - Q simulado en mm/d y en m3/s
    - ET potencial y ET usada
    - Estados S1 y S2
    - ΔS diario
    - Residual del balance hídrico (mm) para auditoría

    Importante:
    - El modelo se resuelve internamente en mm/d.
    - Luego se convierte Qsim a m3/s para comparar con q_cms observado.
    """
    p = df[precip_col].to_numpy(dtype=float)
    params = default_params()
    q_sim_mm_day, et_used, s1, s2, ds = two_tank_run(p, params, return_states=True)
    et_pot = params.ETc + params.beta * p
    coef_cms = area_km2 * MMDAY_TO_CMS_PER_KM2

    out = df.copy()
    out["q_sim_mm_day"] = q_sim_mm_day
    out["q_sim_m3s"] = out["q_sim_mm_day"] * coef_cms
    out["ET_potencial_mm_day"] = et_pot
    out["ET_usada_mm_day"] = et_used
    out["ET_mm_day"] = out["ET_usada_mm_day"]
    out["S1_mm"] = s1
    out["S2_mm"] = s2
    out["dS_mm"] = ds
    out["balance_residual_mm"] = out[precip_col] - out["ET_usada_mm_day"] - out["q_sim_mm_day"] - out["dS_mm"]
    return out


def _plot_hydrograph_png(
    sim: pd.DataFrame,
    precip_col: str,
    precip_label: str,
    gauge_label: str,
    nse_value: float,
    out_png: Path,
) -> None:
    """
    Grafica estática (PNG) del hidrograma no calibrado:

    - Arriba: precipitación diaria (invertida, “crece hacia abajo”).
    - Abajo: caudal observado vs caudal simulado en m³/s.

    Incluye NSE en el título para que se entienda el desempeño rápidamente.
    """
    plt.style.use("dark_background")
    fig, (ax_p, ax_q) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.0], "hspace": 0.05},
    )

    ax_p.bar(
        sim["date"],
        sim[precip_col],
        color="#4ea1ff",
        width=1.0,
        alpha=0.9,
        label=f"P {precip_label} (mm/d)",
    )
    ax_p.set_ylabel("Precipitación (mm/d)", color="white", fontsize=12)
    ax_p.invert_yaxis()
    ax_p.grid(color="white", alpha=0.10, linewidth=0.8)
    ax_p.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)

    ax_q.plot(sim["date"], sim["q_sim_m3s"], color="#ff4d5a", linewidth=1.0, label="Q sim no calibrado (m³/s)")
    ax_q.plot(sim["date"], sim["q_cms"], color="#bfbfbf", linewidth=1.0, label="Q observado (m³/s)")
    ax_q.set_ylabel("Caudal (m³/s)", color="white", fontsize=12)
    ax_q.set_xlabel("Fecha", color="white", fontsize=12)
    ax_q.grid(color="white", alpha=0.10, linewidth=0.8)
    ax_q.legend(loc="upper left", facecolor="black", framealpha=0.45, fontsize=10)
    ax_q.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_q.set_xlim(XMIN, XMAX)

    fig.suptitle(
        f"Modelo no calibrado (NSE={nse_value:.3f})",
        color="white",
        fontsize=17,
        y=0.98,
    )
    for axis in (ax_p, ax_q):
        axis.tick_params(colors="white", labelsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="black")
    plt.close(fig)


def _plot_hydrograph_html(
    sim: pd.DataFrame,
    precip_col: str,
    precip_label: str,
    gauge_label: str,
    nse_value: float,
    out_html: Path,
) -> None:
    """
    Versión interactiva (HTML) del hidrograma no calibrado, con Plotly:

    - Dos paneles: P arriba (invertida) y Q abajo (observado vs simulado).
    - Permite zoom, hover, apagar/encender series, etc.
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
            x=sim["date"],
            y=sim[precip_col],
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
            x=sim["date"],
            y=sim["q_sim_m3s"],
            mode="lines",
            line=dict(color="#ff4d5a", width=1.2),
            name="Q sim no calibrado (m³/s)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sim["date"],
            y=sim["q_cms"],
            mode="lines",
            line=dict(color="#bfbfbf", width=1.2),
            name="Q observado (m³/s)",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Caudal (m³/s)", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_xaxes(range=[XMIN, XMAX], row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"Modelo no calibrado (NSE={nse_value:.3f})",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        )
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


def _write_balance_summary(
    sim: pd.DataFrame,
    precip_col: str,
    source: str,
    area_km2: float,
    nse_value: float,
    kge_value: float,
) -> Path:
    """
    Escribe un resumen de auditoría del balance hídrico y desempeño en un .txt.

    ¿Para qué sirve?
    - Deja “evidencia” de que el balance cierra (residual muy pequeño).
    - Guarda estadísticas del residual (acumulado, máximo diario, promedio).
    - Guarda NSE y KGE (en m³/s) para el caso no calibrado.

    Esto es útil para el reporte porque demuestra que:
    - El modelo está bien implementado (no hay errores de signo o desfases).
    - Los resultados son trazables (se puede reproducir el resumen).
    """
    out_txt = DATOS_DIR / f"two_tank_balance_summary_{source}.txt"
    residual = sim["balance_residual_mm"]
    abs_residual = residual.abs()
    cum_res = float(residual.sum())
    max_abs = float(abs_residual.max())
    mean_abs = float(abs_residual.mean())
    total_p = float(sim[precip_col].sum())
    rel_err = (cum_res / total_p) if total_p != 0 else float("nan")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"source={source}\n")
        f.write(f"period_start={sim['date'].min().date()}\n")
        f.write(f"period_end={sim['date'].max().date()}\n")
        f.write(f"n_days={len(sim)}\n")
        f.write(f"area_km2={area_km2:.6f}\n")
        f.write(f"nse_m3s={nse_value:.6f}\n")
        f.write(f"kge_m3s={kge_value:.6f}\n")
        f.write(f"residual_cumulative_mm={cum_res:.12e}\n")
        f.write(f"max_abs_daily_residual_mm={max_abs:.12e}\n")
        f.write(f"mean_abs_daily_residual_mm={mean_abs:.12e}\n")
        f.write(f"relative_error_final={rel_err:.12e}\n")
    return out_txt


def _cleanup_old_outputs() -> None:
    """
    Borra archivos viejos que podrían confundir si cambiaste nombres de salidas.

    Es decir, evita que queden “sobras” en la carpeta de figuras que no corresponden
    a la corrida actual.
    """
    old_paths = [
        FIG_DIR / "modelo_no_calibrado.png",
        FIG_HTML_DIR / "modelo_no_calibrado.html",
        FIG_DIR / "balance_hidrico_default.png",
    ]
    for path in old_paths:
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["daymet", "nldas"],
        choices=["daymet", "nldas", "maurer"],
    )
    args = parser.parse_args()

    _cleanup_old_outputs()
    gauge_label = f"{args.gauge} ({_gauge_name(args.gauge)})"
    params = default_params()

    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        precip_col = SOURCE_TO_COL[source]
        precip_label = SOURCE_TO_LABEL[source]
        df = _load(precip_col)
        area_km2 = _area_km2_for_gauge(args.gauge, df)
        sim = _run_default(df, precip_col, area_km2)

        nse_value = nse(sim["q_cms"].to_numpy(), sim["q_sim_m3s"].to_numpy())
        kge_value = kge(sim["q_cms"].to_numpy(), sim["q_sim_m3s"].to_numpy())

        out_csv = DATOS_DIR / f"two_tank_uncalibrated_daily_{source}.csv"
        sim.to_csv(out_csv, index=False)

        out_json = DATOS_DIR / f"two_tank_default_params_{source}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gauge_id": args.gauge,
                    "source": source,
                    "precip_column": precip_col,
                    "params": asdict(params),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        out_balance = _write_balance_summary(sim, precip_col, source, area_km2, nse_value, kge_value)

        out_png = FIG_DIR / f"two_tank_uncalibrated_hydrograph_{source}.png"
        out_html = FIG_HTML_DIR / f"two_tank_uncalibrated_hydrograph_{source}.html"
        _plot_hydrograph_png(sim, precip_col, precip_label, gauge_label, nse_value, out_png)
        _plot_hydrograph_html(sim, precip_col, precip_label, gauge_label, nse_value, out_html)

        print(f"[ok] {source}: {out_csv.name}")
        print(f"[ok] {source}: {out_json.name}")
        print(f"[ok] {source}: {out_balance.name}")
        print(f"[ok] {source}: {out_png.name}")
        print(f"[ok] {source}: {out_html.name}")


if __name__ == "__main__":
    main()
