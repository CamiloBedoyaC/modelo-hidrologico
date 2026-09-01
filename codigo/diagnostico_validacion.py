from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modelo import TwoTankParams, two_tank_run
from metadata import area_km2
from paths import DATOS_DIR, FIG_HTML_DIR
from plotly_theme import apply_dark_theme

VAL_START = pd.Timestamp("2000-01-01")
VAL_END = pd.Timestamp("2009-12-31")
MMDAY_TO_CMS_PER_KM2 = 0.0115740741
SOURCE_TO_COL = {"daymet": "prcp_daymet_mm_day", "nldas": "prcp_nldas_mm_day"}


@dataclass
class RunValidation:
    label: str
    source: str
    method: str
    scenario: str
    data: pd.DataFrame
    nse_val: float | None
    kge_val: float | None


def _area_km2_for_gauge(gauge_id: str) -> float:
    return area_km2(gauge_id)


def _load_merged(gauge_id: str) -> pd.DataFrame:
    path = DATOS_DIR / gauge_id / f"{gauge_id}_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta preprocesado primero.")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def _load_results_tables() -> pd.DataFrame:
    tables = []
    standard_path = DATOS_DIR / "tabla_source_x_method.csv"
    extremos_path = DATOS_DIR / "tabla_source_x_method_extremos.csv"
    if standard_path.exists():
        std = pd.read_csv(standard_path, dtype={"gauge_id": "string", "source": "string", "method": "string"})
        std["scenario"] = "estandar"
        tables.append(std)
    if extremos_path.exists():
        ext = pd.read_csv(extremos_path, dtype={"gauge_id": "string", "source": "string", "method": "string"})
        ext["scenario"] = "extremos"
        tables.append(ext)
    if not tables:
        raise FileNotFoundError("No se encontraron tablas de calibración en datos/.")
    return pd.concat(tables, ignore_index=True)


def _params_from_row(row: pd.Series) -> TwoTankParams:
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


def _simulate_validation_runs(df_merged: pd.DataFrame, table: pd.DataFrame, area_km2: float) -> list[RunValidation]:
    coef_cms = area_km2 * MMDAY_TO_CMS_PER_KM2
    runs: list[RunValidation] = []

    for _, row in table.iterrows():
        source = str(row["source"])
        method = str(row["method"])
        scenario = str(row["scenario"])
        precip_col = SOURCE_TO_COL[source]
        params = _params_from_row(row)

        sub = (
            df_merged[["date", precip_col, "q_cms"]]
            .rename(columns={precip_col: "prcp_mm_day", "q_cms": "q_obs_m3s"})
            .dropna(subset=["prcp_mm_day"])
            .copy()
        )
        q_sim_mm_day = two_tank_run(sub["prcp_mm_day"].to_numpy(dtype=float), params, return_states=False)
        sub["q_sim_m3s"] = q_sim_mm_day * coef_cms
        val = sub[(sub["date"] >= VAL_START) & (sub["date"] <= VAL_END)].copy()

        nse_val = float(row["NSE_val"]) if "NSE_val" in row and pd.notna(row["NSE_val"]) else None
        kge_val = float(row["KGE_val"]) if "KGE_val" in row and pd.notna(row["KGE_val"]) else None
        metric_text = f"NSEv={nse_val:.3f}" if nse_val is not None else "NSEv=n/a"
        label = f"{scenario.capitalize()} | {source.upper()}-{method.upper()} ({metric_text})"

        runs.append(
            RunValidation(
                label=label,
                source=source,
                method=method,
                scenario=scenario,
                data=val,
                nse_val=nse_val,
                kge_val=kge_val,
            )
        )

    return runs


def _fdc_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = values[np.isfinite(values)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.array([]), np.array([])
    vals = np.sort(vals)[::-1]
    n = vals.size
    exceed = 100.0 * np.arange(1, n + 1) / (n + 1)
    return exceed, vals


def _build_scatter_html(runs: list[RunValidation], out_html: str) -> None:
    fig = go.Figure()
    colors = [
        "#00bfff", "#ff4d5a", "#38d27a", "#ffd166",
        "#8a5cff", "#ff00ff", "#00ffa6", "#ff7f50",
    ]

    all_vals = []
    for i, run in enumerate(runs):
        d = run.data
        all_vals.append(d["q_obs_m3s"].to_numpy(dtype=float))
        all_vals.append(d["q_sim_m3s"].to_numpy(dtype=float))
        fig.add_trace(
            go.Scattergl(
                x=d["q_obs_m3s"],
                y=d["q_sim_m3s"],
                mode="markers",
                name=run.label,
                marker=dict(color=colors[i % len(colors)], size=4, opacity=0.35),
                customdata=np.c_[d["date"].dt.strftime("%Y-%m-%d")],
                hovertemplate=(
                    "Fecha=%{customdata[0]}<br>"
                    "Q obs=%{x:.2f} m3/s<br>"
                    "Q sim=%{y:.2f} m3/s"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals else 1.0
    vmax = max(vmax, 1.0)
    fig.add_trace(
        go.Scatter(
            x=[0, vmax],
            y=[0, vmax],
            mode="lines",
            line=dict(color="#cccccc", width=1.3, dash="dash"),
            name="Línea 1:1",
            hoverinfo="skip",
        )
    )

    fig.update_layout(title="Validación (2000-2009): dispersión Q observado vs Q simulado")
    fig.update_xaxes(title_text="Q observado (m3/s)", range=[0, vmax * 1.02])
    fig.update_yaxes(title_text="Q simulado (m3/s)", range=[0, vmax * 1.02])
    apply_dark_theme(fig)
    fig.write_html(out_html, include_plotlyjs="cdn")


def _build_fdc_html(runs: list[RunValidation], out_html: str) -> None:
    fig = go.Figure()
    colors = [
        "#00bfff", "#ff4d5a", "#38d27a", "#ffd166",
        "#8a5cff", "#ff00ff", "#00ffa6", "#ff7f50",
    ]

    # Observado referencia (idéntico para todos en validación)
    d0 = runs[0].data if runs else pd.DataFrame(columns=["q_obs_m3s"])
    ex_obs, q_obs = _fdc_xy(d0["q_obs_m3s"].to_numpy(dtype=float))
    fig.add_trace(
        go.Scatter(
            x=ex_obs,
            y=q_obs,
            mode="lines",
            line=dict(color="#e6e6e6", width=2.0),
            name="Q observado (validación)",
        )
    )

    for i, run in enumerate(runs):
        ex_sim, q_sim = _fdc_xy(run.data["q_sim_m3s"].to_numpy(dtype=float))
        fig.add_trace(
            go.Scatter(
                x=ex_sim,
                y=q_sim,
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.4),
                name=run.label,
            )
        )

    fig.update_layout(title="Validación (2000-2009): Curva de duración de caudales (FDC)")
    fig.update_xaxes(title_text="Probabilidad de excedencia (%)")
    fig.update_yaxes(title_text="Caudal (m3/s) [escala log]", type="log")
    apply_dark_theme(fig)
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    args = parser.parse_args()

    table = _load_results_tables()
    gauge_id = str(args.gauge).zfill(8)
    table = table[table["gauge_id"].astype(str).str.zfill(8) == gauge_id].copy()
    if table.empty:
        raise ValueError(f"No hay filas de calibración para gauge {gauge_id}.")

    df_merged = _load_merged(gauge_id)
    area_km2 = _area_km2_for_gauge(gauge_id)
    runs = _simulate_validation_runs(df_merged, table, area_km2)

    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)
    out_scatter = FIG_HTML_DIR / "validacion_scatter_obs_vs_sim.html"
    out_fdc = FIG_HTML_DIR / "validacion_fdc_obs_vs_sim.html"

    _build_scatter_html(runs, str(out_scatter))
    _build_fdc_html(runs, str(out_fdc))

    print(f"[ok] {out_scatter}")
    print(f"[ok] {out_fdc}")


if __name__ == "__main__":
    main()
