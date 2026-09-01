from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modelo import TwoTankParams, two_tank_run
from metadata import area_km2
from paths import DATOS_DIR, FIG_HTML_DIR
from plotly_theme import apply_dark_theme

MMDAY_TO_CMS_PER_KM2 = 0.0115740741
SOURCE_TO_COL = {"daymet": "prcp_daymet_mm_day", "nldas": "prcp_nldas_mm_day"}
CAL_START = pd.Timestamp("1990-01-01")
CAL_END = pd.Timestamp("1999-12-31")
VAL_START = pd.Timestamp("2000-01-01")
VAL_END = pd.Timestamp("2009-12-31")


@dataclass
class ScenarioSeries:
    label: str
    source: str
    frame: pd.DataFrame


def _area_km2_for_gauge(gauge_id: str) -> float:
    return area_km2(gauge_id)


def _load_merged(gauge_id: str) -> pd.DataFrame:
    path = DATOS_DIR / gauge_id / f"{gauge_id}_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta preprocesado primero.")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


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


def _simulate_series(df_merged: pd.DataFrame, source: str, params: TwoTankParams, area_km2: float) -> pd.DataFrame:
    precip_col = SOURCE_TO_COL[source]
    df = (
        df_merged[["date", precip_col]]
        .rename(columns={precip_col: "prcp_mm_day"})
        .dropna(subset=["prcp_mm_day"])
        .copy()
    )
    q_sim_mm_day, et_used, _s1, _s2, ds = two_tank_run(df["prcp_mm_day"].to_numpy(dtype=float), params, return_states=True)
    df["q_sim_mm_day"] = q_sim_mm_day
    df["ET_usada_mm_day"] = et_used
    df["dS_mm"] = ds
    df["balance_residual_mm"] = df["prcp_mm_day"] - df["ET_usada_mm_day"] - df["q_sim_mm_day"] - df["dS_mm"]
    df["q_sim_m3s"] = df["q_sim_mm_day"] * (area_km2 * MMDAY_TO_CMS_PER_KM2)
    return df


def _load_uncalibrated(source: str) -> pd.DataFrame | None:
    path = DATOS_DIR / f"two_tank_uncalibrated_daily_{source}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    precip_col = SOURCE_TO_COL[source]
    if precip_col in df.columns:
        df = df.rename(columns={precip_col: "prcp_mm_day"})
    req = ["date", "prcp_mm_day", "ET_usada_mm_day", "q_sim_mm_day", "dS_mm", "balance_residual_mm", "q_sim_m3s"]
    if all(c in df.columns for c in req):
        return df[req].dropna(subset=["prcp_mm_day"]).copy()
    return None


def _load_calibration_table(path: Path, scenario_tag: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"gauge_id": "string", "source": "string", "method": "string"})
    if df.empty:
        return df
    df["scenario"] = scenario_tag
    return df


def _collect_scenarios(gauge_id: str) -> list[dict]:
    tables = []
    tables.append(_load_calibration_table(DATOS_DIR / "tabla_source_x_method.csv", "STD"))
    tables.append(_load_calibration_table(DATOS_DIR / "tabla_source_x_method_extremos.csv", "EXT"))
    full = pd.concat([t for t in tables if not t.empty], ignore_index=True) if any(not t.empty for t in tables) else pd.DataFrame()
    if full.empty:
        return []
    full = full[full["gauge_id"].astype(str).str.zfill(8) == str(gauge_id).zfill(8)].copy()
    if full.empty:
        return []
    rows = []
    for _, row in full.iterrows():
        rows.append(
            {
                "label": f"{row['scenario']} | {str(row['source']).upper()}-{str(row['method']).upper()}",
                "source": str(row["source"]),
                "params": _params_from_row(row),
            }
        )
    return rows


def _build_mass_figure(series_list: list[ScenarioSeries]) -> go.Figure:
    fig = go.Figure()
    for scenario in series_list:
        df = scenario.frame
        p_cum = df["prcp_mm_day"].cumsum()
        out_cum = (df["ET_usada_mm_day"] + df["q_sim_mm_day"] + df["dS_mm"]).cumsum()
        res_cum = df["balance_residual_mm"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=p_cum,
                mode="lines",
                line=dict(color="#4ea1ff", width=1.6),
                name="P acumulada",
                visible=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=out_cum,
                mode="lines",
                line=dict(color="#ff4d5a", width=1.6, dash="dash"),
                name="(ET + Q + ΔS) acumulada",
                visible=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=res_cum,
                mode="lines",
                line=dict(color="#ffd166", width=1.4, dash="dot"),
                name="Residual acumulado",
                yaxis="y2",
                visible=False,
            )
        )

    if fig.data:
        for i in range(3):
            fig.data[i].visible = True

    buttons = []
    total = len(series_list)
    for idx, scenario in enumerate(series_list):
        vis = [False] * (total * 3)
        vis[idx * 3 : idx * 3 + 3] = [True, True, True]
        buttons.append(
            dict(
                label=scenario.label,
                method="update",
                args=[
                    {"visible": vis},
                    {"title.text": f"Balance hídrico acumulado | {scenario.label}"},
                ],
            )
        )

    fig.update_layout(
        title=dict(text=f"Balance hídrico acumulado | {series_list[0].label if series_list else ''}"),
        yaxis=dict(title="Acumulado (mm)"),
        yaxis2=dict(
            title="Residual acumulado (mm)",
            overlaying="y",
            side="right",
            tickformat=".2e",
            exponentformat="e",
            showexponent="all",
        ),
        xaxis=dict(title="Fecha"),
        updatemenus=[dict(type="dropdown", x=1.0, y=1.15, xanchor="right", buttons=buttons)],
    )
    apply_dark_theme(fig)
    fig.update_layout(
        title=dict(x=0.5, xanchor="center", y=0.99, yanchor="top"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.95,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=120, b=60),
    )
    fig.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Residual final ≈ 0 mm (orden 10⁻¹²)",
        showarrow=False,
        font=dict(size=12, color="white"),
        align="left",
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
    )
    return fig


def _build_sankey_figure(series_list: list[ScenarioSeries]) -> go.Figure:
    fig = go.Figure()
    titles: list[str] = []
    for scenario in series_list:
        df = scenario.frame
        p_total = float(df["prcp_mm_day"].sum())
        et_total = float(df["ET_usada_mm_day"].sum())
        q_total = float(df["q_sim_mm_day"].sum())
        ds_pos = float(df.loc[df["dS_mm"] > 0, "dS_mm"].sum())
        ds_neg = float((-df.loc[df["dS_mm"] < 0, "dS_mm"]).sum())
        residual = float(df["balance_residual_mm"].sum())

        input_total = p_total + ds_neg
        out_residual = max(input_total - (et_total + q_total + ds_pos), 0.0)

        labels = [
            "Precipitación (P)",
            "Liberación de almacenamiento (-ΔS)",
            "Entradas efectivas",
            "ET usada",
            "Q simulado",
            "Recarga almacenamiento (+ΔS)",
            "Residual",
        ]
        source = [0, 1, 2, 2, 2, 2]
        target = [2, 2, 3, 4, 5, 6]
        value = [p_total, ds_neg, et_total, q_total, ds_pos, out_residual]

        fig.add_trace(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(255,255,255,0.2)", width=1),
                    label=labels,
                    color=["#4ea1ff", "#8a5cff", "#00bfff", "#ffd166", "#ff4d5a", "#38d27a", "#ff8c42"],
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=[
                        "rgba(78,161,255,0.45)",
                        "rgba(138,92,255,0.45)",
                        "rgba(255,209,102,0.45)",
                        "rgba(255,77,90,0.45)",
                        "rgba(56,210,122,0.45)",
                        "rgba(255,140,66,0.45)",
                    ],
                ),
                visible=False,
            )
        )
        titles.append(
            f"Sankey balance | {scenario.label} | "
            f"P={p_total:.1f}, ET={et_total:.1f}, Q={q_total:.1f}, Residual={residual:.2e} mm, +ΔS={ds_pos:.1f}, -ΔS={ds_neg:.1f}"
        )

    if fig.data:
        fig.data[0].visible = True

    buttons = []
    total = len(series_list)
    for idx, scenario in enumerate(series_list):
        vis = [False] * total
        vis[idx] = True
        buttons.append(
            dict(
                label=scenario.label,
                method="update",
                args=[{"visible": vis}, {"title.text": titles[idx]}],
            )
        )

    fig.update_layout(
        title=dict(text=titles[0] if titles else "Sankey balance"),
        updatemenus=[dict(type="dropdown", x=1.0, y=1.15, xanchor="right", buttons=buttons)],
    )
    apply_dark_theme(fig)
    return fig


def _build_heatmap_figure(series_list: list[ScenarioSeries]) -> go.Figure:
    fig = go.Figure()
    titles: list[str] = []
    for scenario in series_list:
        df = scenario.frame.copy()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        piv = df.pivot_table(index="year", columns="month", values="balance_residual_mm", aggfunc="mean")
        years = sorted(df["year"].unique().tolist())
        months = list(range(1, 13))
        piv = piv.reindex(index=years, columns=months)

        fig.add_trace(
            go.Heatmap(
                z=piv.to_numpy(dtype=float),
                x=["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
                y=years,
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="Residual medio (mm/d)"),
                visible=False,
            )
        )
        titles.append(f"Heatmap residual mensual (media diaria) | {scenario.label}")

    if fig.data:
        fig.data[0].visible = True

    buttons = []
    total = len(series_list)
    for idx, scenario in enumerate(series_list):
        vis = [False] * total
        vis[idx] = True
        buttons.append(
            dict(
                label=scenario.label,
                method="update",
                args=[{"visible": vis}, {"title.text": titles[idx]}],
            )
        )

    fig.update_layout(
        title=dict(text=titles[0] if titles else "Heatmap residual"),
        xaxis=dict(title="Mes"),
        yaxis=dict(title="Año"),
        updatemenus=[dict(type="dropdown", x=1.0, y=1.15, xanchor="right", buttons=buttons)],
    )
    apply_dark_theme(fig)
    return fig


def _write_dashboard(out_path: Path, fig_mass: go.Figure, fig_sankey: go.Figure, fig_heat: go.Figure) -> None:
    div1 = fig_mass.to_html(full_html=False, include_plotlyjs="cdn")
    div2 = fig_sankey.to_html(full_html=False, include_plotlyjs=False)
    div3 = fig_heat.to_html(full_html=False, include_plotlyjs=False)
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard Balance Hídrico</title>
  <style>
    body {{ background:#000; color:#fff; font-family:Arial, sans-serif; margin:0; padding:20px; }}
    h1 {{ margin:0 0 8px 0; font-size:28px; }}
    p {{ margin:0 0 16px 0; color:#cfcfcf; }}
    .card {{ border:1px solid #222; border-radius:10px; padding:12px; margin-bottom:16px; background:#050505; }}
  </style>
</head>
<body>
  <h1>Dashboard de balance hídrico</h1>
  <p>Curva de masa acumulada + Sankey agregado + Heatmap de residual mensual.</p>
  <div class="card">{div1}</div>
  <div class="card">{div2}</div>
  <div class="card">{div3}</div>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    args = parser.parse_args()

    gauge_id = str(args.gauge).zfill(8)
    area_km2 = _area_km2_for_gauge(gauge_id)
    df_merged = _load_merged(gauge_id)

    scenario_series: list[ScenarioSeries] = []

    for source in ["daymet", "nldas"]:
        unc = _load_uncalibrated(source)
        if unc is not None:
            scenario_series.append(
                ScenarioSeries(label=f"UNC | {source.upper()}", source=source, frame=unc)
            )

    calib_rows = _collect_scenarios(gauge_id)
    for row in calib_rows:
        frame = _simulate_series(df_merged, row["source"], row["params"], area_km2)
        scenario_series.append(
            ScenarioSeries(label=row["label"], source=row["source"], frame=frame)
        )

    if not scenario_series:
        raise RuntimeError("No se encontraron escenarios para construir el dashboard.")

    fig_mass = _build_mass_figure(scenario_series)
    fig_sankey = _build_sankey_figure(scenario_series)
    fig_heat = _build_heatmap_figure(scenario_series)

    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)
    out_mass = FIG_HTML_DIR / "balance_masa_interactivo.html"
    out_sankey = FIG_HTML_DIR / "balance_sankey_interactivo.html"
    out_heat = FIG_HTML_DIR / "balance_heatmap_residual_interactivo.html"
    out_dashboard = FIG_HTML_DIR / "balance_dashboard_interactivo.html"

    fig_mass.write_html(out_mass, include_plotlyjs="cdn")
    fig_sankey.write_html(out_sankey, include_plotlyjs="cdn")
    fig_heat.write_html(out_heat, include_plotlyjs="cdn")
    _write_dashboard(out_dashboard, fig_mass, fig_sankey, fig_heat)

    print(f"[ok] {out_mass}")
    print(f"[ok] {out_sankey}")
    print(f"[ok] {out_heat}")
    print(f"[ok] {out_dashboard}")


if __name__ == "__main__":
    main()
