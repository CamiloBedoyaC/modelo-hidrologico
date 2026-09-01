"""Audita datos, metricas, balance y enlaces sin recalibrar el modelo.

El comando es de solo lectura, salvo que se use ``--write-audits`` para
actualizar los CSV de auditoria a partir de las tablas vigentes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

import calibrar
import calibrar_extremos
from descargar_camels import KNOWN_MD5, _md5
from metadata import area_km2
from modelo import default_params, two_tank_run
from paths import DATOS_DIR, FIG_DIR, INFORME_DIR, ROOT_DIR


TOL = 1e-9
EXPECTED_PACKAGES = {
    "numpy": "2.3.4",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "matplotlib": "3.10.7",
    "plotly": "6.3.0",
    "requests": "2.32.3",
    "tqdm": "4.67.1",
    "pypdf": "6.7.1",
    "pyshp": "2.3.1",
}


class _LocalReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key.lower() in {"src", "href", "data-src"} and value:
                self.references.append(value)


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        print(f"[ERROR] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def require(self, condition: bool, message: str) -> None:
        self.ok(message) if condition else self.fail(message)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _check_environment(audit: Audit) -> None:
    audit.require(sys.version_info[:2] == (3, 12), f"Python 3.12 (actual: {sys.version.split()[0]})")
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            audit.fail(f"Dependencia ausente: {package}=={expected}")
            continue
        if current == expected:
            audit.ok(f"{package}=={expected}")
        else:
            audit.warn(f"{package}: probado con {expected}; instalado {current}")


def _check_inputs(audit: Audit, require_raw: bool) -> pd.DataFrame:
    required = [
        DATOS_DIR / "forzamiento.csv",
        DATOS_DIR / "caudal.csv",
        DATOS_DIR / "atributos.csv",
        DATOS_DIR / "02327100" / "02327100_merged.csv",
        DATOS_DIR / "tabla_source_x_method.csv",
        DATOS_DIR / "tabla_source_x_method_extremos.csv",
    ]
    for path in required:
        audit.require(path.exists() and path.stat().st_size > 0, f"Existe {path.relative_to(ROOT_DIR)}")

    if require_raw:
        for filename, expected_md5 in KNOWN_MD5.items():
            raw = DATOS_DIR / "raw" / filename
            audit.require(raw.exists() and raw.stat().st_size > 0, f"Existe el crudo CAMELS-US: {filename}")
            if raw.exists() and raw.stat().st_size > 0:
                audit.require(
                    _md5(raw).lower() == expected_md5.lower(),
                    f"MD5 oficial correcto: {filename}",
                )

    forcing = pd.read_csv(DATOS_DIR / "forzamiento.csv", parse_dates=["date"])
    flow = pd.read_csv(DATOS_DIR / "caudal.csv", parse_dates=["date"])
    expected_forcing = {"date", "prcp_daymet_mm_day", "prcp_maurer_mm_day", "prcp_nldas_mm_day"}
    expected_flow = {"date", "q_cfs", "q_cms", "q_obs_mm_day"}
    audit.require(expected_forcing.issubset(forcing.columns), "Esquema de forzamiento.csv")
    audit.require(expected_flow.issubset(flow.columns), "Esquema de caudal.csv")
    audit.require(len(forcing) == 12784 and len(flow) == 12784, "12784 fechas diarias (1980-2014)")
    audit.require(
        forcing["date"].min() == pd.Timestamp("1980-01-01")
        and forcing["date"].max() == pd.Timestamp("2014-12-31"),
        "Cobertura temporal 1980-01-01 a 2014-12-31",
    )

    qmask = flow[["q_cfs", "q_cms"]].notna().all(axis=1)
    qdiff = (flow.loc[qmask, "q_cms"] - flow.loc[qmask, "q_cfs"] * 0.0283168).abs().max()
    audit.require(float(qdiff) < 1e-10, "Conversion observada cfs -> m3/s")

    area = area_km2("02327100")
    mm_mask = flow[["q_cms", "q_obs_mm_day"]].notna().all(axis=1)
    qmm = flow.loc[mm_mask, "q_cms"] / (area * calibrar.MMDAY_TO_CMS_PER_KM2)
    audit.require(float((qmm - flow.loc[mm_mask, "q_obs_mm_day"]).abs().max()) < 1e-10, "Conversion m3/s -> mm/d")
    merged = pd.read_csv(DATOS_DIR / "02327100" / "02327100_merged.csv", parse_dates=["date"])
    expected_dates = pd.date_range("1980-01-01", "2014-12-31", freq="D")
    audit.require(
        merged["date"].reset_index(drop=True).equals(pd.Series(expected_dates, name="date")),
        "Serie integrada diaria, ordenada, sin duplicados ni huecos",
    )
    for source in ["daymet", "nldas"]:
        source_frame = calibrar._dataset_for_source(merged, source)
        audit.require(
            len(source_frame) == len(expected_dates),
            f"{source}: la simulacion conserva dias sin caudal observado",
        )
    return merged


def _check_model_invariants(audit: Audit) -> None:
    rng = np.random.default_rng(20260830)
    precip = rng.gamma(shape=0.7, scale=7.0, size=5000)
    params = default_params()
    q_sim, et_used, s1, s2, ds = two_tank_run(precip, params, return_states=True)
    residual = precip - et_used - q_sim - ds
    audit.require(float(np.max(np.abs(residual))) < 1e-12, "Modelo conserva masa diariamente")
    audit.require(bool((q_sim >= 0).all() and (et_used >= 0).all()), "Flujos simulados no negativos")
    audit.require(
        bool((s1 >= 0).all() and (s1 <= params.D1).all() and (s2 >= 0).all() and (s2 <= params.D2).all()),
        "Almacenamientos dentro de sus capacidades",
    )


def _check_standard(audit: Audit, merged: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(DATOS_DIR / "tabla_source_x_method.csv", dtype={"gauge_id": "string"})
    audit.require(len(table) == 4, "Cuatro combinaciones en calibracion estandar")
    audit.require(
        set(zip(table["source"], table["method"]))
        == {("daymet", "da"), ("daymet", "de"), ("nldas", "da"), ("nldas", "de")},
        "Combinaciones Daymet/NLDAS x DA/DE",
    )
    rows: list[dict[str, float | str]] = []
    area = area_km2("02327100")
    for _, row in table.iterrows():
        source, method = str(row["source"]), str(row["method"])
        frame = calibrar._dataset_for_source(merged, source)
        params = calibrar._params_from_row(row)
        _series, met_cal, met_val, _balance, _peaks = calibrar.evaluate(frame, params, area)
        precip, q_obs, calibration_mask = calibrar._calibration_inputs(frame)
        vector = np.array([getattr(params, name) for name in calibrar.PARAM_NAMES], dtype=float)
        objective_nse = -calibrar.objective_nse(
            vector,
            precip,
            q_obs,
            area * calibrar.MMDAY_TO_CMS_PER_KM2,
            calibration_mask,
        )
        cal_diff = float(met_cal["NSE"] - row["NSE_cal"])
        val_diff = float(met_val["NSE"] - row["NSE_val"])
        audit.require(abs(cal_diff) < TOL and abs(val_diff) < TOL, f"NSE reproducido: {source}-{method}")
        audit.require(
            abs(objective_nse - met_cal["NSE"]) < TOL,
            f"Objetivo y evaluacion usan los mismos estados: {source}-{method}",
        )
        rows.append(
            {
                "source": source,
                "method": method,
                "NSE_cal_csv": row["NSE_cal"],
                "NSE_cal_calc": met_cal["NSE"],
                "NSE_cal_diff": cal_diff,
                "NSE_objective_calc": objective_nse,
                "NSE_objective_diff": objective_nse - met_cal["NSE"],
                "NSE_val_csv": row["NSE_val"],
                "NSE_val_calc": met_val["NSE"],
                "NSE_val_diff": val_diff,
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "method"])


def _check_extremes(audit: Audit, merged: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(DATOS_DIR / "tabla_source_x_method_extremos.csv", dtype={"gauge_id": "string"})
    audit.require(len(table) == 4, "Cuatro combinaciones en calibracion de extremos")
    rows: list[dict[str, float | str]] = []
    area = area_km2("02327100")
    for _, row in table.iterrows():
        source, method = str(row["source"]), str(row["method"])
        frame = calibrar_extremos._dataset_for_source(merged, source)
        params = calibrar_extremos._params_from_row(row)
        _series, metrics, _extras = calibrar_extremos.evaluate(frame, params, area)
        precip, q_obs, calibration_mask, q95 = calibrar_extremos._calibration_inputs(frame)
        vector = np.array([getattr(params, name) for name in calibrar_extremos.PARAM_NAMES], dtype=float)
        objective_j = calibrar_extremos.objective_extremos(
            vector,
            precip,
            q_obs,
            area * calibrar_extremos.MMDAY_TO_CMS_PER_KM2,
            calibration_mask,
            q95,
        )
        keys = ["J_cal", "J_val", "NSE_cal", "NSE_val"]
        diffs = {key: float(metrics[key] - row[key]) for key in keys}
        audit.require(all(abs(value) < TOL for value in diffs.values()), f"J/NSE reproducidos: {source}-{method}")
        audit.require(
            abs(objective_j - metrics["J_cal"]) < TOL,
            f"Objetivo J y evaluacion usan los mismos estados: {source}-{method}",
        )
        rows.append(
            {
                "source": source,
                "method": method,
                "J_cal_csv": row["J_cal"],
                "J_cal_calc": metrics["J_cal"],
                "J_cal_diff": diffs["J_cal"],
                "J_objective_calc": objective_j,
                "J_objective_diff": objective_j - metrics["J_cal"],
                "J_val_csv": row["J_val"],
                "J_val_calc": metrics["J_val"],
                "J_val_diff": diffs["J_val"],
                "NSE_cal_csv": row["NSE_cal"],
                "NSE_cal_calc": metrics["NSE_cal"],
                "NSE_val_csv": row["NSE_val"],
                "NSE_val_calc": metrics["NSE_val"],
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "method"])


def _check_bounds(audit: Audit) -> None:
    for filename in ["tabla_source_x_method.csv", "tabla_source_x_method_extremos.csv"]:
        table = pd.read_csv(DATOS_DIR / filename)
        for name, (lower, upper) in zip(calibrar.PARAM_NAMES, calibrar.BOUNDS):
            inside = table[name].between(lower - TOL, upper + TOL).all()
            audit.require(bool(inside), f"{filename}: {name} dentro de [{lower}, {upper}]")


def _check_html(audit: Audit) -> None:
    html_files = (
        list(FIG_DIR.rglob("*.html"))
        + list(INFORME_DIR.glob("*.html"))
        + [ROOT_DIR / "index.html"]
    )
    missing: list[str] = []
    for path in html_files:
        text = path.read_text(encoding="utf-8", errors="strict")
        parser = _LocalReferenceParser()
        parser.feed(text)
        for reference in parser.references:
            cleaned = reference.split("#", 1)[0].split("?", 1)[0]
            if not cleaned or cleaned.startswith(("http://", "https://", "data:", "about:", "#", "javascript:")):
                continue
            target = (path.parent / cleaned).resolve()
            if not target.exists():
                missing.append(f"{path.relative_to(ROOT_DIR)} -> {reference}")
    audit.require(not missing, "Todos los enlaces HTML locales existen")
    for item in missing:
        audit.fail(item)

    standalone = INFORME_DIR / "reporte_tarea1_interactivo_standalone.html"
    if standalone.exists():
        text = standalone.read_text(encoding="utf-8")
        audit.require(text.lower().count("srcdoc=") >= 19, "Standalone contiene las visualizaciones incrustadas")
        has_plotly_script_cdn = bool(
            re.search(
                r"src\s*=\s*(?:[\"']|&quot;)https://cdn\.plot\.ly/plotly-",
                text,
                flags=re.IGNORECASE,
            )
        )
        audit.require(not has_plotly_script_cdn, "Standalone incluye Plotly local, sin CDN de scripts")
        audit.warn("Solo el mapa Leaflet del standalone requiere Internet para su libreria y teselas de fondo.")
    else:
        audit.warn(
            "Standalone opcional ausente; se genera con codigo/generar_reporte_standalone.py."
        )


def _check_reports(audit: Audit) -> None:
    standard = pd.read_csv(DATOS_DIR / "tabla_source_x_method.csv")
    extremes = pd.read_csv(DATOS_DIR / "tabla_source_x_method_extremos.csv")
    html_source = (INFORME_DIR / "reporte_tarea1_interactivo.html").read_text(encoding="utf-8")
    tex_source = (INFORME_DIR / "reporte_tarea1.tex").read_text(encoding="utf-8")

    standard_cols = [
        "NSE_cal", "NSE_val", "KGE_val", "peak_ratio_val", "p95_ratio_val",
        "ETc", "beta", "alpha1", "D1", "k1", "alpha2", "D2", "k2",
    ]
    extremes_cols = [
        "J_cal", "J_val", "NSE_cal", "NSE_val", "NSE_high_val",
        "PeakBias_val", "UnderHigh_val", "ETc", "beta", "alpha1", "D1",
        "k1", "alpha2", "D2", "k2",
    ]

    def check_rows(table: pd.DataFrame, columns: list[str], label: str) -> None:
        for _, row in table.iterrows():
            values = [str(row["source"]), str(row["method"])] + [
                f"{float(row[column]):.3f}" for column in columns
            ]
            html_pattern = r"\s*".join(
                re.escape(f"<td>{value}</td>") for value in values
            )
            tex_row = " & ".join(values) + r" \\"
            suffix = f"{row['source']}-{row['method']}"
            audit.require(
                bool(re.search(html_pattern, html_source)),
                f"HTML sincronizado con {label}: {suffix}",
            )
            audit.require(
                tex_row in tex_source,
                f"TeX sincronizado con {label}: {suffix}",
            )

    check_rows(standard, standard_cols, "calibracion estandar")
    check_rows(extremes, extremes_cols, "calibracion de extremos")

    best_nse = float(standard["NSE_val"].max())
    expected_best = f"{best_nse:.3f}"
    for path in [ROOT_DIR / "README.md", ROOT_DIR / "index.html"]:
        audit.require(
            expected_best in path.read_text(encoding="utf-8"),
            f"{path.name} reporta el mejor NSE_val={expected_best}",
        )

    pdf = INFORME_DIR / "reporte_tarea1.pdf"
    audit.require(pdf.exists() and pdf.stat().st_size > 0, "PDF final existe")
    if pdf.exists():
        reader = PdfReader(pdf)
        audit.require(len(reader.pages) == 17, "PDF final legible (17 paginas)")
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        audit.require(expected_best in pdf_text, f"PDF reporta el mejor NSE_val={expected_best}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-raw", action="store_true", help="Exige tambien el ZIP crudo de 3.4 GB.")
    parser.add_argument("--write-audits", action="store_true", help="Actualiza los CSV de auditoria.")
    args = parser.parse_args()

    audit = Audit()
    _check_environment(audit)
    merged = _check_inputs(audit, require_raw=args.require_raw)
    _check_model_invariants(audit)
    standard = _check_standard(audit, merged)
    extremes = _check_extremes(audit, merged)
    _check_bounds(audit)
    _check_html(audit)
    _check_reports(audit)

    if args.write_audits and not audit.errors:
        standard.to_csv(DATOS_DIR / "auditoria_calibracion_estandar.csv", index=False)
        extremes.to_csv(DATOS_DIR / "auditoria_calibracion_extremos.csv", index=False)
        audit.ok("CSV de auditoria actualizados")

    print(f"\nResumen: {len(audit.errors)} errores, {len(audit.warnings)} advertencias.")
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
