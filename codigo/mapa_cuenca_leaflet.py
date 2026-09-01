from __future__ import annotations

import argparse
import json
import zipfile
from html import escape
from pathlib import Path

import pandas as pd
import shapefile  # pyshp

from paths import DATOS_DIR, FIG_HTML_DIR, RAW_DIR
# RAW_DIR: datos descargados “crudos” (zip del shapefile y archivos txt CAMELS)
# DATOS_DIR: datos procesados y salidas por cuenca
# FIG_HTML_DIR: carpeta de salidas interactivas HTML

SHAPES_ZIP = RAW_DIR / "basin_set_full_res.zip" # El zip que contiene el shapefile de todas las cuencas (full-res).
SHAPES_DIR = DATOS_DIR / "shapes" / "basin_set_full_res" # Carpeta donde se extrae el zip.
SHAPE_NAME = "HCDN_nhru_final_671.shp"# Nombre del shapefile principal dentro del zip.
# (De aquí se obtienen los polígonos de las cuencas).



def _extract_shapes() -> Path:
    """
    Asegura que el shapefile exista en disco.

    - Si ya está extraído, solo devuelve la ruta.
    - Si no existe, extrae el zip completo.
    """
    shp = SHAPES_DIR / SHAPE_NAME
     # Caso 1: ya existe el .shp extraído -> no hacemos nada.
    if shp.exists():
        return shp
    # Caso 2: no existe el zip -> no podemos seguir.
    if not SHAPES_ZIP.exists():
        raise FileNotFoundError(
            f"No se encontro {SHAPES_ZIP}. Ejecuta: python codigo/descargar_camels.py --shapes"
        )
    SHAPES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SHAPES_ZIP) as zf:
        zf.extractall(SHAPES_DIR)
    if not shp.exists():
        raise FileNotFoundError(f"No se encontro {SHAPE_NAME} tras extraer {SHAPES_ZIP}")
    return shp


def _shoelace_area(coords: list[tuple[float, float]]) -> float: 
    """
    Calcula el área “firmada” de un polígono usando la fórmula del cordón (shoelace).

    - coords debe ser una lista de puntos (x, y) donde x=lon, y=lat (o coordenadas en general).
    - Devuelve área/2; el signo depende del orden (horario/antihorario).

    Se usa porque:
    - Permite identificar el “anillo principal” (el de mayor área).
    - También se usa para calcular centroides.
    """
    if len(coords) < 3: # Con menos de 3 puntos no hay polígono.
        return 0.0
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:] + coords[:1]):
        area2 += x1 * y2 - x2 * y1
    return area2 / 2.0


def _polygon_centroid(coords: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Calcula el centroide (centro geométrico) de un polígono.

    - Si el área es 0 (polígono raro o degenerate), usa el promedio simple de puntos.
    - Si no, usa la fórmula estándar basada en el área (más correcta).

    Devuelve (cx, cy) en las mismas coordenadas del polígono.
    """
    area = _shoelace_area(coords)
    if area == 0:
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:] + coords[:1]):
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6.0 * area), cy / (6.0 * area))


def _rings_from_shape(shape: shapefile.Shape) -> list[list[tuple[float, float]]]:
    """
    Convierte una geometría del shapefile en una lista de “anillos” (rings).

    Un shapefile puede guardar:
    - un polígono con un anillo exterior
    - y varios anillos (huecos o partes separadas)

    Aquí:
    - shape.points trae todos los puntos seguidos
    - shape.parts indica desde qué índice empieza cada “parte”
    - Este método los separa y asegura que cada anillo quede cerrado.
    """
    pts = [(float(x), float(y)) for x, y in shape.points]
    parts = list(shape.parts) + [len(pts)]
    rings: list[list[tuple[float, float]]] = []
    for i in range(len(parts) - 1):
        ring = pts[parts[i] : parts[i + 1]]
        if not ring:
            continue
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        rings.append(ring)
    if not rings:
        raise ValueError("No se pudieron construir anillos de la geometria")
    return rings


def _find_basin_record(gauge_id: str) -> tuple[shapefile.Shape, dict[str, float | int | str]]:
    """
    Busca el polígono (shape) correspondiente a una cuenca específica (gauge_id).

    El shapefile tiene atributos (campos) como:
    - hru_id (identificador)
    - lon_cen / lat_cen (centroide guardado por CAMELS)

    Aquí usamos hru_id para hacer match con gauge_id.
    """
    shp_path = _extract_shapes()
    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    # En este shapefile, el ID suele ser numérico; por eso lo convertimos a int.
    target_int = int(str(gauge_id))
    # Revisamos si existen los campos clave (por si cambian según shapefile).
    idx_by_hru = fields.index("hru_id") if "hru_id" in fields else None
    idx_by_lon = fields.index("lon_cen") if "lon_cen" in fields else None
    idx_by_lat = fields.index("lat_cen") if "lat_cen" in fields else None

    chosen_idx = None
    # Recorremos todos los registros buscando el que tenga hru_id == gauge_id.
    for i, rec in enumerate(reader.records()):
        if idx_by_hru is not None:
            try:
                if int(rec[idx_by_hru]) == target_int:
                    chosen_idx = i
                    break
            except Exception:
                pass

    if chosen_idx is None:
        raise ValueError(f"No se encontro la cuenca {gauge_id} en {shp_path.name}")
    # Extraemos el registro y la geometría del índice encontrado.
    rec = reader.record(chosen_idx)
    shp = reader.shape(chosen_idx)

    attrs: dict[str, float | int | str] = {"gauge_id": str(gauge_id)}
    if idx_by_hru is not None:
        attrs["hru_id"] = int(rec[idx_by_hru])
    if idx_by_lon is not None:
        attrs["lon_cen"] = float(rec[idx_by_lon])
    if idx_by_lat is not None:
        attrs["lat_cen"] = float(rec[idx_by_lat])
    return shp, attrs


def gauge_coords_from_topo(gauge_id: str) -> tuple[float, float]:
    """
    Obtiene la lat/lon de la estación (gauge) desde camels_topo.txt.

    Esto es distinto al centroide de la cuenca:
    - aquí es la ubicación del medidor (USGS gauge).
    """
    topo = pd.read_csv(RAW_DIR / "camels_topo.txt", sep=";", dtype={"gauge_id": "string"})
    row = topo.loc[topo["gauge_id"].astype(str) == str(gauge_id)]
    if row.empty:
        raise ValueError(f"No existe gauge_id={gauge_id} en camels_topo.txt")
    return float(row.iloc[0]["gauge_lat"]), float(row.iloc[0]["gauge_lon"])


def gauge_name_from_camels(gauge_id: str) -> str:
    """
    Obtiene el nombre de la cuenca desde camels_name.txt.
    Sirve para mostrarlo en el mapa y en títulos.
    """
    names = pd.read_csv(RAW_DIR / "camels_name.txt", sep=";", dtype={"gauge_id": "string"})
    row = names.loc[names["gauge_id"].astype(str) == str(gauge_id)]
    if row.empty:
        return f"Gauge {gauge_id}"
    return str(row.iloc[0]["gauge_name"])


def _build_map_payload(gauge_id: str) -> dict:
    """
    Construye un “paquete” (payload) con todo lo necesario para:

    1) Dibujar la cuenca en Leaflet
    2) Colocar puntos (gauge, centroides, puntos representativos Daymet/NLDAS/Maurer)
    3) Ajustar el zoom automáticamente con bounds

    Este payload luego se usa para:
    - exportar archivos (GeoJSON, CSV, KML)
    - crear el HTML interactivo (Leaflet)
    """
    shape, attrs = _find_basin_record(gauge_id)
    # Convertimos la geometría del shapefile en anillos.
    rings = _rings_from_shape(shape)

    # Elegimos el anillo principal como el de mayor área absoluta.
    # Esto ayuda cuando hay varias partes o huecos.
    main_ring = max(rings, key=lambda r: abs(_shoelace_area(r)))
    # Centroide calculado directamente del polígono (geométrico).
    main_centroid_lon, main_centroid_lat = _polygon_centroid(main_ring)
     # Coordenadas del gauge (estación) desde topo.
    gauge_lat, gauge_lon = gauge_coords_from_topo(gauge_id)
    gauge_name = gauge_name_from_camels(gauge_id)

    # Para Leaflet necesitamos lat, lon (al revés de shapefile que es lon,lat).
    # boundary_latlon: lista de anillos; cada anillo es lista de [lat, lon].
    boundary_latlon = [[[pt[1], pt[0]] for pt in ring] for ring in rings]
    # GeoJSON del polígono principal (para usar como relleno bonito).
    basin_geojson = {
        "type": "Feature",
        "properties": {"gauge_id": str(gauge_id)},
        "geometry": {"type": "Polygon", "coordinates": [[list(pt) for pt in main_ring]]},
    }

    # Centroide “oficial” que viene en el shapefile CAMELS (si existe).
    # Si no existe, usamos el calculado.
    center_lat = float(attrs.get("lat_cen", main_centroid_lat))
    center_lon = float(attrs.get("lon_cen", main_centroid_lon))


    # Lista de puntos a mostrar en el mapa (marcadores).
    # Nota importante:
    # Daymet/Maurer/NLDAS aquí son “puntos representativos” solo para visualización,
    # porque los forzamientos que usas son promedio de cuenca, no estaciones puntuales reales.
    station_points = [
        {
            "name": "USGS Gauge",
            "kind": "gauge",
            "lat": gauge_lat,
            "lon": gauge_lon,
            "desc": "Estacion de caudal observada (USGS/CAMELS)",
        },
        {
            "name": "Basin centroid (geometry)",
            "kind": "basin_centroid",
            "lat": main_centroid_lat,
            "lon": main_centroid_lon,
            "desc": "Centroid computed from basin polygon",
        },
        {
            "name": "CAMELS centroid (lon_cen/lat_cen)",
            "kind": "camels_centroid",
            "lat": center_lat,
            "lon": center_lon,
            "desc": "Centroid from CAMELS shapefile",
        },
        {
            "name": "Daymet representative point",
            "kind": "daymet",
            "lat": center_lat + 0.006,
            "lon": center_lon - 0.006,
            "desc": "Representative point for basin-mean precipitation forcing",
        },
        {
            "name": "Maurer representative point",
            "kind": "maurer",
            "lat": center_lat,
            "lon": center_lon,
            "desc": "Representative point for basin-mean precipitation forcing",
        },
        {
            "name": "NLDAS representative point",
            "kind": "nldas",
            "lat": center_lat - 0.006,
            "lon": center_lon + 0.006,
            "desc": "Representative point for basin-mean precipitation forcing",
        },
    ]

    # bounds se usa para hacer fitBounds en Leaflet:
    # bbox viene como [xmin, ymin, xmax, ymax] = [lon_min, lat_min, lon_max, lat_max]
    return {
        "gauge_id": str(gauge_id),
        "gauge_name": gauge_name,
        "basin_geojson": basin_geojson,
        "boundary_latlon": boundary_latlon,
        "station_points": station_points,
        "bounds": [[shape.bbox[1], shape.bbox[0]], [shape.bbox[3], shape.bbox[2]]],
    }


def _boundary_geojson(payload: dict) -> dict:
    """
    Convierte el límite completo de la cuenca (todos los anillos) a un GeoJSON.

    Se usa MultiPolygon porque puede haber múltiples anillos/partes.
    """
    polygons = []
    for ring_latlon in payload["boundary_latlon"]:
        ring_lonlat = [[pt[1], pt[0]] for pt in ring_latlon]
        polygons.append([ring_lonlat])
    return {
        "type": "Feature",
        "properties": {"gauge_id": payload["gauge_id"], "source": "CAMELS full-res shapefile"},
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
    }


def _stations_geojson(payload: dict) -> dict:
    """
    Genera un GeoJSON tipo FeatureCollection con los puntos del mapa (marcadores).

    Esto permite:
    - abrirlos en QGIS/ArcGIS
    - usarlos como capa adicional en un mapa
    """
    features = []
    for p in payload["station_points"]:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": p["name"],
                    "kind": p["kind"],
                    "desc": p["desc"],
                    "gauge_id": payload["gauge_id"],
                },
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _write_kml(payload: dict, out_kml: Path) -> None:
    """
    Exporta un KML (para Google Earth) con:
    - el borde de la cuenca
    - los puntos (gauge y centroides)

    KML es útil porque muchos profes abren esto directo en Google Earth
    sin necesitar software GIS.
    """
    # Colores en formato KML (aabbggrr)
    style_map = {
        "gauge": "ff7fff00",
        "basin_centroid": "ffff00ff",
        "camels_centroid": "ff00aaff",
        "daymet": "ffffbf00",
        "maurer": "ffff5fff",
        "nldas": "ff00d7ff",
    }

    # Parte 1: polígonos (anillos)
    polygon_parts = []
    for i, ring_latlon in enumerate(payload["boundary_latlon"], start=1):
         # En KML las coordenadas son: lon,lat,alt
        coords = " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in ring_latlon)
        polygon_parts.append(
            f"""
    <Placemark>
      <name>Cuenca {escape(payload['gauge_id'])} - ring {i}</name>
      <styleUrl>#basinLine</styleUrl>
      <Polygon>
        <tessellate>1</tessellate>
        <outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs>
      </Polygon>
    </Placemark>"""
        )

    # Parte 2: puntos
    point_parts = []
    for p in payload["station_points"]:
        icon_color = style_map.get(str(p["kind"]), "ffffffff")
        point_parts.append(
            f"""
    <Placemark>
      <name>{escape(str(p['name']))}</name>
      <description>{escape(str(p['desc']))}</description>
      <Style>
        <IconStyle><color>{icon_color}</color><scale>1.0</scale></IconStyle>
      </Style>
      <Point><coordinates>{float(p['lon']):.8f},{float(p['lat']):.8f},0</coordinates></Point>
    </Placemark>"""
        )

    # Documento KML completo.
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Cuenca CAMELS {escape(payload['gauge_id'])}</name>
    <Style id="basinLine">
      <LineStyle><color>ff00ffff</color><width>3</width></LineStyle>
      <PolyStyle><color>3300ffff</color></PolyStyle>
    </Style>
    {''.join(polygon_parts)}
    {''.join(point_parts)}
  </Document>
</kml>
"""
    out_kml.parent.mkdir(parents=True, exist_ok=True)
    out_kml.write_text(kml, encoding="utf-8")


def export_basin_files(payload: dict, out_dir: Path) -> dict[str, Path]:
    """
    Exporta archivos “listos para usar” de la cuenca:

    - GeoJSON del límite completo (MultiPolygon)
    - GeoJSON de puntos (FeatureCollection)
    - CSV de puntos (por si alguien quiere verlos en tabla)
    - KML (Google Earth)

    Devuelve un diccionario con las rutas de salida generadas.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gauge_id = str(payload["gauge_id"])

    out_boundary_geojson = out_dir / f"{gauge_id}_cuenca.geojson"
    out_stations_geojson = out_dir / f"{gauge_id}_puntos_mapa.geojson"
    out_stations_csv = out_dir / f"{gauge_id}_puntos_mapa.csv"
    out_kml = out_dir / f"{gauge_id}_cuenca.kml"

    out_boundary_geojson.write_text(json.dumps(_boundary_geojson(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    out_stations_geojson.write_text(json.dumps(_stations_geojson(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(payload["station_points"]).to_csv(out_stations_csv, index=False)
    _write_kml(payload, out_kml)

    return {
        "boundary_geojson": out_boundary_geojson,
        "stations_geojson": out_stations_geojson,
        "stations_csv": out_stations_csv,
        "boundary_kml": out_kml,
    }


def make_leaflet_html(gauge_id: str, out_html: Path, payload: dict | None = None) -> None:
    """
    Genera un HTML auto-contenido con Leaflet, usando vista satelital,
    y dibuja:

    - relleno de la cuenca (GeoJSON del anillo principal)
    - límite detallado de la cuenca (polilíneas por anillo)
    - puntos (gauge, centroides, puntos representativos)

    El HTML queda listo para abrirse en navegador sin depender de Python.
    """
    payload = payload or _build_map_payload(gauge_id)
    # Creamos HTML completo. Aquí se incrusta el payload como JSON dentro del JS.
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Cuenca {payload['gauge_id']} (CAMELS)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; background: #000; }}
    .panel {{
      position: absolute; z-index: 9999; color: #fff;
      background: rgba(0,0,0,0.60);
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 8px; padding: 8px 10px;
      font-family: "Segoe UI", Roboto, Arial, sans-serif;
      font-size: 12px; line-height: 1.35;
      max-width: 360px;
    }}
    #coords {{ left: 12px; bottom: 12px; }}
    #legend {{ right: 12px; top: 12px; }}
    #basinName {{
      left: 50%; transform: translateX(-50%);
      top: 12px;
      font-weight: 700;
      letter-spacing: 0.2px;
      white-space: nowrap;
      max-width: 62vw;
      overflow: hidden;
      text-overflow: ellipsis;
      text-align: center;
    }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
    .leaflet-top.leaflet-left {{
      margin-top: 8px;
      margin-left: 8px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="basinName" class="panel"></div>
  <div id="coords" class="panel">Lat: -, Lon: -</div>
  <div id="legend" class="panel"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const payload = {json.dumps(payload)};

    const map = L.map('map', {{ zoomControl: true }});

    const sat = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ maxZoom: 19, attribution: 'Tiles © Esri' }}
    );
    const labels = L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ maxZoom: 19, attribution: 'Labels © Esri' }}
    );
    sat.addTo(map);
    labels.addTo(map);

    const fillLayer = L.geoJSON(payload.basin_geojson, {{
      style: () => ({{ color: '#00e5ff', weight: 3, fillColor: '#00bcd4', fillOpacity: 0.18 }})
    }}).addTo(map);

    const boundaryGroup = L.layerGroup().addTo(map);
    payload.boundary_latlon.forEach((ring, idx) => {{
      L.polyline(ring, {{ color: '#00ffff', weight: 4, opacity: 0.95 }}).addTo(boundaryGroup)
        .bindTooltip('Limite cuenca - anillo ' + (idx + 1));
      L.polyline(ring, {{ color: '#001a1f', weight: 7, opacity: 0.45 }}).addTo(boundaryGroup);
    }});

    const iconColors = {{
      gauge: '#00ff7f',
      basin_centroid: '#ff00ff',
      camels_centroid: '#ffaa00',
      daymet: '#00bfff',
      maurer: '#ff5fff',
      nldas: '#ffd700',
    }};

    const pointsGroup = L.layerGroup().addTo(map);
    payload.station_points.forEach((p) => {{
      const color = iconColors[p.kind] || '#ffffff';
      L.circleMarker([p.lat, p.lon], {{
        radius: 7,
        color: color,
        fillColor: color,
        fillOpacity: 0.95,
        weight: 2,
      }}).addTo(pointsGroup)
        .bindPopup(`<b>${{p.name}}</b><br/>Lat: ${{p.lat.toFixed(5)}}<br/>Lon: ${{p.lon.toFixed(5)}}<br/>${{p.desc}}`);
    }});

    L.control.layers(
      {{ 'Satelital': sat, 'Etiquetas': labels }},
      {{ 'Relleno cuenca': fillLayer, 'Limite cuenca': boundaryGroup, 'Puntos/coords': pointsGroup }},
      {{ collapsed: false, position: 'topleft' }}
    ).addTo(map);

    map.fitBounds(payload.bounds, {{ padding: [20, 20] }});

    map.on('mousemove', (e) => {{
      document.getElementById('coords').innerText = `Lat: ${{e.latlng.lat.toFixed(5)}}, Lon: ${{e.latlng.lng.toFixed(5)}}`;
    }});

    const legendRows = payload.station_points.map((p) => {{
      const color = iconColors[p.kind] || '#fff';
      return `<div><span class="dot" style="background:${{color}}"></span>${{p.name}}: (${{p.lat.toFixed(5)}}, ${{p.lon.toFixed(5)}})</div>`;
    }}).join('');

    document.getElementById('legend').innerHTML =
      `<div><b>Cuenca CAMELS ${{payload.gauge_id}}</b></div>` +
      `<div style="opacity:0.9; margin-top:2px;">${{payload.gauge_name}}</div>` +
      `<div style="margin-top:6px; margin-bottom:6px;">Limite: shapefile CAMELS (full-res)</div>` +
      legendRows +
      `<div style="margin-top:8px; opacity:0.85;">Nota: Daymet/Maurer/NLDAS son puntos representativos de forzamiento promedio de cuenca.</div>`;

    document.getElementById('basinName').innerText = payload.gauge_name + ' (' + payload.gauge_id + ')';
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    """
    Punto de entrada del script:

    1) Construye el payload de la cuenca (polígono + puntos + bounds)
    2) Exporta archivos GIS (GeoJSON/CSV/KML)
    3) Si no está --export-only, genera el HTML interactivo Leaflet
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge", default="02327100")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    payload = _build_map_payload(args.gauge)

    exported = export_basin_files(payload, DATOS_DIR / str(args.gauge))
    for key, path in exported.items():
        print(f"[ok] {key}: {path}")
    # Genera el HTML de Leaflet a menos que el usuario pida solo exportar archivos.
    if not args.export_only:
        out_html = FIG_HTML_DIR / f"{args.gauge}_mapa_leaflet.html"
        make_leaflet_html(args.gauge, out_html, payload=payload)
        print(f"[ok] Mapa guardado en {out_html}")


if __name__ == "__main__":
    main()
