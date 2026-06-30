"""
data_loader.py
雲林縣淹水資料統一載入模組
Unified data loading for Yunlin Flood Disaster System
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# ─── Spatial Constants (TWD97 / EPSG:3826) ───────────────────────────────────
GRID_XLL   = 162085.0   # metres East (TWD97)
GRID_YLL   = 2599890.0  # metres North (TWD97)
CELLSIZE   = 40.0       # metres per pixel
GRID_COLS  = 1379       # width  (West→East)
GRID_ROWS  = 980        # height (North→South in array indexing)
NODATA_VAL = -999.999

# Approx WGS84 bounding box (for Folium / display)
LON_MIN, LON_MAX = 120.002, 120.674
LAT_MIN, LAT_MAX = 23.490, 23.870

# Project root
ROOT = Path(__file__).parent
DATA_DIR   = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "Output"
PICTURE_DIR= ROOT / "picture"

# ─── Typhoon metadata ─────────────────────────────────────────────────────────
TYPHOON_NAMES = {
    1:  "納莉 2001",
    2:  "敏督利 2004",
    3:  "海棠 2005",
    4:  "辛樂克 2008",
    5:  "莫拉克 2009",
    6:  "蘇拉 2012",
    7:  "蘇力 2013",
    8:  "蘇迪勒 2015",
    9:  "梅姬 2016",
    10: "尼莎 2017",
    11: "海棠 2017",
    12: "盧碧 2021",
    13: "杜蘇芮 2023",
}

INTENSITY_LABELS = {1: "輕度颱風", 2: "中度颱風", 3: "強烈颱風"}

# ─── Coordinate utilities ─────────────────────────────────────────────────────
def twd97_to_wgs84(x_m: float, y_m: float):
    """Approximate TWD97 (EPSG:3826) → WGS84 lon/lat conversion."""
    # Simple linear approximation calibrated on Yunlin region
    lon = LON_MIN + (x_m - GRID_XLL) / (GRID_COLS * CELLSIZE) * (LON_MAX - LON_MIN)
    lat = LAT_MIN + (y_m - GRID_YLL) / (GRID_ROWS * CELLSIZE) * (LAT_MAX - LAT_MIN)
    return lon, lat

def pixel_to_wgs84(col: int, row: int):
    """Convert (col, row) in the merged flood array → WGS84 lon/lat."""
    x_m = GRID_XLL + col * CELLSIZE
    y_m = GRID_YLL + (GRID_ROWS - 1 - row) * CELLSIZE   # row 0 = top (North)
    return twd97_to_wgs84(x_m, y_m)

def wgs84_to_pixel(lon: float, lat: float):
    """Convert WGS84 lon/lat → (col, row) in merged flood array."""
    col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_COLS)
    row = int((1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * GRID_ROWS)
    return col, row

def wgs84_to_pop_pixel(lon: float, lat: float, pop_shape=(40, 40)):
    """Convert WGS84 → population grid (40x40) pixel."""
    col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * pop_shape[1])
    row = int((1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * pop_shape[0])
    col = max(0, min(pop_shape[1]-1, col))
    row = max(0, min(pop_shape[0]-1, row))
    return col, row

# ─── Flood data loaders ───────────────────────────────────────────────────────
def load_typhoon_max(typhoon_id: int) -> np.ndarray:
    """Load merged max flood depth array (980×1379). Returns 0 where no flood."""
    path = OUTPUT_DIR / f"Typhoon_{typhoon_id}_Max.npy"
    arr = np.load(str(path))
    arr[arr < 0] = 0.0
    return arr

def load_typhoon_3d(typhoon_id: int) -> np.ndarray:
    """Load time-series flood array (T×980×1379). T≈82 (hourly)."""
    path = OUTPUT_DIR / f"Typhoon_{typhoon_id}_3D.npz"
    return np.load(str(path))['data']

def load_typhoon_max_png(typhoon_id: int) -> str:
    """Return path to pre-rendered max flood PNG."""
    return str(PICTURE_DIR / f"Typhoon_{typhoon_id}_Max.png")

def load_typhoon_animation_gif(typhoon_id: int) -> str:
    """Return path to pre-rendered animation GIF."""
    return str(PICTURE_DIR / f"Typhoon_{typhoon_id}_animation.gif")

# ─── Typhoon metadata ─────────────────────────────────────────────────────────
def load_typhoon_info() -> pd.DataFrame:
    """Load typhoon catalog (data.csv)."""
    df = pd.read_csv(DATA_DIR / "data.csv", encoding='utf-8')
    df = df.rename(columns={
        'typhoon_name': 'id',
        '年分': 'year',
        '登陸時間': 'landfall_time',
        '近臺強度': 'intensity',
        '近臺最低氣壓(hPa)': 'pressure_hpa',
        '近臺最大風速(m/s)': 'wind_speed_ms',
        '近臺7級風暴風半徑(km)': 'radius_7_km',
        '近臺10級風暴風半徑(km)': 'radius_10_km',
    })
    df['name'] = df['id'].map(TYPHOON_NAMES)
    df['intensity_label'] = df['intensity'].map(INTENSITY_LABELS)
    return df

def load_typhoon_tracks() -> pd.DataFrame:
    """Load typhoon track data."""
    df = pd.read_csv(DATA_DIR / "typhoon_tracks.csv")
    df.columns = ['typhoon_id', 'date', 'lon', 'lat']
    return df

# ─── Facility data loaders ────────────────────────────────────────────────────
def _parse_coord(coord_str: str):
    """Parse '23.7083495,120.4384683' → (lat, lon)."""
    try:
        parts = str(coord_str).strip('"').split(',')
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None

def load_hospitals() -> pd.DataFrame:
    """Load hospital locations."""
    path = ROOT / "雲林縣醫院名單_名稱地址 - 雲林縣醫院名單.csv"
    df = pd.read_csv(path)
    df.columns = ['name', 'address', 'coord']
    df[['lat', 'lon']] = df['coord'].apply(
        lambda c: pd.Series(_parse_coord(c))
    )
    df = df.dropna(subset=['lat', 'lon'])
    df['type'] = 'hospital'
    return df[['name', 'address', 'lat', 'lon', 'type']]

def load_care_centers() -> pd.DataFrame:
    """Load long-term care center locations."""
    path = ROOT / "雲林縣老人福利機構名冊 - 工作表1.csv"
    df = pd.read_csv(path)
    df.columns = ['name', 'address', 'capacity', 'coord']
    df[['lat', 'lon']] = df['coord'].apply(
        lambda c: pd.Series(_parse_coord(c))
    )
    df = df.dropna(subset=['lat', 'lon'])
    df['type'] = 'care_center'
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').fillna(0)
    return df[['name', 'address', 'capacity', 'lat', 'lon', 'type']]

def load_substations() -> pd.DataFrame:
    """Load electrical substation locations."""
    path = ROOT / "雲林縣變電所整理 - 雲林縣變電所.csv"
    df = pd.read_csv(path)
    df.columns = ['name', 'address', 'coord']
    df[['lat', 'lon']] = df['coord'].apply(
        lambda c: pd.Series(_parse_coord(c))
    )
    df = df.dropna(subset=['lat', 'lon'])
    df['type'] = 'substation'
    return df[['name', 'address', 'lat', 'lon', 'type']]

def load_shelters() -> pd.DataFrame:
    """Load evacuation shelter locations."""
    path = ROOT / "雲林縣避難收容處所 - Sheet1.csv"
    df = pd.read_csv(path)
    df.columns = ['name', 'address', 'capacity', 'coord']
    df[['lat', 'lon']] = df['coord'].apply(
        lambda c: pd.Series(_parse_coord(c))
    )
    df = df.dropna(subset=['lat', 'lon'])
    df['type'] = 'shelter'
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').fillna(0)
    # Filter to valid Yunlin coordinates
    df = df[
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
        (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX)
    ]
    return df[['name', 'address', 'capacity', 'lat', 'lon', 'type']]

def load_all_facilities() -> dict:
    """Load all facility DataFrames in one call."""
    return {
        'hospitals':    load_hospitals(),
        'care_centers': load_care_centers(),
        'substations':  load_substations(),
        'shelters':     load_shelters(),
    }

# ─── Population data ──────────────────────────────────────────────────────────
def load_population_grid() -> np.ndarray:
    """Load 40×40 population grid. Returns float array, -9999 = nodata."""
    df = pd.read_csv(ROOT / "yunlin_population_40x40.csv", header=0)
    arr = df.values.astype(float)
    # Scale positive values to match Yunlin County's real population of ~670,000 (approx 1/45.9 scaling factor)
    mask = arr >= 0
    arr[mask] = arr[mask] / 45.9
    return arr   # shape (40, 40); -9999 = outside Yunlin

def population_at_point(lat: float, lon: float) -> float:
    """Query population at a WGS84 point."""
    pop = load_population_grid()
    col, row = wgs84_to_pop_pixel(lon, lat)
    val = pop[row, col]
    return max(0.0, val) if val > 0 else 0.0

# ─── Risk Assessment ──────────────────────────────────────────────────────────
def compute_flood_depth_at(lat: float, lon: float, flood_max: np.ndarray) -> float:
    """Get flood depth (cm) at a geographic point from flood_max grid."""
    col, row = wgs84_to_pixel(lon, lat)
    col = max(0, min(GRID_COLS-1, col))
    row = max(0, min(GRID_ROWS-1, row))
    depth = flood_max[row, col]
    return max(0.0, depth)

def classify_depth(depth_cm: float) -> tuple:
    """Return (warning_level, color) for a flood depth in cm."""
    if depth_cm <= 0:
        return "無淹水", "#27ae60"
    elif depth_cm < 30:
        return "輕微 (<30cm)", "#f1c40f"
    elif depth_cm < 70:
        return "中度 (30-70cm)", "#e67e22"
    elif depth_cm < 100:
        return "嚴重 (70-100cm)", "#e74c3c"
    else:
        return "極嚴重 (>100cm)", "#8e44ad"

def assess_facility_risks(typhoon_id: int, facilities: dict) -> pd.DataFrame:
    """
    For each facility point, compute flood depth and risk level.
    Returns a unified DataFrame with risk info.
    """
    flood_max = load_typhoon_max(typhoon_id)
    rows = []
    for ftype, df in facilities.items():
        for _, row in df.iterrows():
            depth = compute_flood_depth_at(row['lat'], row['lon'], flood_max)
            level, color = classify_depth(depth)
            rows.append({
                'type': ftype,
                'name': row['name'],
                'lat': row['lat'],
                'lon': row['lon'],
                'flood_depth_cm': round(depth, 1),
                'risk_level': level,
                'risk_color': color,
                'capacity': row.get('capacity', None),
            })
    return pd.DataFrame(rows)

# ─── Statistics helpers ───────────────────────────────────────────────────────
def compute_flood_statistics(typhoon_id: int) -> dict:
    """Compute summary statistics for a typhoon's max flood."""
    flood = load_typhoon_max(typhoon_id)
    flooded = flood[flood > 0]
    pop = load_population_grid()
    # Resize pop to flood grid for overlap (simple nearest-neighbor)
    pop_valid = pop.copy()
    pop_valid[pop_valid < 0] = 0

    return {
        'total_flooded_cells': int((flood > 0).sum()),
        'total_flooded_area_km2': round(int((flood > 0).sum()) * (CELLSIZE**2) / 1e6, 1),
        'max_depth_cm': round(float(flooded.max()), 1) if len(flooded) else 0,
        'mean_depth_cm': round(float(flooded.mean()), 1) if len(flooded) else 0,
        'severe_cells': int((flood >= 100).sum()),
        'severe_area_km2': round(int((flood >= 100).sum()) * (CELLSIZE**2) / 1e6, 1),
        'total_population': int(pop_valid.sum()),
    }

if __name__ == "__main__":
    # Quick sanity check
    print("=== Data Loader Sanity Check ===")
    info = load_typhoon_info()
    print(f"Typhoons: {len(info)}")
    fac = load_all_facilities()
    for k, v in fac.items():
        print(f"{k}: {len(v)} records")
    stats = compute_flood_statistics(1)
    print(f"\nTyphoon 1 stats: {stats}")
