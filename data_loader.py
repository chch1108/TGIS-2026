"""
data_loader.py
雲林縣淹水資料統一載入模組
提供雲林縣淹水模擬資料、人口資料和關鍵設施位置讀取功能
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# ─── 空間投影常數 (TWD97 / EPSG:3826 座標系常數) ───────────────────────────────────
GRID_XLL   = 162085.0   # 投影網格left-down角東向座標（單位為公尺）
GRID_YLL   = 2599890.0  # 投影網格left-down角北向座標（單位為公尺）
CELLSIZE   = 40.0       # 每個網格網元解析度大小（單位為公尺）
GRID_COLS  = 1379       # 網格總寬度（西向東網格數）
GRID_ROWS  = 980        # 網格總高度（北向南網格數）
NODATA_VAL = -999.999   # 網格資料中代表無資料預設值

# 大致 WGS84 經緯度範圍（用於地圖顯示和地理編碼邊界限制）
LON_MIN, LON_MAX = 120.002, 120.674
LAT_MIN, LAT_MAX = 23.490, 23.870

# 專案路徑設定
ROOT = Path(__file__).parent
DATA_DIR   = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "Output"
PICTURE_DIR= ROOT / "picture"

# ─── 颱風對照資料表 ─────────────────────────────────────────────────────────
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

# ─── 座標轉換工具函數 ─────────────────────────────────────────────────────
def twd97_to_wgs84(x_m: float, y_m: float):
    """
    將 TWD97 (EPSG:3826) 投影座標大略轉換為 WGS84 經緯度座標
    採用適用於雲林地區線性內插近似算法
    """
    lon = LON_MIN + (x_m - GRID_XLL) / (GRID_COLS * CELLSIZE) * (LON_MAX - LON_MIN)
    lat = LAT_MIN + (y_m - GRID_YLL) / (GRID_ROWS * CELLSIZE) * (LAT_MAX - LAT_MIN)
    return lon, lat

def pixel_to_wgs84(col: int, row: int):
    """
    將二維網格矩陣列和行座標（col，row）轉換為 WGS84 經緯度座標
    矩陣第 0 列代表最北端地理位置
    """
    x_m = GRID_XLL + col * CELLSIZE
    y_m = GRID_YLL + (GRID_ROWS - 1 - row) * CELLSIZE
    return twd97_to_wgs84(x_m, y_m)

def wgs84_to_pixel(lon: float, lat: float):
    """
    將 WGS84 經緯度座標反向投影為二維網格矩陣列和行座標（col，row）
    """
    col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_COLS)
    row = int((1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * GRID_ROWS)
    return col, row

def wgs84_to_pop_pixel(lon: float, lat: float, pop_shape=(40, 40)):
    """
    將 WGS84 經緯度座標投影為 40x40 人口網格座標並進行邊界限制
    """
    col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * pop_shape[1])
    row = int((1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * pop_shape[0])
    col = max(0, min(pop_shape[1]-1, col))
    row = max(0, min(pop_shape[0]-1, row))
    return col, row

# ─── 淹水模擬資料載入器 ───────────────────────────────────────────────────────
def load_typhoon_max(typhoon_id: int) -> np.ndarray:
    """
    載入特定颱風最大累積淹水深度矩陣（形狀為 980×1379）
    低於 0 數值會被校正為 0.0 代表無淹水
    """
    path = OUTPUT_DIR / f"Typhoon_{typhoon_id}_Max.npy"
    arr = np.load(str(path))
    arr[arr < 0] = 0.0
    return arr

def load_typhoon_3d(typhoon_id: int) -> np.ndarray:
    """
    載入特定颱風動態時間序列淹水深度資料（維度為 T×980×1379，T約為82小時）
    """
    path = OUTPUT_DIR / f"Typhoon_{typhoon_id}_3D.npz"
    return np.load(str(path))['data']

def load_typhoon_max_png(typhoon_id: int) -> str:
    """
    取得預先渲染最大淹水深度靜態地圖路徑
    """
    return str(PICTURE_DIR / f"Typhoon_{typhoon_id}_Max.png")

def load_typhoon_animation_gif(typhoon_id: int) -> str:
    """
    取得預先渲染淹水擴散動態 GIF 圖檔路徑
    """
    return str(PICTURE_DIR / f"Typhoon_{typhoon_id}_animation.gif")

# ─── 颱風路徑和基本資訊載入器 ─────────────────────────────────────────────────
def load_typhoon_info() -> pd.DataFrame:
    """
    載入歷年颱風基本統計資訊列表並進行欄位中文化重命名
    """
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
    """
    載入歷年颱風中心位置經緯度路徑資料
    """
    df = pd.read_csv(DATA_DIR / "typhoon_tracks.csv")
    df.columns = ['typhoon_id', 'date', 'lon', 'lat']
    return df

# ─── 關鍵基礎設施資料載入器 ────────────────────────────────────────────────────
def _parse_coord(coord_str: str):
    """
    解析經緯度字串（格式例如 '23.7083495,120.4384683'）為數值對 (lat，lon)
    """
    try:
        parts = str(coord_str).strip('"').split(',')
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None

def load_hospitals() -> pd.DataFrame:
    """
    讀取雲林縣醫院清單和座標資訊
    """
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
    """
    讀取雲林縣老人長照福利機構清單、收容容量和座標資訊
    """
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
    """
    讀取雲林縣電力變電所清單和座標資訊
    """
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
    """
    讀取雲林縣官方避難收容所清單、容量上限和座標資訊
    僅保留座標位在雲林縣經緯度範圍內合適收容所
    """
    path = ROOT / "雲林縣避難收容處所 - Sheet1.csv"
    df = pd.read_csv(path)
    df.columns = ['name', 'address', 'capacity', 'coord']
    df[['lat', 'lon']] = df['coord'].apply(
        lambda c: pd.Series(_parse_coord(c))
    )
    df = df.dropna(subset=['lat', 'lon'])
    df['type'] = 'shelter'
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').fillna(0)
    # 進行雲林縣合法範圍座標過濾
    df = df[
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
        (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX)
    ]
    return df[['name', 'address', 'capacity', 'lat', 'lon', 'type']]

def load_all_facilities() -> dict:
    """
    一鍵載入本系統所需所有關鍵設施位置資料
    """
    return {
        'hospitals':    load_hospitals(),
        'care_centers': load_care_centers(),
        'substations':  load_substations(),
        'shelters':     load_shelters(),
    }

# ─── 人口分佈資料載入功能 ───────────────────────────────────────────────────────
def load_population_grid() -> np.ndarray:
    """
    讀取 40×40 解析度人口密度矩陣
    負值代表無資料區域，在此將數值進行比例缩放（除以 45.9）
    以使加總人口數符合雲林縣實際設籍和居住人口（約 670000 人）
    """
    df = pd.read_csv(ROOT / "yunlin_population_40x40.csv", header=0)
    arr = df.values.astype(float)
    # 將正值區段除以 45.9 進行人口合理縮放
    mask = arr >= 0
    arr[mask] = arr[mask] / 45.9
    return arr   # 形狀為 (40, 40)；數值 -9999 代表超出縣境範圍

def population_at_point(lat: float, lon: float) -> float:
    """
    查詢特定經緯度點位所對應人口密度值
    """
    pop = load_population_grid()
    col, row = wgs84_to_pop_pixel(lon, lat)
    val = pop[row, col]
    return max(0.0, val) if val > 0 else 0.0

# ─── 淹水風險查詢及評估 ───────────────────────────────────────────────────────
def compute_flood_depth_at(lat: float, lon: float, flood_max: np.ndarray) -> float:
    """
    藉由經緯度轉換網格像素位置，從淹水最大深度矩陣查詢特定位置水深（公分）
    """
    col, row = wgs84_to_pixel(lon, lat)
    col = max(0, min(GRID_COLS-1, col))
    row = max(0, min(GRID_ROWS-1, row))
    depth = flood_max[row, col]
    return max(0.0, depth)

def classify_depth(depth_cm: float) -> tuple:
    """
    根據淹水深度數值判定淹水警戒等級以及對應地圖配色樣式
    """
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
    計算每項關鍵設施點位在特定颱風事件下預測水深和風險分級
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

# ─── 統計計算輔助函數 ───────────────────────────────────────────────────────
def compute_flood_statistics(typhoon_id: int) -> dict:
    """
    計算特定颱風在雲林縣全境所造成淹水面積、平均水深和高風險覆蓋情形等基本數據
    """
    flood = load_typhoon_max(typhoon_id)
    flooded = flood[flood > 0]
    pop = load_population_grid()
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
    print("=== Data Loader Sanity Check ===")
    info = load_typhoon_info()
    print(f"Typhoons: {len(info)}")
    fac = load_all_facilities()
    for k, v in fac.items():
        print(f"{k}: {len(v)} records")
    stats = compute_flood_statistics(1)
    print(f"\nTyphoon 1 stats: {stats}")
