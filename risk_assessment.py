"""
risk_assessment.py
風險評估模組
計算人口暴露量、設施風險評量和收容所容量管理
"""

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from data_loader import (
    load_typhoon_max, load_population_grid,
    load_all_facilities, assess_facility_risks,
    GRID_ROWS, GRID_COLS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)


# ─── 人口暴露量評估 ──────────────────────────────────────────────────────
def compute_population_exposure(typhoon_id: int, threshold_cm: float = 30.0) -> dict:
    """
    估算受特定水深臨界值（公分）淹水威脅暴露人口數量
    傳回包含各淹水程度暴露人口及比率字典
    """
    flood_max = load_typhoon_max(typhoon_id)
    pop_grid  = load_population_grid().astype(float)
    pop_grid[pop_grid < 0] = 0.0

    # 使用線性插值將 40×40 人口網格放大映射至 980×1379 淹水網格
    zoom_r = GRID_ROWS / pop_grid.shape[0]
    zoom_c = GRID_COLS / pop_grid.shape[1]
    pop_resized = zoom(pop_grid, (zoom_r, zoom_c), order=1)
    # 計算每個網格像素實際居住人口：人口密度 * 網格面積（平方公里）
    cell_area_km2 = (40.0 ** 2) / 1e6
    pop_per_cell = pop_resized * cell_area_km2

    total_pop = float(pop_per_cell.sum())

    # 依據淹水水深進行災害程度分類遮罩
    mild_mask   = (flood_max > 0)   & (flood_max < 30)
    mod_mask    = (flood_max >= 30)  & (flood_max < 70)
    severe_mask = (flood_max >= 70)  & (flood_max < 100)
    extreme_mask= flood_max >= 100

    return {
        'total_pop_county':   int(total_pop),
        'exposed_mild':       int(pop_per_cell[mild_mask].sum()),
        'exposed_moderate':   int(pop_per_cell[mod_mask].sum()),
        'exposed_severe':     int(pop_per_cell[severe_mask].sum()),
        'exposed_extreme':    int(pop_per_cell[extreme_mask].sum()),
        'total_exposed':      int(pop_per_cell[flood_max >= threshold_cm].sum()),
        'exposure_rate_pct':  round(
            pop_per_cell[flood_max >= threshold_cm].sum() / max(total_pop, 1) * 100, 2
        ),
    }


# ─── 關鍵基礎設施風險評分 ────────────────────────────────────────────────────
FACILITY_WEIGHTS = {
    'hospitals':    5,   # 醫院（最高關鍵權重）
    'care_centers': 4,   # 長照機構
    'substations':  3,   # 變電所
    'shelters':     2,   # 避難收容所
}

def score_facility_risk(depth_cm: float, weight: int) -> float:
    """
    計算設施綜合風險評分（0 至 100 分範圍內）
    水深阻抗區間對照：0公分->0分，30公分->30分，70公分->60分，100公分->80分，150公分以上->100分
    """
    if depth_cm <= 0:
        depth_score = 0.0
    elif depth_cm < 30:
        depth_score = depth_cm / 30 * 30
    elif depth_cm < 70:
        depth_score = 30 + (depth_cm - 30) / 40 * 30
    elif depth_cm < 100:
        depth_score = 60 + (depth_cm - 70) / 30 * 20
    else:
        depth_score = min(100.0, 80 + (depth_cm - 100) / 50 * 20)
    return round(depth_score * weight / 5, 1)

def full_facility_risk_report(typhoon_id: int) -> pd.DataFrame:
    """
    產出特定颱風事件下全縣所有關鍵設施風險評估報表並按分數降冪排序
    """
    facilities = load_all_facilities()
    df = assess_facility_risks(typhoon_id, facilities)

    weights = {
        'hospitals':    5,
        'care_centers': 4,
        'substations':  3,
        'shelters':     2,
    }
    df['weight']     = df['type'].map(weights).fillna(1)
    df['risk_score'] = df.apply(
        lambda r: score_facility_risk(r['flood_depth_cm'], int(r['weight'])), axis=1
    )
    df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)

    type_labels = {
        'hospitals':    '🏥 醫院',
        'care_centers': '🏠 長照中心',
        'substations':  '⚡ 變電所',
        'shelters':     '🤍 收容所',
    }
    df['type_label'] = df['type'].map(type_labels)
    return df


# ─── 避難收容所容量管理和調度 ─────────────────────────────────────────────
def compute_shelter_utilization(
    typhoon_id: int,
    evacuation_factor: float = 0.05
) -> pd.DataFrame:
    """
    藉由空間鄰近分析（KD樹）模擬計算受淹水威脅人口前去最近避難所人流分配情形
    參數 evacuation_factor 為受災人口中前往收容所估算比率（預設為 5%）
    傳回包含收容容量、估算避難人數及容量使用率收容所資料表
    """
    from data_loader import load_shelters, wgs84_to_pixel
    from scipy.spatial import cKDTree

    flood_max = load_typhoon_max(typhoon_id)
    pop_grid  = load_population_grid().astype(float)
    pop_grid[pop_grid < 0] = 0.0

    shelters = load_shelters().reset_index(drop=True)
    shelters = shelters[shelters['capacity'] > 0].reset_index(drop=True)

    # 建置收容所空間 KD樹 以供快速檢索最鄰近站點
    shelter_coords = shelters[['lat', 'lon']].values
    tree = cKDTree(shelter_coords)

    # 將人口密度格網降採樣放大以與淹水網格尺寸一致
    zoom_r = GRID_ROWS / pop_grid.shape[0]
    zoom_c = GRID_COLS / pop_grid.shape[1]
    pop_resized = zoom(pop_grid, (zoom_r, zoom_c), order=1)
    cell_area_km2 = (40.0 ** 2) / 1e6
    pop_per_cell = pop_resized * cell_area_km2

    # 僅統計中度淹水（30 公分以上）區域人流疏散需求
    flooded_mask = flood_max >= 30
    fy_idxs, fx_idxs = np.where(flooded_mask)

    assigned = np.zeros(len(shelters), dtype=float)

    if len(fy_idxs) > 0:
        # 限制取樣數上限以防止網格計算過慢（最多 50000 個像素點）
        if len(fy_idxs) > 50000:
            idx = np.random.choice(len(fy_idxs), 50000, replace=False)
            fy_idxs, fx_idxs = fy_idxs[idx], fx_idxs[idx]

        lats = LAT_MAX - fy_idxs / GRID_ROWS * (LAT_MAX - LAT_MIN)
        lons = LON_MIN + fx_idxs / GRID_COLS * (LON_MAX - LON_MIN)
        pops = pop_per_cell[fy_idxs, fx_idxs] * evacuation_factor

        pts = np.column_stack([lats, lons])
        _, nearest = tree.query(pts, k=1)
        for i, sh_idx in enumerate(nearest):
            assigned[sh_idx] += pops[i]

    shelters = shelters.copy()
    shelters['estimated_arrivals'] = assigned.astype(int)
    shelters['utilization_pct'] = (
        shelters['estimated_arrivals'] / shelters['capacity'].replace(0, np.nan) * 100
    ).fillna(0).round(1)
    shelters['status'] = shelters['utilization_pct'].apply(
        lambda u: '🟢 充裕' if u < 60 else ('🟠 接近飽和' if u < 90 else '🔴 超載')
    )
    return shelters.sort_values('utilization_pct', ascending=False)


# ─── 縣市層級綜合風險指標計算 ─────────────────────────────────────────────
def compute_county_risk_index(typhoon_id: int) -> dict:
    """
    估算全縣在特定颱風事件下綜合風險指標（0 至 100 分範圍內）
    綜合考量：淹水面積、淹水深度、人口暴露比例及風速強度
    """
    from data_loader import compute_flood_statistics, load_typhoon_info

    stats  = compute_flood_statistics(typhoon_id)
    pop_ex = compute_population_exposure(typhoon_id)
    info   = load_typhoon_info()
    typh   = info[info['id'] == typhoon_id].iloc[0]

    # 1. 淹水範圍和水深指標（佔 60% 分數）
    area_score  = min(40, stats['total_flooded_area_km2'] / 5)
    depth_score = min(20, stats['max_depth_cm'] / 5)

    # 2. 人口暴露比例指標（佔 30% 分數）
    pop_score = min(30, pop_ex['exposure_rate_pct'])

    # 3. 風速指標（佔 10% 分數）
    wind_score = min(10, typh['wind_speed_ms'] / 6)

    total = area_score + depth_score + pop_score + wind_score

    return {
        'risk_index':      round(total, 1),
        'risk_grade':      'A' if total < 20 else ('B' if total < 40 else ('C' if total < 60 else 'D')),
        'risk_label':      '低風險' if total < 20 else ('中風險' if total < 40 else ('高風險' if total < 60 else '極高風險')),
        'area_score':      round(area_score, 1),
        'depth_score':     round(depth_score, 1),
        'pop_score':       round(pop_score, 1),
        'wind_score':      round(wind_score, 1),
        'flooded_area_km2': stats['total_flooded_area_km2'],
        'max_depth_cm':    stats['max_depth_cm'],
        'total_exposed':   pop_ex['total_exposed'],
        'exposure_rate':   pop_ex['exposure_rate_pct'],
    }


if __name__ == "__main__":
    print("=== Risk Assessment Test ===")
    for tid in [1, 5, 13]:
        idx = compute_county_risk_index(tid)
        print(f"Typhoon {tid}: Risk={idx['risk_index']}/100 ({idx['risk_label']})"
              f"  Area={idx['flooded_area_km2']}km²  Exposed={idx['total_exposed']:,}")
