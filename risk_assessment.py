"""
risk_assessment.py
風險評估模組 — Risk Assessment Module
計算人口暴露量、設施風險評分、收容所容量管理
"""

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from data_loader import (
    load_typhoon_max, load_population_grid,
    load_all_facilities, assess_facility_risks,
    GRID_ROWS, GRID_COLS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)


# ─── Population Exposure ──────────────────────────────────────────────────────
def compute_population_exposure(typhoon_id: int, threshold_cm: float = 30.0) -> dict:
    """
    Estimate population exposed to flooding above a threshold.
    Returns dict with exposure stats.
    """
    flood_max = load_typhoon_max(typhoon_id)
    pop_grid  = load_population_grid().astype(float)
    pop_grid[pop_grid < 0] = 0.0

    # Resize population grid (40×40) to flood grid (980×1379) via zoom
    zoom_r = GRID_ROWS / pop_grid.shape[0]
    zoom_c = GRID_COLS / pop_grid.shape[1]
    pop_resized = zoom(pop_grid, (zoom_r, zoom_c), order=1)
    # Each flood cell's population = pop_per_km2 * (40m*40m / 1e6 km2)
    cell_area_km2 = (40.0 ** 2) / 1e6
    pop_per_cell = pop_resized * cell_area_km2

    total_pop = float(pop_per_cell.sum())

    # Masks by severity
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


# ─── Facility Risk Scoring ────────────────────────────────────────────────────
FACILITY_WEIGHTS = {
    'hospitals':    5,   # Highest criticality
    'care_centers': 4,
    'substations':  3,
    'shelters':     2,
}

def score_facility_risk(depth_cm: float, weight: int) -> float:
    """
    Compute a 0-100 risk score for a facility.
    Depth thresholds (cm): 0→0, 30→30, 70→60, 100→80, 150→100
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
    Generate a complete risk report for all facilities.
    Returns DataFrame sorted by risk score descending.
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
        'shelters':     '🏕️ 收容所',
    }
    df['type_label'] = df['type'].map(type_labels)
    return df


# ─── Shelter Capacity Management ─────────────────────────────────────────────
def compute_shelter_utilization(
    typhoon_id: int,
    evacuation_factor: float = 0.05
) -> pd.DataFrame:
    """
    Simulate shelter utilization based on nearby population exposure.
    evacuation_factor: fraction of exposed population that evacuates.
    Returns shelters with assigned_pop and utilization_pct.
    """
    from data_loader import load_shelters, wgs84_to_pixel
    from scipy.spatial import cKDTree

    flood_max = load_typhoon_max(typhoon_id)
    pop_grid  = load_population_grid().astype(float)
    pop_grid[pop_grid < 0] = 0.0

    shelters = load_shelters().reset_index(drop=True)
    shelters = shelters[shelters['capacity'] > 0].reset_index(drop=True)

    # Build shelter KD-tree for nearest-neighbor assignment
    shelter_coords = shelters[['lat', 'lon']].values
    tree = cKDTree(shelter_coords)

    # Sample flooded population cells and assign to nearest shelter
    zoom_r = GRID_ROWS / pop_grid.shape[0]
    zoom_c = GRID_COLS / pop_grid.shape[1]
    pop_resized = zoom(pop_grid, (zoom_r, zoom_c), order=1)
    cell_area_km2 = (40.0 ** 2) / 1e6
    pop_per_cell = pop_resized * cell_area_km2

    # Get flooded cells
    flooded_mask = flood_max >= 30  # moderate+ flooding triggers evacuation
    fy_idxs, fx_idxs = np.where(flooded_mask)

    assigned = np.zeros(len(shelters), dtype=float)

    # Convert pixel coords to lat/lon for KD-tree query
    if len(fy_idxs) > 0:
        # Sample at most 50k cells for speed
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
        lambda u: '✅ 充裕' if u < 60 else ('⚠️ 接近飽和' if u < 90 else '🚨 超載')
    )
    return shelters.sort_values('utilization_pct', ascending=False)


# ─── Aggregate County Risk Index ─────────────────────────────────────────────
def compute_county_risk_index(typhoon_id: int) -> dict:
    """
    Compute a 0-100 composite risk index for the county under a typhoon.
    Components: flood area, population exposure, critical facility exposure
    """
    from data_loader import compute_flood_statistics, load_typhoon_info

    stats  = compute_flood_statistics(typhoon_id)
    pop_ex = compute_population_exposure(typhoon_id)
    info   = load_typhoon_info()
    typh   = info[info['id'] == typhoon_id].iloc[0]

    # Flood severity (0-40): based on area + max depth
    area_score  = min(40, stats['total_flooded_area_km2'] / 5)
    depth_score = min(20, stats['max_depth_cm'] / 5)

    # Population exposure (0-30)
    pop_score = min(30, pop_ex['exposure_rate_pct'])

    # Wind speed proxy (0-10)
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
