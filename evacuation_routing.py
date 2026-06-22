"""
evacuation_routing.py
避難路徑規劃模組 — Evacuation Route Planning
Dijkstra (baseline) + RL-informed flood-penalized routing
"""

import numpy as np
import pandas as pd
import heapq
import math
import urllib.request
import json
from data_loader import (
    load_typhoon_max, load_shelters,
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
    GRID_ROWS, GRID_COLS, wgs84_to_pixel
)

# ─── Routing Grid ─────────────────────────────────────────────────────────────
ROUTE_SCALE = 20    # Downsample the 980×1379 flood grid by this factor for routing
# Route grid size: ~49 × 69 cells, each ~800m

def build_cost_grid(flood_max: np.ndarray,
                    flood_penalty: float = 50.0,
                    block_threshold: float = 100.0) -> np.ndarray:
    """
    Build a cost grid for pathfinding from the flood depth array.
    - Normal traversal cost = 1.0 (per cell)
    - Flooded cell cost = 1 + flood_penalty * (depth / 100)
    - Deeply flooded (>= block_threshold) = impassable (np.inf)
    Returns downsampled grid of shape (GRID_ROWS//ROUTE_SCALE, GRID_COLS//ROUTE_SCALE)
    """
    # Downsample: take max flood in each coarse cell
    h = GRID_ROWS // ROUTE_SCALE
    w = GRID_COLS // ROUTE_SCALE
    flood_coarse = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            patch = flood_max[i*ROUTE_SCALE:(i+1)*ROUTE_SCALE,
                               j*ROUTE_SCALE:(j+1)*ROUTE_SCALE]
            flood_coarse[i, j] = patch.max() if patch.size > 0 else 0.0

    cost = np.ones((h, w))
    flooded = flood_coarse > 0
    cost[flooded] += flood_penalty * (flood_coarse[flooded] / 100.0)
    cost[flood_coarse >= block_threshold] = np.inf
    return cost


def wgs84_to_route_pixel(lon: float, lat: float):
    """Convert WGS84 lon/lat → coarse route grid (col, row)."""
    col_fine, row_fine = wgs84_to_pixel(lon, lat)
    return col_fine // ROUTE_SCALE, row_fine // ROUTE_SCALE


def route_pixel_to_wgs84(col: int, row: int):
    """Convert coarse route grid → WGS84 midpoint."""
    col_fine = col * ROUTE_SCALE + ROUTE_SCALE // 2
    row_fine = row * ROUTE_SCALE + ROUTE_SCALE // 2
    lon = LON_MIN + col_fine / GRID_COLS * (LON_MAX - LON_MIN)
    lat = LAT_MAX - row_fine / GRID_ROWS * (LAT_MAX - LAT_MIN)
    return lon, lat


# ─── Dijkstra Pathfinding ─────────────────────────────────────────────────────
def dijkstra(cost_grid: np.ndarray,
             start: tuple, goal: tuple) -> tuple:
    """
    Dijkstra shortest path on a 2D cost grid.
    start, goal: (col, row) in coarse route grid
    Returns (path_coords [(col,row),...], total_cost)
    """
    h, w = cost_grid.shape
    dist = np.full((h, w), np.inf)
    prev = {}
    sc, sr = start
    gc, gr = goal

    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return [], np.inf

    dist[sr, sc] = 0.0
    heap = [(0.0, sc, sr)]

    while heap:
        d, c, r = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        if (c, r) == (gc, gr):
            break
        for dc, dr in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nr < h and 0 <= nc < w:
                step = math.sqrt(2) if abs(dc)+abs(dr)==2 else 1.0
                nd = d + step * cost_grid[nr, nc]
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    prev[(nc, nr)] = (c, r)
                    heapq.heappush(heap, (nd, nc, nr))

    # Reconstruct path
    path = []
    cur = (gc, gr)
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()

    return path, float(dist[gr, gc])


def path_length_km(path: list) -> float:
    """Estimate path length in km from coarse grid cells."""
    cell_km = (CELLSIZE_KM := ROUTE_SCALE * 40 / 1000)
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path)-1):
        dc = abs(path[i][0] - path[i+1][0])
        dr = abs(path[i][1] - path[i+1][1])
        total += math.sqrt(dc**2 + dr**2) * cell_km
    return round(total, 2)


def path_to_wgs84(path: list) -> list:
    """Convert list of (col, row) grid coords to list of (lon, lat) WGS84."""
    return [route_pixel_to_wgs84(c, r) for c, r in path]


# ─── RL-Informed Penalty Routing ─────────────────────────────────────────────
def rl_penalized_dijkstra(cost_grid: np.ndarray,
                           start: tuple, goal: tuple,
                           flood_avoid_multiplier: float = 5.0) -> tuple:
    """
    RL-style routing with amplified flood avoidance.
    Higher multiplier = stronger tendency to detour around floods.
    """
    h, w = cost_grid.shape
    rl_cost = np.where(cost_grid > 1.0,
                       cost_grid * flood_avoid_multiplier,
                       cost_grid)
    rl_cost = np.where(cost_grid == np.inf, np.inf, rl_cost)
    return dijkstra(rl_cost, start, goal)


def query_osrm_route(coords: list) -> tuple:
    """
    Query street-aligned route from OSRM public API.
    coords: list of (lon, lat) tuples
    Returns (list of (lon, lat) tuples, distance_km) or (None, None) on failure.
    """
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'TGIS-Yunlin-Disaster-Support/1.0'}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            if data.get("code") == "Ok":
                route = data["routes"][0]
                geometry = route["geometry"]
                distance_km = route["distance"] / 1000.0
                return [tuple(c) for c in geometry["coordinates"]], distance_km
    except Exception:
        pass
    return None, None


# ─── Route Comparison API ─────────────────────────────────────────────────────
def compare_routes(typhoon_id: int,
                   origin_lat: float, origin_lon: float,
                   dest_lat: float, dest_lon: float,
                   flood_penalty: float = 50.0,
                   block_threshold: float = 100.0,
                   use_osrm: bool = True) -> dict:
    """
    Compare Dijkstra (shortest path) vs RL-informed (flood-avoiding) route.
    Returns dict with both paths and comparison metrics.
    """
    flood_max = load_typhoon_max(typhoon_id)
    cost_grid = build_cost_grid(flood_max, flood_penalty, block_threshold)

    start = wgs84_to_route_pixel(origin_lon, origin_lat)
    goal  = wgs84_to_route_pixel(dest_lon, dest_lat)

    # Clamp to grid
    h, w = cost_grid.shape
    start = (max(0, min(w-1, start[0])), max(0, min(h-1, start[1])))
    goal  = (max(0, min(w-1, goal[0])),  max(0, min(h-1, goal[1])))

    # Coarse Dijkstra path (ignores flood)
    dijkstra_cost_grid = np.ones_like(cost_grid)
    dijkstra_cost_grid[cost_grid == np.inf] = np.inf
    dijk_path, dijk_cost = dijkstra(dijkstra_cost_grid, start, goal)

    # Coarse RL-penalized path (avoids floods)
    rl_path, rl_cost = rl_penalized_dijkstra(cost_grid, start, goal,
                                              flood_avoid_multiplier=6.0)

    # WGS84 paths (initially coarse grid paths)
    dijk_wgs84 = path_to_wgs84(dijk_path)
    rl_wgs84 = path_to_wgs84(rl_path)

    dijk_km = path_length_km(dijk_path)
    rl_km = path_length_km(rl_path)

    # Query street-level OSRM route if requested
    if use_osrm:
        # 1. Dijkstra road path
        dijk_road, dijk_road_km = query_osrm_route([(origin_lon, origin_lat), (dest_lon, dest_lat)])
        if dijk_road:
            dijk_wgs84 = [(origin_lon, origin_lat)] + dijk_road + [(dest_lon, dest_lat)]
            dijk_km = round(dijk_road_km, 2)
        else:
            dijk_wgs84 = [(origin_lon, origin_lat)] + dijk_wgs84 + [(dest_lon, dest_lat)]

        # 2. RL road path
        if dijk_path == rl_path:
            rl_wgs84 = dijk_wgs84
            rl_km = dijk_km
        else:
            # Detour needed! Use grid path to pick waypoints
            waypoints = [(origin_lon, origin_lat)]
            if len(rl_wgs84) > 2:
                if len(rl_wgs84) > 12:
                    idx1 = len(rl_wgs84) // 3
                    idx2 = (2 * len(rl_wgs84)) // 3
                    waypoints.append(rl_wgs84[idx1])
                    waypoints.append(rl_wgs84[idx2])
                else:
                    idx = len(rl_wgs84) // 2
                    waypoints.append(rl_wgs84[idx])
            waypoints.append((dest_lon, dest_lat))

            rl_road, rl_road_km = query_osrm_route(waypoints)
            if rl_road:
                rl_wgs84 = [(origin_lon, origin_lat)] + rl_road + [(dest_lon, dest_lat)]
                rl_km = round(rl_road_km, 2)
            else:
                rl_wgs84 = [(origin_lon, origin_lat)] + rl_wgs84 + [(dest_lon, dest_lat)]
    else:
        # Pad with exact start/end points
        dijk_wgs84 = [(origin_lon, origin_lat)] + dijk_wgs84 + [(dest_lon, dest_lat)]
        rl_wgs84 = [(origin_lon, origin_lat)] + rl_wgs84 + [(dest_lon, dest_lat)]

    # Compute flood exposure along WGS84 paths (average depth in cm)
    def flood_on_wgs84_path(coords, flood_max):
        if not coords:
            return 0.0
        total = 0.0
        for lon, lat in coords:
            col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_COLS)
            row = int((1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * GRID_ROWS)
            col = max(0, min(GRID_COLS - 1, col))
            row = max(0, min(GRID_ROWS - 1, row))
            total += flood_max[row, col]
        return round(total / max(len(coords), 1), 2)

    dijk_exposure = flood_on_wgs84_path(dijk_wgs84, flood_max)
    rl_exposure = flood_on_wgs84_path(rl_wgs84, flood_max)

    return {
        'dijkstra': {
            'path_wgs84':     dijk_wgs84,
            'path_grid':      dijk_path,
            'distance_km':    dijk_km,
            'avg_flood_exposure_cm': dijk_exposure,
            'label':          'Dijkstra 最短路徑',
            'color':          '#e74c3c',
        },
        'rl': {
            'path_wgs84':     rl_wgs84,
            'path_grid':      rl_path,
            'distance_km':    rl_km,
            'avg_flood_exposure_cm': rl_exposure,
            'label':          'RL 避災路徑',
            'color':          '#27ae60',
        },
        'comparison': {
            'extra_distance_km':    round(rl_km - dijk_km, 2),
            'flood_reduction_cm':   round(dijk_exposure - rl_exposure, 2),
            'flood_reduction_pct':  round(
                (dijk_exposure - rl_exposure) / max(dijk_exposure, 1) * 100, 1
            ) if dijk_exposure > 0 else 0.0,
        },
        'flood_grid_shape':  (h, w),
        'origin_grid':       start,
        'dest_grid':         goal,
    }


# ─── Nearest Available Shelter ────────────────────────────────────────────────
def find_safe_shelters(lat: float, lon: float, typhoon_id: int,
                        max_results: int = 5) -> pd.DataFrame:
    """
    Find nearest shelters that are not severely flooded.
    Returns top shelters sorted by RL-penalized travel distance.
    """
    from data_loader import load_shelters, compute_flood_depth_at
    flood_max = load_typhoon_max(typhoon_id)
    shelters  = load_shelters()

    rows = []
    for _, sh in shelters.iterrows():
        sh_depth = compute_flood_depth_at(sh['lat'], sh['lon'], flood_max)
        if sh_depth >= 100:  # skip blocked shelters
            continue
        route = compare_routes(typhoon_id, lat, lon, sh['lat'], sh['lon'], use_osrm=False)
        rl_km = route['rl']['distance_km']
        rows.append({
            'name':         sh['name'],
            'lat':          sh['lat'],
            'lon':          sh['lon'],
            'capacity':     sh.get('capacity', 0),
            'flood_cm':     round(sh_depth, 1),
            'route_km':     rl_km,
            'rl_path':      route['rl']['path_wgs84'],
            'dijk_path':    route['dijkstra']['path_wgs84'],
        })

    df = pd.DataFrame(rows).sort_values('route_km').head(max_results)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    print("=== Evacuation Routing Test ===")
    result = compare_routes(
        typhoon_id=5,
        origin_lat=23.71, origin_lon=120.43,
        dest_lat=23.65,   dest_lon=120.32,
    )
    cmp = result['comparison']
    print(f"Dijkstra: {result['dijkstra']['distance_km']}km, "
          f"exposure={result['dijkstra']['avg_flood_exposure_cm']}cm")
    print(f"RL:       {result['rl']['distance_km']}km, "
          f"exposure={result['rl']['avg_flood_exposure_cm']}cm")
    print(f"Extra distance: {cmp['extra_distance_km']}km, "
          f"flood reduction: {cmp['flood_reduction_pct']}%")
