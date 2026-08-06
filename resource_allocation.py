"""
resource_allocation.py
資源分配模組 (GA + ACO)
使用遺傳演算法 (GA) 和蟻群演算法 (ACO) 進行抽水機資源調度及路徑規劃
"""

import numpy as np
import pandas as pd
import random
import math
from data_loader import (
    load_typhoon_max, load_shelters,
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
)

# ─── 問題設定 ────────────────────────────────────────────────────────────
N_PUMPS = 10            # 可供調度移動式抽水機數量

def get_high_risk_zones(typhoon_id: int, top_k: int = 30) -> pd.DataFrame:
    """
    篩選出淹水最嚴重前 K 個高風險區域作為抽水機部署候選點
    傳回包含經緯度、淹水深度和加權人口 DataFrame
    """
    flood_max = load_typhoon_max(typhoon_id)
    from scipy.ndimage import zoom
    from data_loader import load_population_grid, GRID_ROWS, GRID_COLS

    pop = load_population_grid().astype(float)
    pop[pop < 0] = 0.0
    zoom_r = GRID_ROWS / pop.shape[0]
    zoom_c = GRID_COLS / pop.shape[1]
    pop_r = zoom(pop, (zoom_r, zoom_c), order=1)

    # 綜合評分：淹水深度 * 根號(加權人口 + 1)
    score = flood_max * np.sqrt(pop_r + 1)
    score[flood_max <= 0] = 0

    # 排序取得評分最高前 K 個區域
    flat_idx = np.argsort(score.ravel())[::-1][:top_k]
    rows_idx, cols_idx = np.unravel_index(flat_idx, score.shape)

    lats = LAT_MAX - rows_idx / GRID_ROWS * (LAT_MAX - LAT_MIN)
    lons = LON_MIN + cols_idx / GRID_COLS * (LON_MAX - LON_MIN)
    depths = flood_max[rows_idx, cols_idx]
    pops   = pop_r[rows_idx, cols_idx]
    scores = score[rows_idx, cols_idx]

    return pd.DataFrame({
        'zone_id': range(top_k),
        'lat': lats, 'lon': lons,
        'flood_depth_cm': depths.round(1),
        'population_density': pops.round(0),
        'priority_score': scores.round(1),
    })


def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    計算地球表面兩點半正矢距離（單位為公里）
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ─── 遺傳演算法 (GA) 資源分配 ────────────────────────────────────────────────
class GeneticAllocator:
    """
    基於遺傳演算法抽水機部署決策最佳化
    染色體設計：由區域索引組成串列（長度為 N_PUMPS） 允許重複選擇
    適應度函數：總覆蓋風險分數 / 總行駛距離
    """

    def __init__(self, zones_df: pd.DataFrame, n_pumps: int = N_PUMPS,
                 depot_lat: float = 23.71, depot_lon: float = 120.43,
                 pop_size: int = 100, generations: int = 80,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.7):
        self.zones  = zones_df.reset_index(drop=True)
        self.n      = n_pumps
        self.n_zones= len(zones_df)
        self.depot  = (depot_lat, depot_lon)
        self.pop_sz = pop_size
        self.gens   = generations
        self.mr     = mutation_rate
        self.cr     = crossover_rate
        self.history= []

    def _fitness(self, chrom):
        unique_zones = list(set(chrom))
        risk_covered = sum(self.zones.loc[z, 'priority_score'] for z in unique_zones)

        if not unique_zones:
            return 0.0

        # 以最近鄰啟發式演算法進行初步排序（以車庫為起點）
        unvisited = unique_zones[:]
        tour = []
        curr = self.depot
        while unvisited:
            nearest_z = min(
                unvisited,
                key=lambda z: haversine(
                    curr[0], curr[1],
                    self.zones.loc[z, 'lat'], self.zones.loc[z, 'lon']
                )
            )
            tour.append(nearest_z)
            curr = (self.zones.loc[nearest_z, 'lat'], self.zones.loc[nearest_z, 'lon'])
            unvisited.remove(nearest_z)

        total_dist = 0.0
        prev = self.depot
        for z in tour:
            loc = (self.zones.loc[z, 'lat'], self.zones.loc[z, 'lon'])
            total_dist += haversine(*prev, *loc)
            prev = loc
        # 採開路規劃：車輛最終停留在最後一個部署點 不返回起點車庫
        return risk_covered / max(total_dist, 1.0)

    def _random_chrom(self):
        return [random.randint(0, self.n_zones-1) for _ in range(self.n)]

    def _crossover(self, p1, p2):
        if random.random() > self.cr:
            return p1[:]
        pt = random.randint(1, self.n-1)
        return p1[:pt] + p2[pt:]

    def _mutate(self, chrom):
        return [
            (random.randint(0, self.n_zones-1) if random.random() < self.mr else g)
            for g in chrom
        ]

    def run(self):
        pop = [self._random_chrom() for _ in range(self.pop_sz)]
        best_chrom, best_fit = None, -np.inf

        for gen in range(self.gens):
            fits = [self._fitness(c) for c in pop]
            gen_best = max(fits)
            self.history.append(gen_best)

            if gen_best > best_fit:
                best_fit  = gen_best
                best_chrom= pop[np.argmax(fits)]

            # 競賽選汰機制
            new_pop = []
            for _ in range(self.pop_sz):
                a, b = random.sample(range(self.pop_sz), 2)
                winner = pop[a] if fits[a] >= fits[b] else pop[b]
                new_pop.append(winner)

            # 交配與突變產生子代
            children = []
            for i in range(0, self.pop_sz-1, 2):
                c1 = self._crossover(new_pop[i], new_pop[i+1])
                c2 = self._crossover(new_pop[i+1], new_pop[i])
                children += [self._mutate(c1), self._mutate(c2)]
            pop = children[:self.pop_sz]

        return best_chrom, best_fit

    def decode(self, chrom) -> pd.DataFrame:
        """
        將染色體解碼為具體可行部署方案表格
        """
        assignments = {}
        for pump_id, zone_id in enumerate(chrom):
            if zone_id not in assignments:
                assignments[zone_id] = []
            assignments[zone_id].append(pump_id)

        rows = []
        for zone_id, pumps in assignments.items():
            z = self.zones.loc[zone_id]
            rows.append({
                'zone_id': zone_id,
                'lat': z['lat'], 'lon': z['lon'],
                'flood_depth_cm': z['flood_depth_cm'],
                'priority_score': z['priority_score'],
                'pumps_assigned': len(pumps),
                'pump_ids': pumps,
                'dist_from_depot_km': round(
                    haversine(*self.depot, z['lat'], z['lon']), 2
                ),
            })
        return pd.DataFrame(rows).sort_values('priority_score', ascending=False)


# ─── 蟻群演算法 (ACO) 路徑最佳化 ───────────────────────────────────────────────
class ACORouter:
    """
    使用蟻群演算法進行多站點車輛路徑搜尋 (VRP)
    規劃抽水機從起點車庫出發 走訪完所有指派區域最短行進順序
    """
    def __init__(self, locations: list, depot_lat=23.71, depot_lon=120.43,
                 n_ants=30, iterations=50, alpha=1.0, beta=2.5,
                 evaporation=0.3, Q=100.0):
        self.depot = (depot_lat, depot_lon)
        all_locs = [self.depot] + [(r['lat'], r['lon']) for r in locations]
        self.locs = all_locs
        self.n    = len(all_locs)
        self.n_ants = n_ants
        self.iters  = iterations
        self.alpha  = alpha
        self.beta   = beta
        self.rho    = evaporation
        self.Q      = Q
        self.dist   = self._build_dist_matrix()
        self.pheromone = np.ones((self.n, self.n))
        self.history = []

    def _build_dist_matrix(self):
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    D[i,j] = haversine(*self.locs[i], *self.locs[j])
        return D

    def _ant_tour(self):
        visited = [False] * self.n
        tour = [0]
        visited[0] = True
        for _ in range(self.n - 1):
            cur = tour[-1]
            probs = []
            for j in range(self.n):
                if not visited[j]:
                    tau = self.pheromone[cur,j] ** self.alpha
                    eta = (1.0 / max(self.dist[cur,j], 1e-6)) ** self.beta
                    probs.append((j, tau * eta))
            total = sum(p for _, p in probs)
            if total == 0:
                nxt = random.choice([j for j,_ in probs])
            else:
                r = random.random() * total
                cum = 0.0
                nxt = probs[-1][0]
                for j, p in probs:
                    cum += p
                    if cum >= r:
                        nxt = j
                        break
            tour.append(nxt)
            visited[nxt] = True
        # 採開路規劃：車輛最終停留在最後一個部署點 不返回起點車庫
        return tour

    def _tour_length(self, tour):
        return sum(self.dist[tour[i], tour[i+1]] for i in range(len(tour)-1))

    def run(self):
        best_tour, best_len = None, np.inf
        for it in range(self.iters):
            tours = [self._ant_tour() for _ in range(self.n_ants)]
            lens  = [self._tour_length(t) for t in tours]
            it_best_idx = np.argmin(lens)
            if lens[it_best_idx] < best_len:
                best_len  = lens[it_best_idx]
                best_tour = tours[it_best_idx]
            self.history.append(best_len)
            # 蒸發舊有外激素
            self.pheromone *= (1 - self.rho)
            # 依據路徑長度釋放新外激素
            for t, l in zip(tours, lens):
                dep = self.Q / max(l, 1e-6)
                for i in range(len(t)-1):
                    self.pheromone[t[i], t[i+1]] += dep
                    self.pheromone[t[i+1], t[i]] += dep
        return best_tour, best_len


# ─── 貪婪對比基準演算法 ──────────────────────────────────────────────────────────
def greedy_allocation(zones_df: pd.DataFrame, n_pumps: int = N_PUMPS,
                      depot_lat: float = 23.71, depot_lon: float = 120.43) -> dict:
    """
    簡單貪婪演算法：將所有抽水機指派至淹水最深前幾名候選點
    作為和 GA+ACO 演算法對比基準
    """
    top = zones_df.nlargest(n_pumps, 'flood_depth_cm').reset_index(drop=True)
    total_dist = 0.0
    prev = (depot_lat, depot_lon)
    for _, row in top.iterrows():
        total_dist += haversine(*prev, row['lat'], row['lon'])
        prev = (row['lat'], row['lon'])
    # 採開路規劃：不返回起點車庫

    # 串接 OSRM 以取得精確道路距離
    try:
        from evacuation_routing import query_osrm_route
        greedy_coords = [(depot_lon, depot_lat)] + [(r['lon'], r['lat']) for _, r in top.iterrows()]
        _, road_dist_km = query_osrm_route(greedy_coords)
        if road_dist_km is not None:
            total_dist = road_dist_km
    except Exception:
        pass

    risk_covered = top['priority_score'].sum()
    return {
        'plan': top,
        'total_distance_km': round(total_dist, 2),
        'risk_covered': round(risk_covered, 1),
        'method': 'Greedy (Depth-First)',
    }


# ─── 主要排程介面 API ─────────────────────────────────────────────────────────
def run_resource_allocation(typhoon_id: int,
                             n_pumps: int = N_PUMPS,
                             depot_lat: float = None,
                             depot_lon: float = None,
                             ga_generations: int = 80,
                             progress_cb=None) -> dict:
    """
    執行資源調度完整流程：篩選高風險區域 -> GA進行資源分配 -> ACO進行路徑規劃
    傳回包含 GA 方案、ACO 路徑、Greedy 方案和雙方對比指標字典
    """
    # 1. 篩選出 25 個風險熱點
    zones = get_high_risk_zones(typhoon_id, top_k=25)

    # 若未指定車庫位置，預設為所有候選點幾何中心
    if depot_lat is None or depot_lon is None:
        depot_lat = float(zones['lat'].mean())
        depot_lon = float(zones['lon'].mean())

    # 2. 計算貪婪演算法基準
    greedy = greedy_allocation(zones, n_pumps, depot_lat, depot_lon)
    if progress_cb: progress_cb(20, "Greedy baseline computed...")

    # 3. 執行遺傳演算法
    ga = GeneticAllocator(
        zones, n_pumps=n_pumps, depot_lat=depot_lat, depot_lon=depot_lon,
        pop_size=80, generations=ga_generations, mutation_rate=0.15
    )
    best_chrom, best_fit = ga.run()
    ga_plan = ga.decode(best_chrom)
    if progress_cb: progress_cb(60, "GA allocation complete...")

    # 4. 對 GA 指派站點執行蟻群路徑最佳化
    zone_list = ga_plan.to_dict('records')
    aco = ACORouter(zone_list, depot_lat=depot_lat, depot_lon=depot_lon,
                    n_ants=25, iterations=40)
    best_tour, best_len = aco.run()
    if progress_cb: progress_cb(90, "ACO routing complete...")

    # 將 ACO 結果轉換為軌跡座標
    route_coords = [aco.locs[i] for i in best_tour]
    total_ga_dist = best_len

    # 串接 OSRM 取得實際路網幾何路徑
    try:
        from evacuation_routing import query_osrm_route
        osrm_coords = [(aco.locs[i][1], aco.locs[i][0]) for i in best_tour]
        road_coords, road_dist_km = query_osrm_route(osrm_coords)
        if road_coords:
            route_coords = [(c[1], c[0]) for c in road_coords]
            total_ga_dist = road_dist_km
    except Exception:
        pass

    # 計算對比與改善指標
    ga_risk = ga_plan['priority_score'].sum()
    comparison = {
        'ga_distance_km':     round(total_ga_dist, 2),
        'greedy_distance_km': greedy['total_distance_km'],
        'distance_reduction_pct': round(
            (greedy['total_distance_km'] - total_ga_dist) /
            max(greedy['total_distance_km'], 1) * 100, 1
        ),
        'ga_risk_covered':     round(ga_risk, 1),
        'greedy_risk_covered': greedy['risk_covered'],
        'risk_improvement_pct': round(
            (ga_risk - greedy['risk_covered']) /
            max(greedy['risk_covered'], 1) * 100, 1
        ),
    }
    if progress_cb: progress_cb(100, "Done!")

    return {
        'zones':          zones,
        'ga_plan':        ga_plan,
        'ga_history':     ga.history,
        'aco_route':      route_coords,
        'aco_history':    aco.history,
        'greedy_plan':    greedy['plan'],
        'comparison':     comparison,
        'depot':          (depot_lat, depot_lon),
    }


if __name__ == "__main__":
    print("=== Resource Allocation Test (Typhoon 5) ===")
    result = run_resource_allocation(5, n_pumps=10, ga_generations=50,
                                      progress_cb=lambda p, m: print(f"  [{p}%] {m}"))
    cmp = result['comparison']
    print(f"Distance: GA={cmp['ga_distance_km']}km  Greedy={cmp['greedy_distance_km']}km  "
          f"Reduction={cmp['distance_reduction_pct']}%")
    print(f"Risk: GA={cmp['ga_risk_covered']}  Greedy={cmp['greedy_risk_covered']}  "
          f"Improvement={cmp['risk_improvement_pct']}%")
