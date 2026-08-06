# 雲林縣智慧救災系統 (TGIS) 專案交接及硬編碼 (Hardcoded) 參數整理

本文件彙整本專案中所有**固定（硬編碼）參數、地理常數、模型比例因子、第三方 API 及檔案路徑**。在交接給後續開發人員或移轉至其他縣市（如彰化、嘉義）部署時，必須特別注意以下內容：

---

## 1. 地理空間和網格常數 (Spatial & Coordinate Constants)
本專案網格座標轉換是針對**雲林縣區域**所設計線性映射，數值定義於 [data_loader.py](file:///Users/mac/Documents/antigravity/TGIS/data_loader.py)：

| 參數名稱 | 固定數值 | 說明 | 影響範圍 |
| :--- | :--- | :--- | :--- |
| `GRID_XLL` | `162085.0` | TWD97 投影網格左下角東向座標 (Easting) | 座標轉換及網格映射 |
| `GRID_YLL` | `2599890.0` | TWD97 投影網格左下角北向座標 (Northing) | 座標轉換及網格映射 |
| `CELLSIZE` | `40.0` | 原始淹水網格解析度（每個像素代表 40 公尺） | 距離及面積計算 |
| `GRID_COLS` | `1379` | 原始淹水資料網格寬度 (Columns) | 矩陣形狀及邊界限制 |
| `GRID_ROWS` | `980` | 原始淹水資料網格高度 (Rows) | 矩陣形狀及邊界限制 |
| `LON_MIN`, `LON_MAX` | `120.002`, `120.674` | 雲林縣經度邊界範圍（用於 Folium 地圖和地標搜尋限制） | 地圖顯示和地理編碼範圍 |
| `LAT_MIN`, `LAT_MAX` | `23.490`, `23.870` | 雲林縣緯度邊界範圍（用於 Folium 地圖和地標搜尋限制） | 地圖顯示和地理編碼範圍 |

> [!WARNING]
> **跨縣市移植注意**：若要將系統套用到其他縣市，必須重新計算該縣市網格大小 (`GRID_ROWS`, `GRID_COLS`)、邊界經緯度，以及左下角原點 (`GRID_XLL`, `GRID_YLL`)，否則座標對齊及地圖標記會完全錯位。

---

## 2. 演算法內部權重及比例因子 (Algorithm Weights & Scale Factors)

### A. 人口數值比例因子 (Population Scaling)
* **位置**：[data_loader.py](file:///Users/mac/Documents/antigravity/TGIS/data_loader.py#L201-L203) 的 `load_population_grid`。
* **固定數值**：除以 `45.9`。
* **說明**：由於原始人口網格數據尺度較大，為了讓全縣總加權人口符合雲林縣實際人口數（約 670,000 人），實作中將 `/ 45.9` 作為縮放因子，使避難收容所的容量估算和使用率回歸合理範圍。若更換人口數據源，需重新評估此係數。

### B. 設施脆弱度權重 (Facility Vulnerability Weights)
在社會風險評估和資源分配中，不同關鍵設施脆弱度權重有兩處實作：
* **風險評估模組**（[risk_assessment.py](file:///Users/mac/Documents/antigravity/TGIS/risk_assessment.py#L57-L62)）：
  ```python
  FACILITY_WEIGHTS = {
      'hospitals':    5,   # 醫院 (最高優先權)
      'care_centers': 4,   # 長照中心
      'substations':  3,   # 變電所
      'shelters':     2,   # 收容所
  }
  ```
* **論文模型設計**（[data_loader.py](file:///Users/mac/Documents/antigravity/TGIS/data_loader.py) 註釋與演算法）：
  * 醫院：`3.0`、長照中心：`2.5`、變電所：`2.0`、一般區域：`1.0`。
  * `risk_assessment.py` 中的 `score_facility_risk` 會將權重除以 5 進行歸一化，使其和論文權重設計保持等價性。

### C. 避難所收容估算比例 (Evacuation Factor)
* **位置**：[risk_assessment.py](file:///Users/mac/Documents/antigravity/TGIS/risk_assessment.py#L114) 的 `compute_shelter_utilization`。
* **固定數值**：`evacuation_factor = 0.05`。
* **說明**：系統預設處於淹水威脅（$\ge 30\text{ cm}$）受災人口中，有 **5%** 人會前往官方收容所避難。此比例在實務上可依據歷史災情調整。

### D. 路徑規劃降採樣比例 (Routing Scale)
* **位置**：[evacuation_routing.py](file:///Users/mac/Documents/antigravity/TGIS/evacuation_routing.py#L20)  `ROUTE_SCALE = 20`。
* **說明**：為提升路徑規劃演算速度，將 980×1379 淹水網格降採樣 20 倍（變為約 49×69 網格），使得單次 Dijkstra/RL 運算可以在數毫秒內完成。若需要更精細的路徑（如小巷弄），可調低此數值，但運算時間會隨之增加。

### E. 車庫預設座標 (Default Depot Coordinates)
* **位置**：[resource_allocation.py](file:///Users/mac/Documents/antigravity/TGIS/resource_allocation.py)  `GeneticAllocator` 及 `greedy_allocation` 參數預設值。
* **寫死數值**：`depot_lat = 23.71, depot_lon = 120.43`。
* **說明**：若前端 UI 未傳入車庫座標，系統會預設使用此經緯度作為抽水機車庫起點。在 [app.py](file:///Users/mac/Documents/antigravity/TGIS/app.py) 中，已優化為預設自動計算高風險區域幾何中心點，但後端程式碼中仍保留此預設值。

---

## 3. 第三方服務與外部 API 連線 (Third-Party APIs)
本專案依賴以下兩個公開免費 API 服務，有流量及回應時間限制：

### A. OpenStreetMap Nominatim 地理編碼 API
* **位置**：[app.py](file:///Users/mac/Documents/antigravity/TGIS/app.py#L359) `geocode_address`。
* **URL**：`https://nominatim.openstreetmap.org/search`
* **使用限額與硬編碼防護**：
  * 程式碼中硬編碼 `time.sleep(1.05)`，確保呼叫頻率低於 Nominatim 官方限制的每秒 1 次。
  * 設定了 `User-Agent: TGIS-Yunlin-Disaster-Support/1.0`。
  * 硬編碼限制搜尋範圍：`viewbox=120.002,23.870,120.674,23.490` 且 `bounded=1`，確保搜尋結果鎖定在雲林縣境內。

### B. OSRM 公開道路路由 API (Open Source Routing Machine)
* **位置**：[evacuation_routing.py](file:///Users/mac/Documents/antigravity/TGIS/evacuation_routing.py#L154)  `query_osrm_route`。
* **URL**：`http://router.project-osrm.org/route/v1/driving/`
* **說明**：用於取得實際道路經緯度軌跡和路網距離。
* **交接警訊**：該服務為公開免費伺服器，**無連線可用性保證**（Uptime Guarantee），且有速度限制。若要在生產環境部署，強烈建議建立**自建的本地端 OSRM Docker 容器**（使用台灣路網 `.osm.pbf` 檔案編譯），並將此 URL 改為本地端網址（例如 `http://localhost:5000/route/v1/driving/`），以確保系統穩定性並防範斷網。

---

## 4. 本地檔案與目錄結構 (Data Files & Folder Structure)
專案中的資料加載與輸出路徑，皆使用相對路徑 `Path(__file__).parent` 定義在 [data_loader.py](file:///Users/mac/Documents/antigravity/TGIS/data_loader.py)，避免絕對路徑造成執行失敗：

* **核心目錄**：
  * 原始資料集：`ROOT / "Dataset"`
  * 演算法及預測輸出：`ROOT / "Output"`
  * 預渲染圖像及動圖：`ROOT / "picture"`
* **可依照不同縣市調整檔案清單**：
  * 醫療機構：`雲林縣醫院名單_名稱地址 - 雲林縣醫院名單.csv`
  * 老人福利機構：`雲林縣老人福利機構名冊 - 工作表1.csv`
  * 變電所：`雲林縣變電所整理 - 雲林縣變電所.csv`
  * 避難收容處所：`雲林縣避難收容處所 - Sheet1.csv`
  * 人口資料：`yunlin_population_40x40.csv`
  * 颱風歷史列表：`Dataset/data.csv`
  * 颱風路徑資料：`Dataset/typhoon_tracks.csv`
  * 淹水模擬陣列：`Output/Typhoon_{id}_Max.npy` 和 `Output/Typhoon_{id}_3D.npz`
