# 雲林縣智慧救災決策支援系統

本專案是一套以雲林縣為應用範圍的智慧防災決策平台。系統整合淹水模擬資料、人口分布、重要設施、避難收容所、資源分配演算法及避災路徑規劃功能。操作介面使用 Streamlit 建置。

## 系統功能

### 淹水總覽

- 顯示淹水面積、最大淹水深度、受影響人口、人口暴露率及縣市風險指數
- 使用互動式地圖呈現淹水熱點、颱風路徑及重要設施
- 顯示淹水深度分布及人口暴露分布
- 播放逐時淹水演進 GIF

### 設施風險評估

- 評估醫院、長照中心、變電所及避難收容所的淹水風險
- 顯示受影響設施數量及高風險設施清單
- 提供設施類型篩選及風險分數分布圖

### 資源分配

- 使用遺傳演算法 GA 分配抽水機
- 使用蟻群演算法 ACO 規劃部署路線
- 顯示高優先度淹水區域、演算法收斂曲線、部署地圖及方法比較
- 可調整抽水機數量、演化代數及車庫位置

### 避難路徑規劃

- 支援輸入雲林縣內地址或地標
- 使用 OpenStreetMap Nominatim 將地址轉換為座標
- 比較 Dijkstra 最短路徑及 RL 避災路徑
- 顯示路徑距離、平均淹水暴露及暴露降低比例

### 收容所管理

- 計算收容所預估到達人數及容量使用率
- 標示超載收容所及容量充裕收容所
- 顯示收容所狀態清單及使用率分布

### 自訂颱風

颱風事件選單提供自訂功能。使用者可輸入以下資料：

- 颱風名稱
- 年份
- 登陸時間
- 近臺強度
- 近臺最低氣壓
- 近臺最大風速
- 7 級暴風半徑
- 10 級暴風半徑
- 淹水情境參考

目前淹水分析依賴預先建立的淹水網格。自訂颱風需選擇一組既有事件作為淹水情境參考。預設參考事件為莫拉克颱風。

## 使用技術

- Python 3.12
- Streamlit
- NumPy
- Pandas
- SciPy
- Folium
- Streamlit Folium
- Plotly
- Matplotlib
- Pillow
- OpenStreetMap Nominatim

## 專案結構

```text
TGIS/
├─ app.py
├─ data_loader.py
├─ risk_assessment.py
├─ resource_allocation.py
├─ evacuation_routing.py
├─ requirement.txt
├─ Dataset/
│  ├─ data.csv
│  ├─ typhoon_tracks.csv
│  └─ yunlin_population_40x40.csv
├─ Output/
│  ├─ Typhoon_*_Max.npy
│  ├─ Typhoon_*_3D.npz
│  └─ Typhoon_*_Track_Vector.npy
├─ picture/
│  ├─ Typhoon_*_animation.gif
│  └─ Typhoon_*_Max.png
├─ yunlin_population_40x40.csv
├─ 雲林縣醫院名單_名稱地址 - 雲林縣醫院名單.csv
├─ 雲林縣老人福利機構名冊 - 工作表1.csv
├─ 雲林縣變電所整理 - 雲林縣變電所.csv
└─ 雲林縣避難收容處所 - Sheet1.csv
```

## 安裝方式

建議使用 Python 3.12 建立虛擬環境。

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

## 啟動系統

在專案根目錄執行：

```powershell
python -m streamlit run app.py
```

啟動後開啟：

```text
http://localhost:8501
```

## 操作流程

1. 從左側選擇颱風事件
2. 選擇需要顯示的地圖圖層及設施
3. 從功能選單切換分析模組
4. 查看分析結果或輸入參數執行演算法
5. 如需模擬其他事件可選擇自訂並填寫颱風資訊

## 資料說明

- `Dataset/data.csv` 儲存颱風基本資料
- `Dataset/typhoon_tracks.csv` 儲存颱風路徑
- `Output/` 儲存淹水深度網格及時間序列資料
- `picture/` 儲存最大淹水圖及淹水動畫
- 設施 CSV 儲存醫院、長照中心、變電所及收容所的位置資料

## 注意事項

- 地址定位功能需要網路連線
- 地址搜尋範圍限制在雲林縣
- Nominatim 屬於公開服務。請避免短時間大量送出地址查詢
- 資源分配演算法可能需要數十秒完成
- 請勿任意更改資料檔名稱或資料夾位置
- 自訂颱風只會覆蓋事件資訊及風速風險分數。淹水資料仍取自選定的參考情境

## 授權及資料來源

地址定位資料來自 OpenStreetMap contributors。其他資料及研究成果請依專案原始資料來源及研究文件規範使用。
