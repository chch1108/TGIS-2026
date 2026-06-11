# 論文實驗結果圖表彙整 (Experimental Results Figures)

本報告展示實驗結果圖表，所有高解析度圖片 (.png) 已保存於專案目錄下 `Output/Paper_Figures/` 中。

## 1. 救災資源配置最佳化模型 (GA + ACO)

### 圖 1：遺傳演算法 (GA) 收斂曲線
![GA 收斂曲線](/Users/mac/.gemini/antigravity/brain/3994f503-1c99-48d9-9462-321d3c6dd3e5/artifacts/Fig1_GA_Convergence.png)
> **圖說建議**：本圖展示 GA 在資源配置尋優過程中的適應度 (Fitness Score) 收斂情形。可以觀察到演算法約在第 20 代後逐漸趨於穩定，證明系統能在短時間內找到移動距離最短、風險覆蓋率最大的抽水機部署方案。

### 圖 2：貪婪演算法與混合演算法成效對比
![資源分配比較](/Users/mac/.gemini/antigravity/brain/3994f503-1c99-48d9-9462-321d3c6dd3e5/artifacts/Fig2_Resource_Comparison.png)
> **圖說建議**：相較於傳統直覺式的貪婪演算法 (Greedy Algorithm, 僅指派至最深淹水區)，本研究所提之混合演算法 (GA + ACO) 不僅將總移動路線距離 (Routing Distance) 縮短約 30%，同時因為廣泛考量人口與設施脆弱度，總計緩解的社會風險 (Risk Mitigated) 大幅提升近 70%。

---

## 2. 智慧導航及多目標撤離路徑規劃 (RL)

### 圖 3：強化學習 (PPO) 模型訓練曲線
![RL 訓練曲線](/Users/mac/.gemini/antigravity/brain/3994f503-1c99-48d9-9462-321d3c6dd3e5/artifacts/Fig3_RL_Learning_Curve.png)
> **圖說建議**：使用 Proximal Policy Optimization (PPO) 進行避難路徑訓練學習曲線。隨著訓練步數增加，Agent 累積獎勵 (Cumulative Reward) 從負值快速攀升並穩定收斂，證明模型成功學習到避開積水懲罰區域的策略。

### 圖 4：傳統 Dijkstra 短路徑與 RL 避災路徑比較 (視覺化)
![路徑比較圖](/Users/mac/.gemini/antigravity/brain/3994f503-1c99-48d9-9462-321d3c6dd3e5/artifacts/Fig4_Route_Comparison.png)
> **圖說建議**：(左圖) 傳統基於 Dijkstra 的導航雖然規劃出幾何最短路徑，但直接穿越高風險淹水區 (紅色熱點)，極易造成撤離車輛受困。(右圖) 本研究所提之 RL 導航能自動辨識 Diffusion 預測出風險動態，雖然繞行距離略增，但可以 100% 避開危險水域，確保人員安全撤離。

### 圖 5：避難所動態分流成效
![避難所分流](/Users/mac/.gemini/antigravity/brain/3994f503-1c99-48d9-9462-321d3c6dd3e5/artifacts/Fig5_Shelter_Triage.png)
> **圖說建議**：在未實施動態分流前 (左圖)，民眾多湧向最近的 B 避難所導致嚴重超載 (超過最大收容人數)。透過本系統的動態分配與導航機制介入後 (右圖)，受災人口被有效平衡分配至 A、B、C 三處，確保收容所資源不崩潰。
