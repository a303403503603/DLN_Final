# AGENTS.md

本檔提供給（例如 opencode / Copilot / 其他自動化 agent）在此 repo 工作時的「讀取範圍、探索策略、常用入口點」指南。

## 0) 核心原則（避免超大 token / 超慢）

- **預設不要整檔讀取大型檔案**：優先用搜尋（grep/semantic）定位到需要的函式/區塊，再用「小範圍」讀取。
- **預設不要讀資料檔**（CSV/大 JSON）除非任務明確需要。
- **不要把整份專案總結/報告當成主要上下文來源**：先看程式碼入口點與設定檔即可。
- **改動要小且可驗證**：只修與需求相關的部分，避免大規模重構。

## 1) 預設忽略（除非明確要求）

這些內容非常容易造成 token 爆量、且多數任務不需要。

- `grullr_stock/data/**/*.csv`（資料量大）
- `grullr_stock/data/**/stock_symbol_map.json`（必要時再讀，通常用 key 查詢即可）
- `GRU_PPO_股票交易系統專案總結.md`（長文檔，除非要改報告/找描述）
- 任何產出物（若存在）：`**/*.pt`、`**/*.pth`、`**/*.pkl`、`**/*.zip`、`**/*.parquet`、`**/*.feather`、`**/*.npy`、`**/*.npz`

> 若你要讓工具層面也忽略，建議另外建立 `.gitignore`/`.opencodeignore`/工具專用 ignore（依你使用的工具而定）。

## 2) 探索策略（先找入口、再縮小閱讀）

### A. 先找「入口檔」

這個專案常見入口點（通常只需要從這裡開始讀）：

- `grullr_stock/main.py`：單一入口點（所有日常操作 + 訓練）
- `grullr_stock/pipeline/run_daily_pipeline.py`：自動化交易 pipeline（fetch→gru→trade）
- `grullr_stock/pipeline/config.py`：所有設定與超參數
- `grullr_stock/training/train_gru.py`：GRU 訓練
- `grullr_stock/training/train_ppo.py`：PPO 訓練
- `grullr_stock/training/run_batch_sim.py`：多時段回測

### B. 再用「符號/關鍵字搜尋」縮小範圍

建議先用下列關鍵字搜尋（而不是讀完整檔案）：

- 環境：`TradingEnv`、`step`、`reset`、`reward`、`observation`
- PPO：`PPO`、`Agent`、`select_action`、`update`、`rollout`
- GRU：`GRU`、`forward`、`hidden`、`MultiTimeGRU`
- 資料：`DataProcessor`、`load`、`preprocess`、`features`、`normalize`
- 交易：`order`、`position`、`buy`、`sell`、`commission`、`slippage`

### C. 只讀你要改的那 30–200 行

- 若工具支援「行範圍讀取」，請先定位函式，再只讀該函式前後小段。
- 若只能整檔讀取，請先用搜尋縮小到最少檔案數（理想：≤3 個檔）。

## 3) 模組地圖（讀碼順序建議）

- 環境：`grullr_stock/rl_env/trading_env.py`
- 模型：
  - `grullr_stock/model/multi_time_gru.py`
  - `grullr_stock/model/ppo_agent.py`
- 資料處理：`grullr_stock/pipeline/data_processor.py`
- Pipeline 入口：`grullr_stock/main.py` → `grullr_stock/pipeline/run_daily_pipeline.py`
- 設定：`grullr_stock/pipeline/config.py`
- API/資料抓取：`grullr_stock/data/stock_api/`（通常只在抓資料或修 API 時才需要）

## 4) 變更規範（避免不必要的上下文需求）

- **不要**因為「可能更好」就順便調參或改訓練流程；除非需求明確。
- **不要**把大型資料檔納入 patch 或在討論中貼全檔內容。
- 修改設定優先集中在 `grullr_stock/config.py`，避免散落。

## 5) 執行/驗證（只在需要時）

- 依任務選擇最小可行驗證：
  - 修 bug：優先跑對應的單一腳本/最小流程
  - 改模型：先做 import / shape 檢查，再跑短步數訓練
  - 改環境：先跑 reset/step 幾回合，確認不崩潰

> 注意：若工具執行時會自動掃描大量資料檔，請先改成使用少量樣本或 mock。

## 6) 你可以直接問我的問題模板（可降低探索成本）

當你要 agent 幫忙時，建議用以下格式：

- 目標：要修什麼/要新增什麼
- 範圍：限定只改哪些檔案（最多 1–3 個）
- 禁止：不要讀哪些資料/不要改哪些流程
- 驗證：你希望我跑哪個腳本或做到什麼輸出

範例：

- 目標：修正 `TradingEnv.step()` reward 計算
- 範圍：只改 `grullr_stock/env/trading_env.py`
- 禁止：不要讀 `grullr_stock/data/**/*.csv`
- 驗證：跑 `run_simulation.py` 只走 10 steps 不要 crash
