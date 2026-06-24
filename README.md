# GRU + PPO 自動股票交易系統

NCKU 模擬交易 API 的端到端自動化股票交易系統，使用多時間窗口 GRU 預測 + PPO 強化學習動態選股，搭配 Discord Bot 即時控制與 24/7 排程。

---

## 整體架構

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         資料管線 (每天 16:30)                            │
│                                                                         │
│  NCKU API ─► api_cache/*.pkl ──┐                                        │
│                                ├──► price_data/{code}.csv ──► features │
│  訓練資料 (2003-2022)  ────────┘                                        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          GRU 預訓練 (one-time)                            │
│  MultiTimeGRU                                                          │
│  ├── 8 windows × 5 features → multi-time encoder                       │
│  ├── Hidden=256, Layers=3, Dropout=0.3                                 │
│  ├── Loss = weighted MSE(OHLC × 7 horizons)                            │
│  └── Output: 2048 latent + 28 preds → cache.npy (per stock)            │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PPO 強化學習 (TradingEnv V2)                        │
│  Observation (n=50 × 2115 dim)                                          │
│  Action: Box(-1, 1, 51) — act[0:50]=預估%, act[50]=cash_weight         │
│  Reward B: 持艙加權 Σ(return × position_weight) × profit_weight        │
│  Pool 結構: Hold (前 N=10 by ROI 降冪) + Watch (後 40 by GRU score)     │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                即時交易 (Discord Bot, 每天 10:00 自動)                   │
│  LiveTrader (V2 mirror): 對齊 API holdings → 重建 V2 pool →            │
│  PPO inference → softmax 解析 → target 權重 → 提案 → 確認下單          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 資料管線

### 1.1 來源
- **NCKU 模擬交易 API**: `https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/`
- **價量資料**: `get_taiwan_stock_data(code, start, end)` → DataFrame (date/capacity/turnover/high/low/close/change/...)
- **股票清單**: `get_all_stock_list()` → 1942 檔

### 1.2 兩層快取
- `data/api_cache/{code}.pkl`: API 回傳原始格式，**交易 bot 使用**
- `data/price_data/{code}.csv`: 標準化 6 欄 (Date/Open/High/Low/Close/Volume) + 第 2 行 metadata (e.g. `2330.TW`)，**訓練使用**

### 1.3 Phase 1 — API Fetch (`pipeline/_utils.fetch_parallel`)
- ThreadPoolExecutor, 10 workers
- Timeout 15s × retry 2 (避免死鎖)
- 同時寫 `api_cache/*.pkl` + `price_data/*.csv` (`append_to_price_data`)

### 1.4 Phase 2 — GRU Cache (`phase_gru`)
- 載入 `price_data/{code}.csv` + `api_cache/{code}.pkl` 合併去重
- 計算 5 個 ratio 特徵 (`pipeline/data_processor.engineer_features`):
  - `close_ret`, `open_ret`, `high_close_ratio`, `low_close_ratio`, `vol_ret`
- 標準化 (StandardScaler)
- `MultiTimeGRU.compute_gru_for_stock(features, model)` → (n_days, 2076) cache
- 存成 dict 格式 `{'cache': array, 'gru_date': 'YYYY-MM-DD'}`，共 **1941 檔** (`gru_cache/*.npy`)

---

## 2. GRU 模型 (`model/multi_time_gru.py`)

### 2.1 架構
```
Input: 8 time-windows × 5 features
  Window sizes: [1, 3, 5, 7, 14, 30, 60, 120] 交易日
  Features (per day): [close_ret, open_ret, high_close_ratio, low_close_ratio, vol_ret]
            ↓
MultiTimeGRUEncoder (per window):
  nn.GRU(input=5, hidden=256, layers=3, dropout=0.3, batch_first=True)
            ↓
Concat 8 latents → 2048 dim
            ↓
Linear(2048 → 28)  # 7 horizons × 4 OHLC
```

### 2.2 預測目標
- 7 horizons: `[1, 3, 5, 7, 14, 30, 60]` 個交易日
- 每個 horizon 預測 4 個 OHLC 報酬率 (`high_ret/close_ret/low_ret/open_ret`)
- 訓練 loss: weighted MSE，權重 `[0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03]` (近期高)

### 2.3 訓練 (`training/train_gru.py`)
- Optimizer: Adam(lr=1e-3)
- Epochs: 100, batch: 2048, patience: 10
- 早停於 epoch 18
- 最終模型: `checkpoints/gru_shared.pth` (12,548,388 params)

### 2.4 Inference
- 全市場 1941 檔一次性推論
- 結果存 `gru_cache/{code}.npy` (dict 格式)
- 線上 PPO 環境只讀 cache，不重跑 GRU

---

## 3. PPO 強化學習 (V2 環境)

### 3.1 環境 (`rl_env/trading_env.py`)

#### 觀察空間
- 候選池大小 `CANDIDATE_POOL_SIZE = 50`
- 每檔特徵 `PER_STOCK_DIM = 2115`:
  - `GRU latent 2048` (8 windows × 256)
  - `GRU preds 28` (7 × 4)
  - `OHLCV + spread 5`
  - `industry one-hot 34`
- 總維度 `OBS_TOTAL = 50 × 2115 = 105,750`

#### V2 動作語意
```
action ∈ Box(-1, 1, 51)
├── act[0:50]: 50 檔的未來漲跌預估 % (連續)
│              解析時 clip 到 [-PRED_REWARD_CLIP, +PRED_REWARD_CLIP] = ±30%
└── act[50]:   現金權重 (clip [0, 1])
```

**環境解析**:
1. `softmax_probs = softmax(act[0:50])` — 機率分布
2. `entry_threshold = mean(softmax_probs) = 1/50 = 0.02` — 進場門檻
3. `target_weight[i] = softmax_probs[i] × (1 - cash_weight)`

#### V2 候選池 (Hold / Watch 分段)
```
Pool (50 檔):
├── Hold 池 (前 N=10):  持艙按 ROI 降冪 → 取前 10 檔
│                       跌出 top N 的持艙 → 移到 Watch 池（不凍結）
└── Watch 池 (後 40):  GRU score 排序補滿 (含跌出的持艙)
```

#### V2 Reward (持艙加權)
```
r_profit = Σ(return_i × position_weight_i) × profit_weight
         (i ∈ all 50 stocks, position_weight 為 softmax 解析後的 target)
r_drawdown, r_diversity, r_turnover, r_excess_cash ...
```

### 3.2 PPO 訓練 (`training/train_ppo.py`)

#### 超參數 (`pipeline/config.py:PPO`)
- `total_timesteps: 500,000`
- `n_steps: 2048, batch_size: 64, n_epochs: 5`
- `lr: 3e-4, gamma: 0.99, gae_lambda: 0.95, clip_range: 0.2`
- `ent_coef: 0.01, vf_coef: 0.5, max_grad_norm: 0.5`

#### 訓練結果 (V2 N=10, seed=42, 200 epochs)
- **Return: 250.54%** (val 期間 2023-2024)
- **Sharpe: 3.75**
- **MaxDD: 15.95%**
- **val_final: $350.5M** (從 $100M 初始)
- **結構性改善**: hold 3→50, trades/day 10→48

#### 模型儲存
- `checkpoints/ppo_best_seed42.pth` (best val)
- `checkpoints/ppo_ep{50,100,150,200}_seed42.pth`
- `checkpoints/ppo_final_seed42.pth`

---

## 4. 即時交易 (`pipeline/run_daily_pipeline.py:LiveTrader`)

### 4.1 V2 Mirror
即時端跟訓練環境對齊，避免 V1/V2 不一致:
- `_screen_pool`: 與 `trading_env._screen_pool` 相同 (Hold 10 by ROI + Watch 40 by GRU score + tech boost)
- `_compute_target`: 與 `trading_env.step` 動作解析相同 (clip ±0.3 → softmax → threshold=1/n → target_weight)
- `_build_observation`: 從 `gru_cache` 讀 2048 latent + 28 preds
- `cost_basis`: 從 API `Get_User_Stocks` 抓 `beginning_price` 存到 state 供 ROI 排序

### 4.2 流程 (`compute_proposal` → `execute_proposal`)

```
09:30 開盤
  ↓
10:00 bot 自動觸發 do_trade_flow (或按 Discord "PPO 提案" 按鈕)
  ↓
1. 載入 api_cache/* + engineer_features → stock_data
2. Get_User_Stocks → api_holdings + api_cost → 更新 cost_basis
3. 對帳 pending_buys (確認成交/退款) + pending_sells
4. self.holdings = api_holdings (以 API 為真)
5. 若還有 pending_buys → skip (return None)
6. _screen_pool → pool (50 檔: 10 hold + 40 watch)
7. PPO inference → action[0:51] → softmax 解析 → target weights
8. Build proposal: buys (target > current) + sells (target < current)
  ↓
送出 Discord embed + 確認/取消/調整 buttons
  ↓
用戶按「確認，執行」
  ↓
execute_proposal:
  - Buy_Stock / Sell_Stock (NCKU API)
  - 更新 state: cash, holdings, cost_basis, pending_buys/sells
  - 寫 logs/trades_{date}.csv
```

### 4.3 對帳 (`reconcile_state`)
每天 16:30 自動對帳:
- `Get_User_Stocks` → 跟 state.holdings 比對
- 買單：code 出現在 holdings → 確認成交；否則 → 退款
- 賣單：code 不在 holdings → 確認成交退款；否則 → 仍 pending
- **負現金保護**: 對帳後若 cash < 0 → reset 為 INITIAL_CAPITAL

---

## 5. Discord Bot (`pipeline/discord_bot.py`)

### 5.1 排程任務 (`@tasks.loop(minutes=1)`)
- `check_trade_time`: 週一-五 10:00 自動 PPO 提案 (避免重複 `last_trade_date`)
- `auto_post_market_update`: 週一-五 16:30 自動 fetch + GRU + 對帳

### 5.2 控制面板 (`ControlPanelView`)
8 個按鈕 (含 persistent custom_id):
- **狀態**: 現金/累計成交/待成交/各股成本與投報率
- **股價查詢**: 開 `PriceModal` 輸入代碼 → 顯示最新價 + GRU 預測
- **盤後更新**: 立即跑 fetch + GRU + 對帳
- **查詢交易**: 開 `TradeQueryModal` 查 `logs/trades_*.csv`
- **GRU 排行**: `GRURankView` 含 t+1/3/5/7/14/30/60 下拉選單 (5 分鐘 file-cache 加速)
- **手動買入/賣出**: `ManualBuyModal` / `ManualSellModal` 直接下單
- **PPO 提案**: 立即計算提案 → 開 `TradeProposalView` 確認/取消/調整

### 5.3 持久化機制
- `setup_hook` 內 `bot.add_view(ControlPanelView())` 註冊 persistent view
- `on_ready` 內 `_cleanup_old_panels` 刪除舊 panel 訊息再送新 (避免重啟後舊按鈕失效)
- 所有 interaction log 走 `on_interaction` + 模組 logger

### 5.4 錯誤防護
- `_safe_followup`: 包 401/HTTPException，token 過期時靜默
- `_safe_channel_send`: 同上
- `append_to_price_data`: 寫 price_data CSV 失敗不影響 api_cache
- `fetch_parallel`: timeout 15s + retry 2 (避免單一 API 卡死整批)

---

## 6. 排程與部署

### 6.1 已註冊的 Windows Task Scheduler
| Task | 觸發 | 動作 |
|------|------|------|
| `PPO_Bot` | onlogon | 啟動 Discord bot (pythonw.exe 背景) |
| `PPO_Daily` | 週一-五 16:30 | fetch + GRU + 對帳 |

### 6.2 重新註冊
```cmd
cd <repo_root>\pipeline
setup_schedule.bat   # (需 Admin 權限)
```

### 6.3 啟動 / 停止 bot
```powershell
# 啟動
Start-Process "C:\ProgramData\anaconda3\envs\dl_final\pythonw.exe" `
  -ArgumentList "-u","-m","pipeline.discord_bot" `
  -WorkingDirectory "<repo_root>"

# 停止
Stop-Process -Name pythonw -Force
```

### 6.4 環境需求
- Python 3.10 (dl_final conda env)
- PyTorch + CUDA
- `requests`, `pandas`, `numpy`, `discord.py[voice]`, `stable-baselines3`, `gymnasium`, `scikit-learn`

---

## 7. 關鍵檔案路徑

| 用途 | 路徑 |
|------|------|
| 設定 | `pipeline/config.py` |
| 環境 | `rl_env/trading_env.py` |
| GRU 模型 | `model/multi_time_gru.py` |
| PPO 模型 | `model/ppo_agent.py` |
| 訓練 | `training/train_gru.py`, `training/train_ppo.py` |
| 即時交易 | `pipeline/run_daily_pipeline.py` |
| Bot | `pipeline/discord_bot.py` |
| 工具 | `pipeline/_utils.py` |
| 特徵工程 | `pipeline/data_processor.py` |
| API | `data/stock_api/core.py` |
| GRU 權重 | `checkpoints/gru_shared.pth` |
| PPO 權重 | `checkpoints/ppo_best_seed42.pth` |
| State | `logs/portfolio_state.json` |
| Pipeline state | `logs/pipeline_state.json` |
| GRU cache | `gru_cache/{code}.npy` (1941 dict-format) |
| Price data | `data/price_data/{code}.csv` |
| API cache | `data/api_cache/{code}.pkl` |
| Trade log | `logs/trades_{YYYY-MM-DD}.csv` |
| GRU rank cache | `logs/.gru_rank_cache.json` (5min TTL) |

---

## 8. 開盤測試清單

- [ ] **狀態**: 確認 cash + 22 持股顯示正確
- [ ] **股價查詢**: 輸入 2330 → 顯示日期 (yyyy-mm-dd) + OHLCV + GRU 預測
- [ ] **盤後更新**: 按下後 4 個 embed (fetch / gru / 對帳 / 完成)
- [ ] **查詢交易**: 查今日 `trades_2026-XX-XX.csv`
- [ ] **GRU 排行**: 顯示前 20 名 → 切換 t+1/3/5/7/14/30/60
- [ ] **手動買入**: 輸入 2330 1 張 100 → API 確認 → state 更新
- [ ] **手動賣出**: 同上反向
- [ ] **PPO 提案**: 10:00 自動觸發 → 50 檔權重 + 確認/取消
- [ ] **16:30 自動對帳**: 確認 53 筆累積成交
- [ ] **重啟 bot**: 舊 panel 自動刪除，新 panel 按鈕可正常回應

---

## 9. 故障排除

| 症狀 | 原因 | 解法 |
|------|------|------|
| Bot 上線但按鈕沒反應 | 舊 message 跟新 process 沒綁 | 刪舊 panel 等新 panel |
| 401 Unauthorized | DISCORD_TOKEN 過期 | 更新 `.env` 重啟 bot |
| gru_rank 超慢 | 第一次載入 1941 個 npy | 5 分鐘內重複按會快 (file cache) |
| state.cash 變負 | 多次退款累計 | 對帳時自動 reset 為 INITIAL_CAPITAL |
| Sell 沒成交 | 對方仍在 holdings | 對帳時 pending_sells 留著，等下次 |
| log 出現 FutureWarning (pd.fillna) | pandas 版本 | 無害，可忽略 |
