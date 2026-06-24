"""GRU+PPO 股票交易系統配置"""
import os

# ── 從 .env 檔載入環境變數 ──
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(_env_path):
    with open(_env_path, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith('#'):
                continue
            if '=' in _line:
                _k, _v = _line.split('=', 1)
                _k, _v = _k.strip(), _v.strip().strip('"\'')
                if _k and not os.environ.get(_k):
                    os.environ[_k] = _v

# 路徑設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')
MODEL_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'gru_cache'), exist_ok=True)

# 股票清單
TARGET_STOCKS = ['2330', '2317', '2454', '2327', '2353', '2303', '2313', '2408', '0050', '0056', '2308', '2301', '2305', '2356', '2357', '2363', '2379', '2382', '2395', '2368']
STOCK_NAMES = {
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2327': '聯電',
    '2353': '瑞昱', '2303': '宏碁', '2313': 'nano', '2408': '奇美電',
    '0050': '元大台灣50', '0056': '元大台20', '2308': '聯發科',
    '2301': '聯電', '2305': '華星', '2356': '聯茂', '2357': '欣興',
    '2363': '光寶科', '2379': '台達電', '2382': '研華', '2395': '美律',
    '2368': '微星', '2412': '創意'
}

# 多時序設定
TIME_WINDOWS = [1, 3, 5, 7, 14, 30, 60, 120]
GRU_HIDDEN_SIZE = 256
GRU_NUM_LAYERS = 3
DROPOUT = 0.3

# 預測區間設定（×4 for OHLC）
PRED_HORIZONS = [1, 3, 5, 7, 14, 30, 60]
N_HORIZONS = len(PRED_HORIZONS)
N_PRED_TARGETS = N_HORIZONS * 4
# 各 horizon 在 loss 中的權重（越近期越高）
HORIZON_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03]

# PPO 動態選股 - 候選池大小
CANDIDATE_POOL_SIZE = 50

# V2: Hold / Watch 池分開
# Hold 池 = 固定 N 檔（依 ROI 降冪納入）→ PPO 動作前 N 維
# Watch 池 = 補滿 50-N 檔（GRU top K 非持艙）→ PPO 動作 N~50 維
HOLD_POOL_SIZE = 10  # 預設值，CLI --hold-pool-size 可調
PRED_REWARD_CLIP = 0.3  # 預估漲跌 clip 到 [-0.3, +0.3] = ±30%

# 特徵設定（全 ratio，消除價格 level）
FEATURE_COLS = ['close_ret', 'open_ret', 'high_close_ratio', 'low_close_ratio', 'vol_ret']

# 預訓練參數
GRU_PRETRAIN = dict(
    epochs=100,
    lr=0.001,
    batch_size=2048,
    hidden_size=GRU_HIDDEN_SIZE,
    num_layers=GRU_NUM_LAYERS,
    dropout=DROPOUT,
    val_ratio=0.1,
    patience=10,
)

# 平行環境數量
N_ENVS = 6

# PPO 參數
PPO = dict(
    total_timesteps=500000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# Daily turnover cap (portion of portfolio that can be rebalanced)
# Increased from 0.05 -> 0.20 to boost trade count for 100-trade requirement
TURNOVER_LIMIT = 0.20

# Minimum trade: skip changes smaller than this fraction of position
# Reduced from 0.1 -> 0.0 to allow any-size trades
MIN_TRADE_PCT = 0.0

# Minimum cash reserve (fraction of portfolio that must stay as cash)
MIN_CASH_RATIO = 0.05

# Maximum cash ratio: cash above this triggers a penalty
MAX_CASH_RATIO = 0.3
EXCESS_CASH_PENALTY = 0.2

# Maximum single stock position (fraction of portfolio)
MAX_POSITION_RATIO = 0.3

# Tech stock boost: industries to favor + multiplier applied to GRU scores
TECH_INDUSTRIES = {'24', '25', '26', '28', '31'}
TECH_BOOST = 1.15

# Stop-loss: force sell if loss exceeds this AND GRU predicts further decline
STOP_LOSS_PCT = 0.15



# 產業分類（33 類 + 1 未知/ETF）
NUM_INDUSTRIES = 34

# 每檔候選股貢獻：GRU_latent(2048) + preds(28) + OHLCV+Vol(5) + industry_onehot(34) = 2115
PER_STOCK_DIM = GRU_HIDDEN_SIZE * len(TIME_WINDOWS) + N_HORIZONS * 4 + 5 + NUM_INDUSTRIES

# 觀察空間（動態候選池版）
OBS_TOTAL = CANDIDATE_POOL_SIZE * PER_STOCK_DIM

# 訓練/測試期間
TRAIN_START = '2003-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2024-12-31'
SIM_START = '2023-01-01'
SIM_END = '2024-12-31'

# 交易參數
INITIAL_CAPITAL = 100_000_000
COMMISSION_BUY = 0.001
COMMISSION_SELL = 0.003
MIN_COMMISSION = 20

# PPO 訓練期間
RL_TRAIN_START = '2003-01-01'
RL_TRAIN_END = '2022-12-31'
RL_SIM_START = '2023-01-01'
RL_SIM_END = '2024-12-31'

# 獎勵函數_weight
REWARD_WEIGHTS = dict(
    profit=10.0,
    drawdown=0.5,
    invalid_action_penalty=0.01,
    turnover_penalty=1.0,
)

# 每日最少交易數量（軟性獎勵：達不到就扣分）
MIN_DAILY_TRADES = 5
MIN_DAILY_TRADE_PENALTY = 0.05

# 換股多樣性懲罰：避免 PPO 重複買賣同一支股票
DIVERSITY_WINDOW = 10            # 追蹤最近 N 步的買入記錄
DIVERSITY_PENALTY = 0.01         # 同股票在 window 內重複買的額外扣分

# PPO 輸出維度：CANDIDATE_POOL_SIZE 支股票 + 現金權重
OUTPUT_DIM = CANDIDATE_POOL_SIZE + 1
INPUT_DIM = OBS_TOTAL

# 資料路徑
PRICE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'price_data')
CSV_FILE = PRICE_DATA_DIR  # legacy alias
GRU_CACHE_DIR = os.path.join(BASE_DIR, 'gru_cache')

# C1: 統一狀態檔路徑（避免 discord_bot.py / run_daily_pipeline.py 用字面字串）
PORTFOLIO_STATE_FILE = os.path.join(LOG_DIR, 'portfolio_state.json')
PIPELINE_STATE_FILE = os.path.join(LOG_DIR, 'pipeline_state.json')

# Trading API credentials (must be set in .env)
TRADING_ACCOUNT = os.environ.get('TRADING_ACCOUNT')
TRADING_PASSWORD = os.environ.get('TRADING_PASSWORD')

# Discord Bot
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN', '')
DISCORD_CHANNEL_ID = int(os.environ.get('DISCORD_CHANNEL_ID', '0'))

# GPU
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'
