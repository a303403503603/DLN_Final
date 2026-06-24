import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from training.train_ppo import _load_stock_data
from rl_env.trading_env import MultiStockTradingEnv
from pipeline.config import RL_TRAIN_START, RL_TRAIN_END, CANDIDATE_POOL_SIZE, HOLD_POOL_SIZE, GRU_CACHE_DIR

# Load 5 stocks for smoke test
data = _load_stock_data(min_days=200, max_stocks=5, date_range=(RL_TRAIN_START, RL_TRAIN_END))
codes = list(data.keys())
print(f'Loaded {len(codes)} stocks: {codes}')

env = MultiStockTradingEnv(
    all_stocks=codes,
    data=data,
    gru_cache_dir=GRU_CACHE_DIR,
    candidate_pool_size=min(CANDIDATE_POOL_SIZE, len(codes)),
    hold_pool_size=min(HOLD_POOL_SIZE, len(codes) - 1),
)
print(f'env: candidate={env.candidate_pool_size} hold={env.hold_pool_size} watch={env.watch_pool_size}')

obs, _ = env.reset()
print(f'reset OK: obs shape={obs.shape}')

act_dim = env.action_space.shape[0]
print(f'action dim = {act_dim}')

for i in range(3):
    act = np.random.uniform(-0.1, 0.1, size=act_dim).astype(np.float32)
    act[-1] = 0.1
    new_obs, reward, terminated, truncated, info = env.step(act)
    val = info['portfolio_value']
    hold = len(info['holdings'])
    hpool = info['hold_pool_size']
    wpool = info['watch_pool_size']
    wentry = info['watch_entries']
    trades = info['trades_count']
    rd = info.get('reward_debug', {})
    print(f'  step {i}: r={reward:.4f} val={val:,.0f} hold={hold} hpool={hpool} wpool={wpool} wentry={wentry} trades={trades}')
    print(f'    reward breakdown: r_profit={rd.get("r_profit",0):.4f} r_drawdown={rd.get("r_drawdown",0):.4f} r_turnover={rd.get("r_turnover",0):.4f}')
    if terminated or truncated:
        break
print('SMOKE TEST OK')
