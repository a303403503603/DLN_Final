"""PPO RL 訓練 - 動態候選池 + 共用 GRU Cache"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import logging
from datetime import datetime
from collections import deque

from pipeline.config import (
    PPO, MODEL_DIR, OBS_TOTAL, OUTPUT_DIM,
    GRU_PRETRAIN, FEATURE_COLS, PRICE_DATA_DIR,
    CANDIDATE_POOL_SIZE, RL_TRAIN_START, RL_TRAIN_END,
    GRU_CACHE_DIR, HOLD_POOL_SIZE
)
from rl_env.trading_env import MultiStockTradingEnv
from model.ppo_agent import PPOAgent
from pipeline.data_processor import engineer_features

logger = logging.getLogger(__name__)


def _load_stock_data(min_days=100, max_stocks=200, date_range=None):
    """共用資料 loader，從 price_data 目錄讀取。"""
    import pandas as pd

    files = sorted([f for f in os.listdir(PRICE_DATA_DIR) if f.endswith('.csv')])
    all_rows = []
    for fname in files:
        code = fname.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(PRICE_DATA_DIR, fname), skiprows=[1])
            df = df.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            })
            df['stock_id'] = code
            if date_range:
                df = df[(df['date'] >= date_range[0]) &
                        (df['date'] <= date_range[1])].copy()
            if not df.empty:
                all_rows.append(df)
        except Exception as e:
            print(f"  Skip {code}: {e}")

    if not all_rows:
        return {}

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all.dropna(subset=['high', 'low'], inplace=True)
    df_all = df_all[df_all['close'] > 0].copy()
    df_all = df_all.sort_values('date').reset_index(drop=True)

    stock_volume = df_all.groupby('stock_id')['volume'].sum()
    stock_first = df_all.groupby('stock_id')['close'].first()
    stock_last = df_all.groupby('stock_id')['close'].last()
    total_return = (stock_last / stock_first - 1).clip(-0.99, 10)
    score = stock_volume * (1 + total_return)
    top_stocks = score.sort_values(ascending=False).head(max_stocks).index.tolist()

    stock_data = {}
    for code in top_stocks:
        df = df_all[df_all['stock_id'] == code].sort_values('date').reset_index(drop=True)
        if len(df) >= min_days:
            df = engineer_features(df)
            stock_data[code] = df

    return stock_data


def load_universe_data(max_stocks=200):
    """Load OHLCV data for up to max_stocks from CSV."""
    stock_data = _load_stock_data(
        min_days=100, max_stocks=max_stocks,
        date_range=(RL_TRAIN_START, RL_TRAIN_END)
    )
    return stock_data


def get_active_stock_codes(stock_data):
    return list(stock_data.keys())


def _run_validation(agent, log_dir='logs', seed=None, hold_pool_size=None):
    """Use current model to run a validation episode and log returns."""
    from pipeline.config import RL_SIM_START, RL_SIM_END

    val_data = _load_stock_data(
        min_days=50, max_stocks=50,
        date_range=(RL_SIM_START, RL_SIM_END)
    )

    if not val_data:
        logger.warning("No validation data available")
        return

    env = MultiStockTradingEnv(
        all_stocks=list(val_data.keys()),
        data=val_data,
        gru_cache_dir=GRU_CACHE_DIR,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        hold_pool_size=hold_pool_size,
    )

    values = []
    obs, _ = env.reset()
    done = False
    while not done:
        action, _, _ = agent.get_action(obs, stochastic=False)
        new_obs, reward, terminated, truncated, info = env.step(action.flatten())
        values.append(info['portfolio_value'])
        obs = new_obs
        done = terminated or truncated

    if len(values) < 2:
        return

    initial_cap = env.initial_capital
    final_cap = values[-1]
    realized = (final_cap / initial_cap - 1) * 100
    returns = np.diff(values) / np.array(values[:-1])
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    held = {k: v for k, v in env.holdings.items() if v > 0}

    logger.info(
        f"  [Val] ret={realized:6.2f}%  Sharpe={sharpe:6.2f}  "
        f"MaxDD={max_dd:.3%}  val_final=${final_cap:,.0f}  held={held}"
    )


def train_ppo(stock_data, stock_codes, log_dir='logs', seed=None, num_episodes=200,
              hold_pool_size=None):
    """PPO 訓練主函式（V2：Hold/Watch 池 + 預估動作語意）"""
    os.makedirs(log_dir, exist_ok=True)
    seed_tag = f'_seed{seed}' if seed is not None else ''
    n_hold = hold_pool_size if hold_pool_size is not None else HOLD_POOL_SIZE



    logger.info(f"Observation dim: {OBS_TOTAL}")
    logger.info(f"Action dim: {OUTPUT_DIM}")
    logger.info(f"Candidate pool: {CANDIDATE_POOL_SIZE}")
    logger.info(f"Hold pool: {n_hold}  Watch pool: {CANDIDATE_POOL_SIZE - n_hold}")
    logger.info(f"Stock universe: {len(stock_codes)}")

    # 1. Load shared GRU model
    logger.info("[1/3] Verifying GRU model...")
    gru_path = os.path.join(MODEL_DIR, 'gru_shared.pth')
    if os.path.exists(gru_path):
        from model.multi_time_gru import MultiTimeGRU
        n_feat = len(FEATURE_COLS)
        dummy_model = MultiTimeGRU(n_feat, GRU_PRETRAIN['hidden_size'],
                                    GRU_PRETRAIN['num_layers'],
                                    GRU_PRETRAIN['dropout'])
        dummy_model.load_state_dict(
            torch.load(gru_path, map_location='cpu', weights_only=True))
        logger.info(f"  [OK] Loaded {gru_path}")
        del dummy_model
    else:
        logger.info(f"  [!] No GRU model found at {gru_path}")
        logger.info(f"  Using zero-filled GRU cache (train may be suboptimal)")

    # 2. Initialize environment (V2: hold/watch pool)
    logger.info("[2/3] Initializing environment...")
    env = MultiStockTradingEnv(
        all_stocks=stock_codes,
        data=stock_data,
        gru_cache_dir=GRU_CACHE_DIR,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        hold_pool_size=n_hold,
    )

    # 3. PPO Agent
    logger.info("[3/3] Initializing PPO Agent...")
    agent = PPOAgent(
        input_dim=OBS_TOTAL,
        output_dim=OUTPUT_DIM,
        learning_rate=PPO['learning_rate']
    )
    gamma = PPO['gamma']

    logger.info(f"Starting PPO training: {num_episodes} episodes, "
                f"{len(stock_codes)} stocks in universe")

    episode_rewards = deque(maxlen=100)
    episode_total_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')

    # CSV logging
    csv_path = os.path.join(log_dir, 'ppo_training_log.csv')
    f_csv = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow([
        'episode', 'avg_r', 'total_r', 'steps', 'portfolio_value', 'cash',
        'holdings_count', 'trades', 'turnover', 'policy_loss', 'value_loss',
        'hold_pool_size', 'watch_pool_size', 'watch_entries'
    ])

    episode_log = {
        'episode': [], 'avg_r': [], 'total_r': [], 'steps': [],
        'portfolio_value': [], 'cash': [], 'holdings_count': [],
        'trades': [], 'turnover': [], 'policy_loss': [], 'value_loss': []
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()
        ep_states, ep_actions, ep_log_probs, ep_rewards, ep_values = [], [], [], [], []
        ep_stock_rewards_arr = []
        ep_dones = []  # track terminated/truncated per step
        ep_trades = 0
        ep_turnover = 0.0
        ep_rd_sum = {}
        done = False
        step = 0

        while not done:
            action, log_prob, value = agent.get_action(obs, stochastic=True)
            new_obs, reward, terminated, truncated, info = env.step(action.flatten())

            ep_states.append(obs)
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            ep_values.append(float(value))
            ep_stock_rewards_arr.append(info['per_stock_rewards'])
            ep_dones.append((terminated, truncated))
            ep_trades += int(info.get('trades_count', 0))
            ep_turnover += float(info.get('turnover', 0.0))

            obs = new_obs
            done = terminated or truncated
            step += 1

            rd = info.get('reward_debug', {})
            if rd:
                for k, v in rd.items():
                    ep_rd_sum[k] = ep_rd_sum.get(k, 0.0) + v

        episode_total_rewards.append(np.sum(ep_rewards))
        avg_total_r = np.mean(episode_total_rewards)

        # GAE-style: per-step advantages using value estimates
        n_steps = len(ep_rewards)

        values_arr = np.array(ep_values + [0.0], dtype=np.float32)
        advantages = np.zeros(n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n_steps)):
            terminated, truncated = ep_dones[t]
            terminal = terminated or truncated
            # last step of episode: bootstrap = 0
            mask = 0.0 if terminal else 1.0
            delta = ep_rewards[t] + gamma * values_arr[t + 1] * mask - values_arr[t]
            gae = delta + gamma * PPO.get('gae_lambda', 0.95) * mask * gae
            advantages[t] = gae

        # Value targets for critic
        returns_t = advantages + values_arr[:-1]

        # PPO update
        states_t = torch.FloatTensor(np.array(ep_states, dtype=np.float32)).to(agent.device)
        actions_t = torch.FloatTensor(np.array(ep_actions, dtype=np.float32)).to(agent.device)
        log_probs_t = torch.FloatTensor(np.array(ep_log_probs, dtype=np.float32)).to(agent.device)
        returns_t = torch.FloatTensor(returns_t).to(agent.device)
        advantages_t = torch.FloatTensor(advantages).to(agent.device)

        if torch.isnan(states_t).any() or torch.isnan(actions_t).any() or torch.isnan(log_probs_t).any():
            logger.warning("NaN in input data before update!")
            continue

        for _ in range(PPO['n_epochs']):
            loss_dict = agent.update(
                states_t, actions_t, log_probs_t,
                returns_t, advantages_t
            )

        episode_rewards.append(np.mean(ep_rewards))
        avg_r = np.mean(episode_rewards)
        avg_total_r = np.mean(episode_total_rewards)

        # CSV log every episode
        csv_writer.writerow([
            episode + 1, f"{avg_r:.6f}", f"{avg_total_r:.1f}", step,
            info['portfolio_value'], info['cash'], len(info['holdings']),
            ep_trades, f"{ep_turnover:.3f}",
            f"{loss_dict['policy_loss']:.6f}", f"{loss_dict['value_loss']:.6f}",
            info.get('hold_pool_size', 0), info.get('watch_pool_size', 0),
            info.get('watch_entries', 0)
        ])
        f_csv.flush()
        episode_log['episode'].append(episode + 1)
        episode_log['avg_r'].append(avg_r)
        episode_log['total_r'].append(avg_total_r)
        episode_log['steps'].append(step)
        episode_log['portfolio_value'].append(info['portfolio_value'])
        episode_log['cash'].append(info['cash'])
        episode_log['holdings_count'].append(len(info['holdings']))
        episode_log['trades'].append(ep_trades)
        episode_log['turnover'].append(ep_turnover)
        episode_log['policy_loss'].append(loss_dict['policy_loss'])
        episode_log['value_loss'].append(loss_dict['value_loss'])

        if (episode + 1) % 10 == 0:
            avg_turnover = ep_turnover / max(step, 1)
            avg_trades_per_day = ep_trades / max(step, 1)
            rd = info.get('reward_debug', {})
            logger.info(
                f"Ep {episode+1:3d}/{num_episodes}  "
                f"avg_r={avg_r:.3f}  "
                f"total_r={avg_total_r:.1f}  "
                f"steps={step}  "
                f"val=${info['portfolio_value']:,.0f}  "
                f"cash=${info['cash']:,.0f}  "
                f"eq=${info['portfolio_value']-info['cash']:,.0f}  "
                f"hold={len(info['holdings'])}  "
                f"hpool={info.get('hold_pool_size', 0)}  "
                f"wpool={info.get('watch_pool_size', 0)}  "
                f"wentry={info.get('watch_entries', 0)}  "
                f"trades={ep_trades}  "
                f"tpd={avg_trades_per_day:.2f}  "
                f"avg_turnover={avg_turnover:.3f}  "
                f"pl={loss_dict['policy_loss']:.4f}  "
                f"vl={loss_dict['value_loss']:.4f}"
            )
            if ep_rd_sum:
                rd_steps = max(step, 1)
                ep_rd_avg = {k: v / rd_steps for k, v in ep_rd_sum.items()}
                logger.info(f"  reward: sr={ep_rd_avg['step_return']:.4f} "
                            f"profit={ep_rd_avg['r_profit']:.4f} "
                            f"dd={ep_rd_avg['r_drawdown']:.4f} "
                            f"turnover={ep_rd_avg.get('r_turnover', 0.0):.4f} "
                            f"invalid={ep_rd_avg.get('r_invalid', 0.0):.4f} "
                            f"term={ep_rd_avg['terminal']:.4f} "
                            f"xcash={ep_rd_avg.get('r_excess_cash', 0.0):.4f} "
                            f"mtrades={ep_rd_avg.get('r_min_trades', 0.0):.4f} "
                            f"div={ep_rd_avg.get('r_diversity', 0.0):.4f}")
            stocks_str = ' '.join(sorted(info['holdings'].keys()))
            logger.info(f"  holdings: {stocks_str}")

        # Save best model + run validation every 50 episodes
        if avg_r > best_avg_reward:
            best_avg_reward = avg_r
            agent.save(os.path.join(MODEL_DIR, f'ppo_best{seed_tag}.pth'))

        if (episode + 1) % 50 == 0:
            agent.save(os.path.join(MODEL_DIR, f'ppo_ep{episode+1}{seed_tag}.pth'))
            _run_validation(agent, log_dir=log_dir, seed=seed, hold_pool_size=n_hold)

    f_csv.close()

    # Save training history
    np.savez(os.path.join(log_dir, 'ppo_training_history.npz'), **episode_log)

    agent.save(os.path.join(MODEL_DIR, f'ppo_final{seed_tag}.pth'))
    logger.info(f"Training complete. Best avg reward: {best_avg_reward:.3f}")
    return agent, best_avg_reward


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--max-cash-ratio', type=float, default=None,
                        help='Override MAX_CASH_RATIO in config')
    parser.add_argument('--excess-cash-penalty', type=float, default=None,
                        help='Override EXCESS_CASH_PENALTY in config')
    parser.add_argument('--diversity-window', type=int, default=None,
                        help='Override DIVERSITY_WINDOW in config')
    parser.add_argument('--diversity-penalty', type=float, default=None,
                        help='Override DIVERSITY_PENALTY in config')
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='Override PPO ent_coef in config')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override PPO learning_rate in config')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Suffix for log/model filenames')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Override number of training episodes')
    parser.add_argument('--hold-pool-size', type=int, default=None,
                        help='Override HOLD_POOL_SIZE (V2: hold pool = top N ROI holdings)')
    args = parser.parse_args()

    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    from pipeline.config import MAX_CASH_RATIO, EXCESS_CASH_PENALTY, PPO
    tag_parts = []
    if args.max_cash_ratio is not None:
        import pipeline.config as cfg
        cfg.MAX_CASH_RATIO = args.max_cash_ratio
        import rl_env.trading_env as te
        te.MAX_CASH_RATIO = args.max_cash_ratio
        tag_parts.append(f"mc{args.max_cash_ratio}")
    if args.excess_cash_penalty is not None:
        import pipeline.config as cfg
        cfg.EXCESS_CASH_PENALTY = args.excess_cash_penalty
        import rl_env.trading_env as te
        te.EXCESS_CASH_PENALTY = args.excess_cash_penalty
        tag_parts.append(f"ep{args.excess_cash_penalty}")
    if args.diversity_window is not None:
        import pipeline.config as cfg
        cfg.DIVERSITY_WINDOW = args.diversity_window
        import rl_env.trading_env as te
        te.DIVERSITY_WINDOW = args.diversity_window
        tag_parts.append(f"dw{args.diversity_window}")
    if args.diversity_penalty is not None:
        import pipeline.config as cfg
        cfg.DIVERSITY_PENALTY = args.diversity_penalty
        import rl_env.trading_env as te
        te.DIVERSITY_PENALTY = args.diversity_penalty
        tag_parts.append(f"dp{args.diversity_penalty}")
    if args.ent_coef is not None:
        import pipeline.config as cfg
        cfg.PPO['ent_coef'] = args.ent_coef
        tag_parts.append(f"ec{args.ent_coef}")
    if args.learning_rate is not None:
        import pipeline.config as cfg
        cfg.PPO['learning_rate'] = args.learning_rate
        tag_parts.append(f"lr{args.learning_rate}")
    if args.hold_pool_size is not None:
        import pipeline.config as cfg
        cfg.HOLD_POOL_SIZE = args.hold_pool_size
        import rl_env.trading_env as te
        te.HOLD_POOL_SIZE = args.hold_pool_size
        tag_parts.append(f"hp{args.hold_pool_size}")

    base_dir = os.path.dirname(__file__)
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    seed_tag = f'_seed{args.seed}' if args.seed is not None else ''
    run_tag = f"_{args.run_name}" if args.run_name else ('_' + '_'.join(tag_parts) if tag_parts else '')
    log_path = os.path.join(log_dir, f'ppo_train_{ts}{run_tag}{seed_tag}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'),
                  logging.StreamHandler()]
    )

    if tag_parts:
        logger.info("Overrides: " + ' | '.join(tag_parts))

    logger.info("Loading universe data...")
    stock_data = load_universe_data(max_stocks=100)
    stock_codes = get_active_stock_codes(stock_data)
    logger.info(f"Stock universe: {len(stock_codes)} stocks")

    agent, best_r = train_ppo(
        stock_data, stock_codes, seed=args.seed,
        num_episodes=args.max_episodes or 200,
        hold_pool_size=args.hold_pool_size,
    )

    summary_path = os.path.join(log_dir, f'sweep_results{run_tag}{seed_tag}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"run_name: {args.run_name or '_'.join(tag_parts)}\n")
        f.write(f"best_avg_reward: {best_r:.6f}\n")
        f.write(f"max_cash_ratio: {args.max_cash_ratio or MAX_CASH_RATIO}\n")
        f.write(f"excess_cash_penalty: {args.excess_cash_penalty or EXCESS_CASH_PENALTY}\n")
        f.write(f"ent_coef: {args.ent_coef or PPO['ent_coef']}\n")
        f.write(f"learning_rate: {args.learning_rate or PPO['learning_rate']}\n")
    logger.info("Done!")
