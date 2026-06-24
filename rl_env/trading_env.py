"""PPO 交易環境 - 動態候選池 + GRU Cache 查表"""
import json
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
from gymnasium import spaces
from pipeline.config import (
    REWARD_WEIGHTS, INITIAL_CAPITAL,
    COMMISSION_BUY, COMMISSION_SELL,
    CANDIDATE_POOL_SIZE, GRU_HIDDEN_SIZE,
    TIME_WINDOWS, N_PRED_TARGETS, GRU_CACHE_DIR,
    PER_STOCK_DIM, NUM_INDUSTRIES, TURNOVER_LIMIT,
    MIN_TRADE_PCT, MIN_CASH_RATIO, STOP_LOSS_PCT,
    HORIZON_WEIGHTS, TECH_INDUSTRIES, TECH_BOOST, MAX_POSITION_RATIO,
    MAX_CASH_RATIO, EXCESS_CASH_PENALTY,
    MIN_DAILY_TRADES, MIN_DAILY_TRADE_PENALTY,
    DIVERSITY_WINDOW, DIVERSITY_PENALTY,
    HOLD_POOL_SIZE, PRED_REWARD_CLIP,
)


# Column indices for pre-converted numpy arrays (OHLCV only)
_COL_CLOSE = 0
_COL_OPEN = 1
_COL_HIGH = 2
_COL_LOW = 3
_COL_VOLUME = 4


class MultiStockTradingEnv(gym.Env):
    """
    多股票交易環境 - 每日動態候選池 + GRU 離線查表
    觀察空間：CANDIDATE_POOL_SIZE 檔股票 × (GRU_features + horizon_preds + OHLCV)
    """
    metadata = {'render_mode': 'console'}

    def __init__(self, all_stocks=None, data=None,
                 gru_cache_dir=GRU_CACHE_DIR,
                 candidate_pool_size=CANDIDATE_POOL_SIZE,
                 initial_capital=INITIAL_CAPITAL,
                 reward_weights=None,
                 turnover_limit=TURNOVER_LIMIT,
                 render_mode=None,
                 hold_pool_size=None):
        self.all_stocks = all_stocks or []
        self.candidate_pool_size = candidate_pool_size
        self.hold_pool_size = hold_pool_size if hold_pool_size is not None else HOLD_POOL_SIZE
        assert self.hold_pool_size < self.candidate_pool_size, \
            f"hold_pool_size ({self.hold_pool_size}) must be < candidate_pool_size ({self.candidate_pool_size})"
        self.watch_pool_size = self.candidate_pool_size - self.hold_pool_size
        self.initial_capital = initial_capital
        self.data = data or {}
        self.reward_weights = reward_weights or REWARD_WEIGHTS
        self.render_mode = render_mode
        self.gru_cache_dir = gru_cache_dir
        self.turnover_limit = turnover_limit
        self.min_trade_pct = MIN_TRADE_PCT

        # Pre-convert DataFrames to numpy arrays for fast indexing
        self._preprocess_data()

        # Cache per stock: lazy load on first access
        self._gru_cache = {}

        # Per-stock feature dim (GRU features + horizon preds + env features)
        self._gru_feat_dim = GRU_HIDDEN_SIZE * len(TIME_WINDOWS)

        # Preprocess industry map
        self._load_industry_map()

        self.observation_dim = candidate_pool_size * PER_STOCK_DIM
        # V2 action: act[0:N]  = future return predictions % for each pool stock
        #            act[N]    = cash weight
        # PPO 給連續預估 (-30% ~ +30%)，環境解析為 softmax 機率 + 進場門檻 + 權重
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(candidate_pool_size + 1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,), dtype=np.float32
        )

        # V2 state: track hold/watch split for current pool
        self._pool_hold_idx = set()   # indices in self.pool that are hold
        self._pool_watch_idx = set()  # indices in self.pool that are watch

        self.max_drawdown = 0
        self.prev_drawdown = 0.0
        self.peak_value = 0
        self.action_history = []
        self.prev_portfolio_value = None
        self._last_reward_debug = {}
        self._cost_basis = {}
        self._buy_history = []  # recent buys for diversity penalty

    def _preprocess_data(self):
        """Convert all DataFrames to numpy arrays for fast indexing (OHLCV only)."""
        self._data_arr = {}
        for code, df in self.data.items():
            n = len(df)
            arr = np.empty((n, 5), dtype=np.float32)
            arr[:, _COL_CLOSE] = df['close'].values.astype(np.float32)
            arr[:, _COL_OPEN] = df['open'].values.astype(np.float32)
            arr[:, _COL_HIGH] = df['high'].values.astype(np.float32)
            arr[:, _COL_LOW] = df['low'].values.astype(np.float32)
            arr[:, _COL_VOLUME] = df['volume'].values.astype(np.float32)
            self._data_arr[code] = arr

    def _load_gru_cache(self, stock_code):
        if stock_code in self._gru_cache:
            return self._gru_cache[stock_code]
        path = os.path.join(self.gru_cache_dir, f'{stock_code}.npy')
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == np.object_:
                data = data.item()
            if isinstance(data, dict):
                cache = data['cache'].astype(np.float32)
            else:
                cache = data.astype(np.float32)
        else:
            # No cache yet: return zeros (will improve after GRU training)
            ndays = len(self.data.get(stock_code, []))
            cache = np.zeros((max(ndays, 1), self._gru_feat_dim + N_PRED_TARGETS),
                             dtype=np.float32)
        self._gru_cache[stock_code] = cache
        return cache

    def _load_industry_map(self):
        map_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'industry_map.json')
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            all_codes = sorted(set(raw.values()))  # e.g. ['01','02',...,'38']
            self._industry_to_idx = {c: i for i, c in enumerate(all_codes)}
            self._stock_industry = raw
            self._n_industries = NUM_INDUSTRIES  # 33 + 1 default
        else:
            self._industry_to_idx = {}
            self._stock_industry = {}
            self._n_industries = NUM_INDUSTRIES

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_capital
        self.holdings = {}
        self._cost_basis = {}
        self._buy_history = []
        self.initial_value = self.cash
        self.peak_value = self.initial_capital
        self.max_drawdown = 0
        self.prev_drawdown = 0.0
        self.day_idx = 0
        self.trades = []
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_portfolio_value = self.initial_capital
        self.pool = self._screen_pool(0)

        assert self.cash == self.initial_capital, f"cash mismatch: {self.cash} != {self.initial_capital}"
        assert len(self.holdings) == 0, f"holdings not cleared: {self.holdings}"
        assert not np.isnan(self._get_observation()).any(), "obs has NaN after reset"

        return self._get_observation(), {}

    HOLDING_CAP = 20  # deprecated: no longer used; pool cap is candidate_pool_size

    def _screen_pool(self, day_idx):
        """
        V2 候選池篩選（Hold / Watch 分段）：

        - Hold 池（前 hold_pool_size 名額）：
            持倉按 ROI 降冪排 → 取前 hold_pool_size 檔
            跌出 top N 的持倉 → 自動移入 Watch 池（不凍結）
        - Watch 池（後 watch_pool_size 名額）：
            從 universe（非 Hold）依 GRU score 補滿

        回傳：pool list（長度 CANDIDATE_POOL_SIZE）
        並更新 self._pool_hold_idx / self._pool_watch_idx
        """
        n = self.candidate_pool_size
        n_hold = self.hold_pool_size

        # ── 計算所有股票的 GRU score ──
        scores = {}
        for stock in self.all_stocks:
            if stock not in self._data_arr or day_idx >= len(self._data_arr[stock]):
                continue
            cache = self._load_gru_cache(stock)
            if day_idx >= len(cache):
                scores[stock] = 0.0
                continue

            row = cache[day_idx]
            feat_dim = self._gru_feat_dim
            score_nom = 0.0
            vol_penalty = 0.0
            for h, w in enumerate(HORIZON_WEIGHTS):
                cret = row[feat_dim + h * 4 + 0]      # close_ret
                hret = row[feat_dim + h * 4 + 2]      # high_ret
                lret = row[feat_dim + h * 4 + 3]      # low_ret
                score_nom += w * cret
                vol_penalty += w * abs(hret - lret)

            final_score = score_nom - 0.5 * vol_penalty
            # Tech boost
            if TECH_BOOST != 1.0 and self._stock_industry.get(stock, '') in TECH_INDUSTRIES:
                final_score *= TECH_BOOST
            scores[stock] = final_score

        # ── 持倉按 ROI 降冪 ──
        holdings_roi = []
        for stock in self.holdings:
            if stock not in self._data_arr or day_idx >= len(self._data_arr[stock]):
                continue
            close_price = self._data_arr[stock][day_idx, _COL_CLOSE]
            cost = self._cost_basis.get(stock, close_price)
            roi = close_price / cost - 1 if cost > 0 else 0.0
            holdings_roi.append((roi, stock))
        holdings_roi.sort(key=lambda x: -x[0])  # ROI descending

        # ── 區分 in-hold-top-N vs out-of-hold-top-N ──
        hold_codes = []   # 前 hold_pool_size 名額
        watch_from_holdings = []  # 跌出 top N 的持倉 → 進 watch 池
        for i, (_, stock) in enumerate(holdings_roi):
            if i < n_hold:
                hold_codes.append(stock)
            else:
                watch_from_holdings.append(stock)

        # ── Watch 池：先放跌出 top N 的持倉，再補 GRU top 非持艙 ──
        watch_codes = list(watch_from_holdings)
        watch_set = set(watch_codes)

        # 從 ranked（GRU score 排序）補滿 watch 池
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        for stock, _ in ranked:
            if len(watch_codes) >= self.watch_pool_size:
                break
            if stock in hold_codes or stock in watch_set:
                continue
            watch_codes.append(stock)
            watch_set.add(stock)

        # ── 組裝 pool：前 hold 後 watch ──
        pool = hold_codes + watch_codes
        # Pad if universe is small
        while len(pool) < n:
            pool.append(pool[-1] if pool else '')

        # 更新 hold/watch 索引
        self._pool_hold_idx = set(range(len(hold_codes)))
        self._pool_watch_idx = set(range(len(hold_codes), len(hold_codes) + len(watch_codes)))

        return pool[:n]

    def _get_observation(self):
        """建構觀察向量 (CANDIDATE_POOL_SIZE × per_stock_dim,) — numpy 直接填入"""
        n = self.candidate_pool_size
        obs = np.empty(n * PER_STOCK_DIM, dtype=np.float32)
        gru_offset = self._gru_feat_dim + N_PRED_TARGETS
        env_offset = 5  # OHLCV only
        ind_offset = self._n_industries

        for i, stock in enumerate(self.pool):
            base = i * PER_STOCK_DIM
            cache = self._load_gru_cache(stock)
            if self.day_idx < len(cache):
                obs[base:base + gru_offset] = np.nan_to_num(cache[self.day_idx])
            else:
                obs[base:base + gru_offset] = 0

            env_start = base + gru_offset
            if stock in self._data_arr and self.day_idx < len(self._data_arr[stock]):
                row = self._data_arr[stock][self.day_idx]
                env_vals = [
                    row[_COL_OPEN], row[_COL_HIGH], row[_COL_LOW],
                    row[_COL_CLOSE], row[_COL_VOLUME],
                ]
                obs[env_start:env_start + env_offset] = np.nan_to_num(env_vals, nan=0.0)
            else:
                obs[env_start:env_start + env_offset] = 0

            # Industry one-hot
            ind_start = env_start + env_offset
            one_hot = np.zeros(ind_offset, dtype=np.float32)
            ind_code = self._stock_industry.get(stock, None)
            if ind_code is not None:
                idx = self._industry_to_idx.get(ind_code)
                if idx is not None and idx < ind_offset - 1:
                    one_hot[idx] = 1.0
                else:
                    one_hot[-1] = 1.0
            else:
                one_hot[-1] = 1.0
            obs[ind_start:ind_start + ind_offset] = one_hot

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_gru_composite(self, day_idx, stock):
        if stock not in self._gru_cache:
            self._load_gru_cache(stock)
        cache = self._gru_cache.get(stock)
        if cache is None or day_idx >= len(cache):
            return 0.0
        composite = 0.0
        # Average close_ret across horizons as composite score
        for h in range(7):  # N_HORIZONS = 7
            idx = self._gru_feat_dim + h * 4  # close_ret for this horizon
            composite += cache[day_idx, idx] * HORIZON_WEIGHTS[h]
        return composite

    def step(self, action):
        """
        V2 動作語意：
        - act[0:n]: 50 檔的未來漲跌預估 %（連續，clip 到 [-PRED_REWARD_CLIP, +PRED_REWARD_CLIP]）
        - act[n]:   現金權重（連續，clip [0, 1]）

        環境解析：
        1. softmax_probs = softmax(act[0:n])     # 機率分布
        2. threshold = mean(softmax_probs) = 1/n  # 進場門檻
        3. 對每檔：target_weight = softmax_probs[i] × (1 - cash_weight)
        """
        # Snapshot prev state for per-stock PnL
        prev_holdings = self.holdings.copy()
        positions_changed = False
        executed_trades = 0
        invalid_actions = 0
        traded_value = 0.0
        stock_cash_flow = {}

        n = self.candidate_pool_size
        act = np.array(action, dtype=np.float32).flatten()
        if act.size < n + 1:
            act = np.pad(act, (0, n + 1 - act.size), constant_values=0.0)
        elif act.size > n + 1:
            act = act[:n + 1]

        if not np.isfinite(act).all():
            invalid_actions += 1
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

        # V2 動作語意：act[0:n] = 預估漲跌 %，act[n] = 現金權重
        preds = act[:n].astype(np.float32)
        cash_act = float(act[n])

        # Clip 預估範圍
        preds = np.clip(preds, -PRED_REWARD_CLIP, PRED_REWARD_CLIP)
        # Clip 現金權重 [0, 1]
        cash_weight = float(np.clip(cash_act, 0.0, 1.0))

        # Softmax 機率分布
        # 數值穩定性：減去最大值
        preds_shifted = preds - preds.max()
        exp_preds = np.exp(preds_shifted)
        softmax_probs = exp_preds / exp_preds.sum()
        # 進場門檻 = mean = 1/n
        entry_threshold = 1.0 / n

        # 目標權重 = softmax × (1 - cash_weight)
        target_stock_weights = softmax_probs * (1.0 - cash_weight)
        target_weights = np.zeros(n + 1, dtype=np.float32)
        target_weights[:n] = target_stock_weights
        target_weights[-1] = cash_weight

        # 記錄進場信號（給 log 用）
        watch_entries = int(np.sum(
            (softmax_probs > entry_threshold) & np.isin(np.arange(n), list(self._pool_watch_idx))
        ))

        # Current portfolio value and weights
        prices = np.zeros(n, dtype=np.float32)
        stock_values = np.zeros(n, dtype=np.float32)
        current_value = self.cash
        for i, stock in enumerate(self.pool):
            if stock not in self._data_arr:
                continue
            if self.day_idx >= len(self._data_arr[stock]):
                continue
            price = self._data_arr[stock][self.day_idx, _COL_CLOSE]
            prices[i] = price
            shares = self.holdings.get(stock, 0)
            stock_values[i] = shares * 1000 * price
            current_value += stock_values[i]

        if current_value <= 0:
            current_value = max(self.cash, 1.0)

        current_weights = np.zeros(n + 1, dtype=np.float32)
        current_weights[:n] = stock_values / current_value
        current_weights[-1] = self.cash / current_value

        # Apply turnover limit
        turnover = float(np.sum(np.abs(target_weights[:n] - current_weights[:n])))
        if self.turnover_limit is not None and turnover > self.turnover_limit:
            scale = self.turnover_limit / max(turnover, 1e-8)
            target_weights = current_weights + (target_weights - current_weights) * scale
            target_weights = np.clip(target_weights, 0.0, None)
            t_sum = target_weights.sum()
            if t_sum <= 0:
                invalid_actions += 1
                target_weights = np.zeros_like(target_weights)
                target_weights[-1] = 1.0
            else:
                target_weights = target_weights / t_sum
            turnover = float(np.sum(np.abs(target_weights[:n] - current_weights[:n])))

        # Enforce minimum cash reserve (after turnover limit so it can't be overridden)
        if target_weights[-1] < MIN_CASH_RATIO:
            stock_weights_sum = target_weights[:n].sum()
            if stock_weights_sum > 0:
                scale = (1.0 - MIN_CASH_RATIO) / stock_weights_sum
                target_weights[:n] *= scale
                target_weights[-1] = MIN_CASH_RATIO
            else:
                target_weights[-1] = 1.0
                target_weights[:n] = 0.0

        # Enforce maximum cash ratio (force minimum stock allocation)
        # Prevents degenerate "do nothing" policy collapse
        if target_weights[-1] > MAX_CASH_RATIO:
            remaining = 1.0 - MAX_CASH_RATIO
            stock_weights_sum = target_weights[:n].sum()
            if stock_weights_sum > 0:
                scale = remaining / stock_weights_sum
                target_weights[:n] *= scale
            else:
                target_weights[:n] = remaining / n
            target_weights[-1] = MAX_CASH_RATIO

        start_value = current_value
        target_values = target_weights[:n] * current_value
        target_cash = target_weights[-1] * current_value

        target_shares = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if prices[i] <= 0:
                continue
            target_shares[i] = int(target_values[i] / (prices[i] * 1000))

        # Filter negligible trades (< min_trade_pct of position)
        for i, stock in enumerate(self.pool):
            current = self.holdings.get(stock, 0)
            diff = abs(target_shares[i] - current)
            min_trade = max(1, int(current * self.min_trade_pct)) if current > 0 else 1
            if 0 < diff < min_trade:
                target_shares[i] = current

        # Max position size constraint: no single stock > MAX_POSITION_RATIO of portfolio
        for i in range(n):
            if prices[i] <= 0:
                continue
            max_value = current_value * MAX_POSITION_RATIO
            max_shares = int(max_value / (prices[i] * 1000))
            if target_shares[i] > max_shares:
                target_shares[i] = max_shares

        # Sell first to free cash
        for i, stock in enumerate(self.pool):
            if prices[i] <= 0:
                continue
            current_shares = self.holdings.get(stock, 0)
            desired_shares = target_shares[i]
            if desired_shares < current_shares:
                shares_to_sell = current_shares - desired_shares
                revenue = shares_to_sell * 1000 * prices[i] * (1 - COMMISSION_SELL)
                self.cash += revenue
                if desired_shares > 0:
                    self.holdings[stock] = desired_shares
                    # Cost basis stays the same for partial sells
                else:
                    self.holdings.pop(stock, None)
                    self._cost_basis.pop(stock, None)
                positions_changed = True
                executed_trades += 1
                traded_value += shares_to_sell * 1000 * prices[i]
                stock_cash_flow[stock] = stock_cash_flow.get(stock, 0) + revenue
                self.trades.append(('SELL', stock, prices[i], shares_to_sell))

        # ---- Cash management & stop-loss ----
        sold_composite_threshold = -float('inf')

        # Raise cash: if cash < MIN_CASH_RATIO, sell profitable stocks with worst GRU composite
        min_cash = self.initial_capital * MIN_CASH_RATIO
        if self.cash < min_cash:
            candidates = []
            fallback_candidates = []
            for stock, shares in list(self.holdings.items()):
                if stock not in self._data_arr or self.day_idx >= len(self._data_arr[stock]):
                    continue
                price = self._data_arr[stock][self.day_idx, _COL_CLOSE]
                cost = self._cost_basis.get(stock, price)
                composite = self._get_gru_composite(self.day_idx, stock)
                if price > cost * 1.001:  # at least 0.1% profit
                    candidates.append((composite, stock, shares, price, cost))
                elif cost > 0:
                    loss_ratio = (cost - price) / cost
                    fallback_candidates.append((loss_ratio, composite, stock, shares, price, cost))

            # Sort profitable by GRU composite ascending (worst first)
            candidates.sort(key=lambda x: x[0])
            # Sort loss-making by loss ratio ascending (least loss first)
            fallback_candidates.sort(key=lambda x: (-x[1], x[0]))

            for composite, stock, shares, price, cost in candidates:
                if self.cash >= min_cash:
                    break
                revenue = shares * 1000 * price * (1 - COMMISSION_SELL)
                self.cash += revenue
                self.holdings.pop(stock, None)
                self._cost_basis.pop(stock, None)
                sold_composite_threshold = max(sold_composite_threshold, composite)
                executed_trades += 1
                traded_value += shares * 1000 * price
                positions_changed = True
                stock_cash_flow[stock] = stock_cash_flow.get(stock, 0) + revenue
                self.trades.append(('SELL(cash)', stock, price, shares))

            # Fallback: sell least-bad stocks if still low on cash
            for loss_ratio, composite, stock, shares, price, cost in fallback_candidates:
                if self.cash >= min_cash:
                    break
                revenue = shares * 1000 * price * (1 - COMMISSION_SELL)
                self.cash += revenue
                self.holdings.pop(stock, None)
                self._cost_basis.pop(stock, None)
                sold_composite_threshold = max(sold_composite_threshold, composite)
                executed_trades += 1
                traded_value += shares * 1000 * price
                positions_changed = True
                stock_cash_flow[stock] = stock_cash_flow.get(stock, 0) + revenue
                self.trades.append(('SELL(cash/fallback)', stock, price, shares))

        # Stop-loss: sell losers that GRU predicts will keep falling
        for stock, shares in list(self.holdings.items()):
            if stock not in self._data_arr or self.day_idx >= len(self._data_arr[stock]):
                continue
            price = self._data_arr[stock][self.day_idx, _COL_CLOSE]
            cost = self._cost_basis.get(stock, price)
            if cost <= 0:
                continue
            loss_pct = (cost - price) / cost
            if loss_pct > STOP_LOSS_PCT:
                composite = self._get_gru_composite(self.day_idx, stock)
                if composite < 0:  # GRU predicts further decline
                    revenue = shares * 1000 * price * (1 - COMMISSION_SELL)
                    self.cash += revenue
                    self.holdings.pop(stock, None)
                    self._cost_basis.pop(stock, None)
                    sold_composite_threshold = max(sold_composite_threshold, composite)
                    executed_trades += 1
                    traded_value += shares * 1000 * price
                    positions_changed = True
                    stock_cash_flow[stock] = stock_cash_flow.get(stock, 0) + revenue
                    self.trades.append(('SELL(stop)', stock, price, shares))

        # Prevent re-buy of stocks sold for cash/stop-loss
        if sold_composite_threshold > -float('inf'):
            for i, stock in enumerate(self.pool):
                if stock not in self.holdings and target_shares[i] > 0:
                    target_shares[i] = 0

        # Buy to reach target
        for i, stock in enumerate(self.pool):
            if prices[i] <= 0:
                continue
            current_shares = self.holdings.get(stock, 0)
            desired_shares = target_shares[i]
            if desired_shares > current_shares:
                # Buy protection: only buy if GRU composite >= sold stocks' threshold
                if sold_composite_threshold > -float('inf'):
                    buy_composite = self._get_gru_composite(self.day_idx, stock)
                    if buy_composite < sold_composite_threshold - 0.001:
                        continue
                shares_to_buy = desired_shares - current_shares
                unit_cost = prices[i] * 1000 * (1 + COMMISSION_BUY)
                max_affordable = int(self.cash / unit_cost) if unit_cost > 0 else 0
                if shares_to_buy > max_affordable:
                    shares_to_buy = max_affordable
                # Cash floor: don't spend below MIN_CASH_RATIO
                spendable_after_floor = max(0, self.cash - min_cash)
                cash_floor_max = int(spendable_after_floor / unit_cost) if unit_cost > 0 else 0
                limited_by_floor = shares_to_buy > cash_floor_max
                if limited_by_floor:
                    shares_to_buy = cash_floor_max
                if shares_to_buy > 0:
                    cost = shares_to_buy * unit_cost
                    self.cash -= cost
                    old_shares = current_shares
                    old_cost = self._cost_basis.get(stock, 0.0)
                    new_shares = current_shares + shares_to_buy
                    self.holdings[stock] = new_shares
                    self._cost_basis[stock] = (old_cost * old_shares + prices[i] * shares_to_buy) / new_shares
                    positions_changed = True
                    executed_trades += 1
                    traded_value += shares_to_buy * 1000 * prices[i]
                    stock_cash_flow[stock] = stock_cash_flow.get(stock, 0) - cost
                    self.trades.append(('BUY', stock, prices[i], shares_to_buy))
                    # Track buy for diversity penalty
                    self._buy_history.append(stock)
                elif not limited_by_floor:
                    invalid_actions += 1

        # Calculate current portfolio value after trades
        current_value = self.cash
        for stock, shares in self.holdings.items():
            if stock in self._data_arr and self.day_idx < len(self._data_arr[stock]):
                price = self._data_arr[stock][self.day_idx, _COL_CLOSE]
                current_value += shares * 1000 * price

        # Per-stock rewards (V2: returns position_weights)
        per_stock_returns, position_weights = self._compute_per_stock_rewards(
            prev_holdings, stock_cash_flow)

        # Trim buy history to diversity window
        if len(self._buy_history) > DIVERSITY_WINDOW:
            self._buy_history = self._buy_history[-DIVERSITY_WINDOW:]

        self.day_idx += 1
        done = self.day_idx >= self.max_days - 1 or current_value < self.initial_capital * 0.1

        turnover = traded_value / max(start_value, 1.0)
        reward = self._calculate_reward(
            current_value,
            per_stock_returns,
            position_weights,
            done,
            executed_trades=executed_trades,
            invalid_actions=invalid_actions,
            turnover=turnover,
        )

        self.peak_value = max(self.peak_value, current_value)
        current_drawdown = max(0, (self.peak_value - current_value) / self.peak_value)
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Refresh pool for next step
        self.pool = self._screen_pool(self.day_idx)

        info = {
            'portfolio_value': current_value,
            'cash': self.cash,
            'holdings': self.holdings.copy(),
            'drawdown': current_drawdown,
            'trades_count': executed_trades,
            'daily_trades': executed_trades,
            'turnover': traded_value / max(start_value, 1.0),
            'reward_debug': self._last_reward_debug,
            'per_stock_rewards': per_stock_returns.copy(),
            'position_weights': position_weights.copy(),
            'hold_pool_size': len(self._pool_hold_idx),
            'watch_pool_size': len(self._pool_watch_idx),
            'watch_entries': watch_entries,
        }

        self.last_action = np.array(action, dtype=np.float32)
        return self._get_observation(), reward, done, False, info

    def _compute_per_stock_rewards(self, prev_holdings, stock_cash_flow):
        """
        V2 + B：回傳 (per_stock_returns, position_weights)
          - per_stock_returns[i]: 股票 i 當日價格變動率 % (clip ±10%)
          - position_weights[i]: 上一步持倉 i 占組合比重 (shares * 1000 * price / prev_portfolio)
        """
        n_stocks = self.candidate_pool_size
        per_stock_returns = np.zeros(n_stocks, dtype=np.float32)
        position_weights = np.zeros(n_stocks, dtype=np.float32)
        prev_portfolio = self.prev_portfolio_value if self.prev_portfolio_value else self.initial_capital
        if prev_portfolio <= 0:
            return per_stock_returns, position_weights

        for i, stock in enumerate(self.pool):
            if stock not in self._data_arr:
                continue
            if self.day_idx >= len(self._data_arr[stock]):
                continue

            curr_price = self._data_arr[stock][self.day_idx, _COL_CLOSE]
            prev_idx = max(0, self.day_idx - 1)
            prev_price = self._data_arr[stock][prev_idx, _COL_CLOSE]
            if prev_price <= 0:
                continue

            old_shares = prev_holdings.get(stock, 0)

            # 持倉部位價值（用昨日收盤價）
            position_value = old_shares * 1000 * prev_price
            position_weights[i] = position_value / prev_portfolio

            # 該持倉的當日價格變動率
            price_return = (curr_price - prev_price) / prev_price
            per_stock_returns[i] = price_return * 100  # 轉成 % scale

        per_stock_returns = np.clip(per_stock_returns, -10.0, 10.0)
        return per_stock_returns, position_weights

    def _calculate_reward(self, current_value, per_stock_returns, position_weights,
                          done=False, executed_trades=0, invalid_actions=0, turnover=0.0):
        if self.prev_portfolio_value is not None and self.prev_portfolio_value > 0:
            step_return = (current_value - self.prev_portfolio_value) / self.prev_portfolio_value * 100
        else:
            step_return = 0.0

        # V2 + B：r_profit = 持艙加權（只算實際持艙部位的 PnL）
        #          = Σ(per_stock_return[i] × position_weight[i])
        # 對比 V1 的 per_stock_rewards.mean()，這個讓 PPO 學會
        # 「真的拿部位賺錢」，不會被「挑最強 3 隻」給誘惑。
        r_profit = float(np.sum(per_stock_returns * position_weights)) * self.reward_weights['profit']

        # Incremental drawdown penalty (only penalize when drawdown worsens)
        drawdown = max(0, (self.peak_value - current_value) / self.peak_value)
        drawdown_increment = max(0, drawdown - self.prev_drawdown)
        r_drawdown = -self.reward_weights['drawdown'] * drawdown_increment
        self.prev_drawdown = drawdown

        # Turnover penalty (discourage excessive churning)
        turnover_penalty = self.reward_weights.get('turnover_penalty', 0.0)
        r_turnover = -turnover_penalty * turnover

        # Invalid action penalty
        invalid_action_penalty = self.reward_weights.get('invalid_action_penalty', 0.0)
        r_invalid = -invalid_action_penalty * invalid_actions

        # Penalty for holding too much cash (> MAX_CASH_RATIO of portfolio)
        cash_ratio = self.cash / max(current_value, 1.0)
        excess_cash = max(0.0, cash_ratio - MAX_CASH_RATIO)
        r_excess_cash = -EXCESS_CASH_PENALTY * excess_cash

        # Minimum daily trades penalty (encourage at least MIN_DAILY_TRADES/day)
        trade_shortfall = max(0, MIN_DAILY_TRADES - executed_trades)
        r_min_trades = -MIN_DAILY_TRADE_PENALTY * trade_shortfall

        # Diversity penalty: penalize repeating the same stock buy in the recent window
        # _buy_history is trimmed in step() before this is called
        r_diversity = 0.0
        if len(self._buy_history) > 1:
            from collections import Counter
            counts = Counter(self._buy_history)
            repeat_count = sum(c - 1 for c in counts.values() if c > 1)
            r_diversity = -DIVERSITY_PENALTY * repeat_count

        # Terminal bonus: reward based on total return over entire episode
        r_terminal = 0.0
        if done:
            total_return = (current_value / self.initial_capital) - 1.0
            r_terminal = total_return * self.reward_weights['profit']

        self._last_reward_debug = dict(
            step_return=step_return, r_profit=r_profit,
            r_drawdown=r_drawdown, r_turnover=r_turnover,
            terminal=r_terminal, r_invalid=r_invalid,
            r_excess_cash=r_excess_cash, r_min_trades=r_min_trades,
            r_diversity=r_diversity,
        )

        self.prev_portfolio_value = current_value

        return (r_profit + r_drawdown + r_turnover + r_invalid
                + r_terminal + r_excess_cash + r_min_trades
                + r_diversity)

    @property
    def max_days(self):
        if not self.data or not self.all_stocks:
            return 300
        lengths = [len(self._data_arr[s]) for s in self.all_stocks
                   if s in self._data_arr and len(self._data_arr[s]) > 0]
        return min(lengths) if lengths else 300

    def render(self):
        print(f"Day {self.day_idx}/{self.max_days}")
        print(f"Cash: {self.cash:,.0f}")
        print(f"Holdings: {self.holdings}")
        value = self.cash
        for stock, shares in self.holdings.items():
            if stock in self._data_arr and self.day_idx < len(self._data_arr[stock]):
                value += shares * 1000 * self._data_arr[stock][self.day_idx, _COL_CLOSE]
        print(f"Value: {value:,.0f}")
