"""Daily pipeline orchestrator — fetch → gru → trade"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime

from pipeline.config import (
    INITIAL_CAPITAL, OBS_TOTAL, OUTPUT_DIM,
    CANDIDATE_POOL_SIZE, GRU_CACHE_DIR,
    MAX_CASH_RATIO, MIN_CASH_RATIO,
    PER_STOCK_DIM, NUM_INDUSTRIES,
    GRU_HIDDEN_SIZE, TIME_WINDOWS, N_HORIZONS,
    COMMISSION_SELL, COMMISSION_BUY,
    HOLD_POOL_SIZE, PRED_REWARD_CLIP,
    HORIZON_WEIGHTS, TECH_INDUSTRIES, TECH_BOOST,
    PORTFOLIO_STATE_FILE, PIPELINE_STATE_FILE,
)
from pipeline import _utils
from model.ppo_agent import PPOAgent
from pipeline.data_processor import engineer_features
from data.stock_api import Buy_Stock, Sell_Stock, Get_User_Stocks

try:
    import twstock
    HAS_TWSTOCK = True
except ImportError:
    HAS_TWSTOCK = False
    twstock = None

_PER_STOCK_OBS_GRU = _utils.CACHE_FEAT_DIM
_OBS_ENV = 5  # OHLCV only — matches PER_STOCK_DIM allocation
_OBS_IND = NUM_INDUSTRIES

STATE_FILE = PORTFOLIO_STATE_FILE

# Module-level GRU cache: 避免跨 LiveTrader 實例重複 disk I/O
_GRU_MEM_CACHE = {}  # code → np.ndarray

os.makedirs('logs', exist_ok=True)
log = _utils.setup_logger('pipeline')


def phase_fetch(universe_size=None):
    """Download API data for top stocks + any held stocks.

    B3 fix: universe 範圍由 top 60 改為 price_data 全檔（或指定 n），
    避免錯過新上市/高分的低排名股票。
    A3 fix: 同時寫入 data/price_data/{code}.csv（PPO 訓練從這讀）。
    """
    log.info("=" * 60)
    log.info("PHASE 1: API Data Fetch")
    log.info("=" * 60)

    if universe_size is None:
        universe_size = _utils.TOP_N
    universe = _utils.get_top_stocks(universe_size)
    if not universe:
        log.error("  No universe stocks from CSV")
        return False
    log.info(f"  Universe: {len(universe)} stocks (top {universe_size} from price_data)")

    # B3: 加上持倉（API 來源），確保持有股票一定更新
    try:
        from data.stock_api import Get_User_Stocks
        account, password = _utils.get_credentials()
        raw = Get_User_Stocks(account, password)
        held_codes = set()
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    code = item.get('stock_code_id', '')
                    if code:
                        held_codes.add(code)
        if held_codes:
            universe = list(set(universe) | held_codes)
            log.info(f"  + {len(held_codes)} held stocks → total {len(universe)}")
    except Exception as e:
        log.debug(f"  no holdings fetch: {e}")

    need_fetch = []
    for code in universe:
        cache_path = os.path.join(_utils.CACHE_DIR, f'{code}.pkl')
        if not os.path.exists(cache_path):
            need_fetch.append(code)
        else:
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                latest = pd.to_datetime(df['date'].max())
                if latest.date() < datetime.now().date():
                    need_fetch.append(code)
            except Exception:
                need_fetch.append(code)

    if not need_fetch:
        log.info("  All stocks up-to-date in cache")
        # C2 fix: 不再 fallback 到 universe[:10]，直接 return
        log.info("=" * 60)
        return True

    log.info(f"  Fetching {len(need_fetch)} stocks...")
    fresh = _utils.fetch_parallel(need_fetch, days_back=120)
    saved = 0
    for code, df in fresh.items():
        dst = os.path.join(_utils.CACHE_DIR, f'{code}.pkl')
        if os.path.exists(dst):
            try:
                with open(dst, 'rb') as f:
                    old = pickle.load(f)
                old['date'] = pd.to_datetime(old['date'])
                df['date'] = pd.to_datetime(df['date'])
                combined = pd.concat([old, df]).drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
                df = combined
            except Exception:
                pass
        with open(dst, 'wb') as f:
            pickle.dump(df, f)
        # A3 fix: 同步寫入 data/price_data/{code}.csv（PPO 訓練從這讀）
        _utils.append_to_price_data(code, df)
        saved += 1

    log.info(f"  Saved/updated {saved} stock cache files (+ price_data CSV)")
    log.info("=" * 60)
    return True


# ─────────────────────────────────────────────
# Phase 2: GRU cache update (function now in _utils)
# ─────────────────────────────────────────────


def phase_gru():
    """Recompute GRU cache for stocks with new API data."""
    log.info("=" * 60)
    log.info("PHASE 2: GRU Cache Update")
    log.info("=" * 60)

    import torch
    from datetime import datetime
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")

    today_str = datetime.now().strftime('%Y-%m-%d')

    mdl = _utils.load_gru_model(device)
    if mdl is None:
        return False

    needed = _utils.get_stocks_needing_gru_update()
    if not needed:
        log.info("  No stocks need GRU update")
        return True

    log.info(f"  {len(needed)} stocks need update")
    from sklearn.preprocessing import StandardScaler

    stock_features = []
    skipped = 0
    for code in needed:
        df = _utils.load_stock_combined_data(code)
        if df is None:
            skipped += 1
            continue

        df = engineer_features(df)
        df.dropna(subset=_utils.FEATURE_COLS, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) < _utils.MAX_W + 5:
            skipped += 1
            continue

        features = df[_utils.FEATURE_COLS].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        features = scaler.fit_transform(features).astype(np.float32)
        stock_features.append((code, features))

    log.info(f"  Collected {len(stock_features)} stock features ({skipped} skipped)")

    today_str = datetime.now().strftime('%Y-%m-%d')
    done = 0
    for idx, (code, features) in enumerate(stock_features):
        cache = _utils.compute_gru_for_stock(features, mdl, device)
        if cache is not None:
            tmp_path = os.path.join(GRU_CACHE_DIR, f'.tmp_{code}.npy')
            out_path = os.path.join(GRU_CACHE_DIR, f'{code}.npy')
            gru_data = {'cache': cache, 'gru_date': today_str}
            np.save(tmp_path, gru_data)
            os.replace(tmp_path, out_path)
            done += 1
        if (idx + 1) % 20 == 0:
            log.info(f"  Updated {idx+1}/{len(stock_features)} (skipped {skipped})")
            torch.cuda.empty_cache()

    del mdl
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log.info(f"  Done: {done} updated, {skipped} skipped")
    log.info("=" * 60)
    _GRU_MEM_CACHE.clear()
    return True


# ─────────────────────────────────────────────
# Phase 3: Trading
# ─────────────────────────────────────────────

# load_industry_map is now in _utils


class Proposal:
    def __init__(self, date, cash, port_value, cash_ratio, buys, sells):
        self.date = date
        self.cash = cash
        self.port_value = port_value
        self.cash_ratio = cash_ratio
        self.buys = buys      # list of {'code', 'lots', 'price', 'close'}
        self.sells = sells    # list of {'code', 'lots', 'price', 'close'}


class LiveTrader:
    def __init__(self, account, password, ppo_path='checkpoints/ppo_best.pth',
                 candidate_pool_size=CANDIDATE_POOL_SIZE, initial_capital=INITIAL_CAPITAL):
        self.account = account
        self.password = password
        self.initial_capital = initial_capital
        self.candidate_pool_size = candidate_pool_size

        self.agent = PPOAgent(input_dim=OBS_TOTAL, output_dim=OUTPUT_DIM)
        self.agent.load(ppo_path)
        self.agent.actor_critic.eval()

        self.stock_industry, self.industry_to_idx = _utils.load_industry_map()
        self.trades = []
        # B2 fix: 成本基線用於 ROI 排序（V2 持艙按 ROI 降冪）
        self.cost_basis = {}
        self._load_portfolio_state()

    def _load_portfolio_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    s = json.load(f)
                self.cash = s.get('cash', self.initial_capital)
                self.holdings = s.get('holdings', {})
                self.pending_buys = s.get('pending_buys', {})
                self.pending_sells = s.get('pending_sells', {})
                self.confirmed_trades = s.get('confirmed_trades', 0)
                # B2: 讀成本基線（如果有）
                self.cost_basis = s.get('cost_basis', {})
                # B2: 偵測負現金並 reset
                if self.cash < 0:
                    log.warning(f"  ⚠️ 偵測到負現金 cash={self.cash:.0f}，重置為初始資本")
                    self.cash = self.initial_capital
                log.info(f"  State: cash={self.cash:.0f}, {len(self.holdings)} holdings, "
                         f"{len(self.pending_buys)} pend_buy, {len(self.pending_sells)} pend_sell, "
                         f"{self.confirmed_trades} confirmed trades")
            except Exception as e:
                log.warning(f"  State load failed: {e}, starting fresh")
                self._reset_state()
        else:
            self._reset_state()

    def _reset_state(self):
        self.cash = self.initial_capital
        self.holdings = {}
        self.pending_buys = {}
        self.pending_sells = {}
        self.confirmed_trades = 0
        self.cost_basis = {}

    def _save_portfolio_state(self):
        state = {
            'cash': self.cash,
            'holdings': self.holdings,
            'pending_buys': self.pending_buys,
            'pending_sells': self.pending_sells,
            'confirmed_trades': self.confirmed_trades,
            'cost_basis': self.cost_basis,
            'updated_at': datetime.now().isoformat(),
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_gru_cache(self, code, target_len):
        if code in _GRU_MEM_CACHE:
            return _GRU_MEM_CACHE[code]
        p = os.path.join(GRU_CACHE_DIR, f'{code}.npy')
        if os.path.exists(p):
            try:
                data = np.load(p, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.dtype == np.object_:
                    data = data.item()
                if isinstance(data, dict):
                    arr = data['cache'].astype(np.float32)
                else:
                    arr = data.astype(np.float32)
                _GRU_MEM_CACHE[code] = arr
                return arr
            except Exception as e:
                log.warning(f"  _load_gru_cache {code}: {e}")
        return np.zeros((target_len, _PER_STOCK_OBS_GRU), dtype=np.float32)

    def _get_cache_idx(self, cache_len, df_len, day_idx):
        """Align cache index to API data index.
           Cache was built from combined CSV+API data (longer than API-only).
           The last `df_len` rows of cache correspond to the API data period."""
        if cache_len > df_len:
            offset = cache_len - df_len
            return min(offset + day_idx, cache_len - 1)
        return min(day_idx, cache_len - 1)

    def _fetch_realtime_prices(self, codes):
        """用 twstock.realtime.get() 批次抓即時股價。回傳 {code: {open, high, low, close}}"""
        if not HAS_TWSTOCK or not codes:
            return {}
        realtime = {}
        # twstock batch limit ~50，分批
        batch_size = 40
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            try:
                result = twstock.realtime.get(batch)
                if not result.get('success', False):
                    log.warning(f"  twstock realtime failed: {result.get('msg', 'unknown')}")
                    continue
                for code, data in result.items():
                    if code == 'success':
                        continue
                    rt = data.get('realtime', {})
                    def _to_float(v):
                        if v is None or v == '' or v == '-':
                            return None
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            return None
                    o = _to_float(rt.get('open') or rt.get('o'))
                    h = _to_float(rt.get('high') or rt.get('h'))
                    lo = _to_float(rt.get('low') or rt.get('l'))
                    c = _to_float(rt.get('latest_trade_price') or rt.get('z'))
                    if c is None or c <= 0:
                        bids = rt.get('best_bid_price', [])
                        if bids and bids[0] and bids[0] != '-':
                            c = _to_float(bids[0])
                    if c is None or c <= 0:
                        c = o
                    if o is not None and o > 0:
                        realtime[code] = {
                            'open': o,
                            'high': h if h and h > 0 else o,
                            'low': lo if lo and lo > 0 else o,
                            'close': c if c and c > 0 else o,
                        }
            except Exception as e:
                log.warning(f"  twstock batch {i}:{i+batch_size} error: {e}")
        if realtime:
            log.info(f"  Realtime prices fetched: {len(realtime)} stocks")
        return realtime

    def _build_observation(self, stock_data, pool, day_idx, realtime_prices=None):
        n = self.candidate_pool_size
        obs = np.zeros(n * PER_STOCK_DIM, dtype=np.float32)
        if realtime_prices is None:
            realtime_prices = {}
        for i, code in enumerate(pool[:n]):
            base = i * PER_STOCK_DIM
            df = stock_data.get(code)
            df_len = len(df) if df is not None else day_idx + 1
            cache = self._load_gru_cache(code, df_len)
            cache_idx = self._get_cache_idx(len(cache), df_len, day_idx) if len(cache) > 0 else 0
            obs[base:base + _PER_STOCK_OBS_GRU] = cache[cache_idx]
            env_start = base + _PER_STOCK_OBS_GRU
            # 即時價格優先，fallback 到 api_cache
            rt = realtime_prices.get(code)
            if rt:
                obs[env_start] = rt['open']
                obs[env_start + 1] = rt['high']
                obs[env_start + 2] = rt['low']
                obs[env_start + 3] = rt['close']
                obs[env_start + 4] = 0  # 即時 API 沒有 volume
            elif df is not None and day_idx < len(df):
                r = df.iloc[day_idx]
                obs[env_start] = r.get('open', 0)
                obs[env_start + 1] = r.get('high', 0)
                obs[env_start + 2] = r.get('low', 0)
                obs[env_start + 3] = r.get('close', 0)
                vol = r.get('volume', r.get('Trading_Volume', r.get('capacity', 0)))
                obs[env_start + 4] = vol
            ind_start = env_start + _OBS_ENV
            oh = np.zeros(_OBS_IND, dtype=np.float32)
            ind = self.stock_industry.get(code)
            if ind and ind in self.industry_to_idx:
                oh[min(self.industry_to_idx[ind], _OBS_IND - 1)] = 1.0
            else:
                oh[-1] = 1.0
            obs[ind_start:ind_start + _OBS_IND] = oh
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _screen_pool(self, stock_data, day_idx, current_holdings, hold_pool_size=None):
        """V2 hold/watch 池（跟 trading_env._screen_pool 對齊）"""
        n = self.candidate_pool_size
        n_hold = hold_pool_size if hold_pool_size is not None else HOLD_POOL_SIZE
        n_hold = min(n_hold, n - 1)
        n_watch = n - n_hold

        # ── 計算 GRU score（多 horizon 加權 + vol penalty + tech boost）──
        scores = {}
        for code, df in stock_data.items():
            if day_idx >= len(df):
                continue
            cache = self._load_gru_cache(code, len(df))
            if len(cache) == 0:
                scores[code] = 0.0
                continue
            cache_idx = self._get_cache_idx(len(cache), len(df), day_idx)
            if cache_idx >= len(cache):
                scores[code] = 0.0
                continue
            row = cache[cache_idx]
            feat_dim = GRU_HIDDEN_SIZE * len(TIME_WINDOWS)
            score_nom = 0.0
            vol_penalty = 0.0
            for h, w in enumerate(HORIZON_WEIGHTS):
                cret = row[feat_dim + h * 4 + 0]
                hret = row[feat_dim + h * 4 + 2]
                lret = row[feat_dim + h * 4 + 3]
                score_nom += w * cret
                vol_penalty += w * abs(hret - lret)
            final_score = score_nom - 0.5 * vol_penalty
            if TECH_BOOST != 1.0 and self.stock_industry.get(code, '') in TECH_INDUSTRIES:
                final_score *= TECH_BOOST
            scores[code] = final_score

        # ── 持倉按 ROI 降冪 ──
        holdings_roi = []
        for stock in current_holdings:
            if stock not in stock_data or day_idx >= len(stock_data[stock]):
                continue
            close_price = stock_data[stock].iloc[day_idx]['close']
            # 從 holdings 抓成本（如果有）；沒有的話用 close
            cost = self.cost_basis.get(stock, close_price)
            roi = close_price / cost - 1 if cost > 0 else 0.0
            holdings_roi.append((roi, stock))
        holdings_roi.sort(key=lambda x: -x[0])

        # ── 區分 in-hold-top-N vs out-of-hold-top-N ──
        hold_codes = []
        watch_from_holdings = []
        for i, (_, stock) in enumerate(holdings_roi):
            if i < n_hold:
                hold_codes.append(stock)
            else:
                watch_from_holdings.append(stock)

        # ── Watch 池：先放跌出 top N 的持倉，再補 GRU top 非持艙 ──
        watch_codes = list(watch_from_holdings)
        watch_set = set(watch_codes)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        for stock, _ in ranked:
            if len(watch_codes) >= n_watch:
                break
            if stock in hold_codes or stock in watch_set:
                continue
            watch_codes.append(stock)
            watch_set.add(stock)

        pool = hold_codes + watch_codes
        while len(pool) < n:
            pool.append(pool[-1] if pool else (list(stock_data.keys())[0] if stock_data else ''))
        return pool[:n]

    def _compute_target(self, obs, port_value, prices):
        """V2 動作解析（跟 training_env.step 對齊）：
        - act[0:n]: 預估漲跌% (clip ±PRED_REWARD_CLIP)
        - act[n]:   cash weight
        - softmax(act[:n]) × (1 - cash_weight) → target_stock_weight
        """
        action, _, _ = self.agent.get_action(obs, stochastic=False)
        n = self.candidate_pool_size
        act = np.array(action, dtype=np.float32).flatten()
        if act.size < n + 1:
            act = np.pad(act, (0, n + 1 - act.size), constant_values=0.0)
        elif act.size > n + 1:
            act = act[:n + 1]

        preds = act[:n].astype(np.float32)
        preds = np.clip(preds, -PRED_REWARD_CLIP, PRED_REWARD_CLIP)
        cash_weight = float(np.clip(act[n], 0.0, 1.0))

        # Softmax
        preds_shifted = preds - preds.max()
        exp_preds = np.exp(preds_shifted)
        softmax_probs = exp_preds / exp_preds.sum()

        # target weight = softmax × (1 - cash_weight)
        target_stock_weights = softmax_probs * (1.0 - cash_weight)
        w = np.zeros(n + 1, dtype=np.float32)
        w[:n] = target_stock_weights
        w[-1] = cash_weight

        # Cash 上下限 clamp
        if w[-1] < MIN_CASH_RATIO:
            sw = w[:n].sum()
            if sw > 0:
                w[:n] *= (1.0 - MIN_CASH_RATIO) / sw
                w[-1] = MIN_CASH_RATIO
        if w[-1] > MAX_CASH_RATIO:
            rem = 1.0 - MAX_CASH_RATIO
            sw = w[:n].sum()
            if sw > 0:
                w[:n] *= rem / sw
            w[-1] = MAX_CASH_RATIO

        target = {}
        for i, code in enumerate(self.pool[:n]):
            if prices.get(code, 0) > 0:
                target[code] = int(w[i] * port_value / (prices[code] * 1000))
        return target, w[-1]

    def compute_proposal(self):
        """Compute proposed trades without executing. Returns Proposal or None."""
        today = datetime.now()
        log.info(f"\n{'─'*60}")
        log.info(f"PHASE 3: Trade Execution — {today.strftime('%Y-%m-%d')}")
        log.info(f"{'─'*60}")

        # ── Load candidate data from API cache ──
        stock_data = {}
        for fname in os.listdir(_utils.CACHE_DIR):
            if not fname.endswith('.pkl'):
                continue
            code = fname.replace('.pkl', '')
            try:
                with open(os.path.join(_utils.CACHE_DIR, fname), 'rb') as f:
                    df = pickle.load(f)
                if 'Trading_Volume' in df.columns and 'volume' not in df.columns:
                    df = df.rename(columns={'Trading_Volume': 'volume'})
                elif 'capacity' in df.columns and 'volume' not in df.columns:
                    df = df.rename(columns={'capacity': 'volume'})
                df = engineer_features(df)
                stock_data[code] = df
            except Exception:
                pass

        if not stock_data:
            log.error("  No cached stock data available")
            return None

        day_idx = min(len(df) - 1 for df in stock_data.values())
        log.info(f"  Data: {len(stock_data)} stocks, day_idx={day_idx}")

        # ── 1. Fetch current holdings from API ──
        try:
            raw = Get_User_Stocks(self.account, self.password)
            if isinstance(raw, list):
                api_holdings = {}
                api_cost = {}
                for item in raw:
                    if isinstance(item, dict):
                        code = item.get('stock_code_id', '')
                        shares = int(item.get('shares', 0))
                        cost = float(item.get('beginning_price', 0))
                        if code and shares > 0:
                            api_holdings[code] = shares
                            api_cost[code] = cost
            elif isinstance(raw, dict):
                api_holdings = {k: int(v) for k, v in raw.items() if int(v) > 0}
                api_cost = {}
            else:
                api_holdings = {}
                api_cost = {}
        except Exception as e:
            log.warning(f"  Get_User_Stocks failed: {e}")
            api_holdings = {}
            api_cost = {}

        log.info(f"  API holdings: {len(api_holdings)} positions, State: {len(self.holdings)} holdings")
        # B2: 用 API 成本更新 cost_basis（V2 ROI 排序用）
        for code, cost in api_cost.items():
            if cost > 0:
                self.cost_basis[code] = cost

        # ── 2. Reconcile pending_buys with API ──
        reconciled_buys = []
        expired_buys = []
        for code, order in list(self.pending_buys.items()):
            if code in api_holdings:
                shares = order['shares']
                price = order['price']
                log.info(f"  ✓ BUY {code} EXECUTED: {shares} lot @ {price}")
                reconciled_buys.append(code)
                self.confirmed_trades += 1
            else:
                price = order['price']
                shares = order['shares']
                refund = shares * 1000 * price * (1 + COMMISSION_BUY)
                self.cash += refund
                log.info(f"  ✗ BUY {code} NOT executed, cash refunded: +{refund:.0f}")
                expired_buys.append(code)
        for code in reconciled_buys + expired_buys:
            del self.pending_buys[code]

        # ── 3. Reconcile pending_sells with API ──
        reconciled_sells = []
        for code, order in list(self.pending_sells.items()):
            if code not in api_holdings:
                price = order['price']
                shares = order['shares']
                cash_in = shares * 1000 * price * (1 - COMMISSION_SELL)
                self.cash += cash_in
                log.info(f"  ✓ SELL {code} EXECUTED: {shares} lot @ {price}, cash +{cash_in:.0f}")
                reconciled_sells.append(code)
                self.confirmed_trades += 1
            else:
                # 還在 api_holdings → 沒成交，直接取消（不重送單）
                log.info(f"  ✗ SELL {code} NOT executed, cancelling pending")
                reconciled_sells.append(code)
        for code in reconciled_sells:
            del self.pending_sells[code]

        # ── 4. Trust API as source of truth for current holdings ──
        self.holdings = api_holdings

        log.info(f"  After reconciliation: cash={self.cash:.0f}, holdings={len(self.holdings)}, "
                 f"confirmed_trades={self.confirmed_trades}")

        # ── 6. If pending BUYS remain → skip new orders ──
        # Pending sells are fine — they don't tie up cash.
        if self.pending_buys:
            log.warning("  Still has pending BUYS (not yet matched), skip new trading today")
            self._save_portfolio_state()
            return None

        # ── 7. Screen candidate pool ──
        self.pool = self._screen_pool(stock_data, day_idx, self.holdings)

        # ── 7.5 Fetch real-time prices via twstock ──
        # 也抓 held stocks 的即時價（可能不在 pool 裡但要顯示/計算 held_value）
        all_codes_for_rt = list(set(self.pool + list(self.holdings.keys())))
        realtime_prices = self._fetch_realtime_prices(all_codes_for_rt)

        # ── 8. Prices + limit prices ──
        # 即時價優先（今天開盤），fallback 到 api_cache（昨天收盤）
        prices = {}
        for code in self.pool:
            rt = realtime_prices.get(code)
            df = stock_data.get(code)
            if rt and rt['close'] > 0:
                close = rt['close']
            elif df is not None and day_idx < len(df):
                close = df.iloc[day_idx]['close']
            else:
                close = 0
            prices[code] = close

        # 買入/賣出限價 = 即時現價（不加 GRU 預測偏移）
        limit_prices = {}
        for code in self.pool:
            rt = realtime_prices.get(code)
            if rt and rt['close'] > 0:
                limit_prices[code] = round(rt['close'], 1)
            elif code in prices and prices[code] > 0:
                limit_prices[code] = round(prices[code], 1)
            else:
                limit_prices[code] = 0

        # held_value: 即時價 fallback api_cache
        held_value = 0.0
        for c, shares in self.holdings.items():
            hp = prices.get(c, 0)
            if hp <= 0:
                rt_h = realtime_prices.get(c)
                if rt_h and rt_h['close'] > 0:
                    hp = rt_h['close']
                elif c in stock_data and day_idx < len(stock_data[c]):
                    hp = stock_data[c].iloc[day_idx]['close']
            held_value += shares * 1000 * hp
        port_value = self.cash + held_value
        if port_value <= 0:
            port_value = self.initial_capital
        log.info(f"  Cash: {self.cash:.0f}, Holdings: {held_value:.0f}, Total: {port_value:.0f}")

        # ── 9. PPO inference ──
        obs = self._build_observation(stock_data, self.pool, day_idx, realtime_prices=realtime_prices)
        target_shares, cash_ratio = self._compute_target(obs, port_value, prices)
        log.info(f"  Target cash ratio: {cash_ratio:.2%}")

        # ── 10. Build proposal（帶交易限制）──
        # 限制：單日賣出不超過總資產 20%，買入不超過現金 50%
        MAX_SELL_VALUE = port_value * 0.20
        MAX_BUY_VALUE = self.cash * 0.50
        # 每檔最小交易量：至少 1 張
        MIN_LOTS = 1

        buys = []
        sells = []
        remaining_cash = self.cash
        total_sell_value = 0.0
        total_buy_value = 0.0
        preds_per_code = {}  # 暫存每檔的 GRU preds

        # 先算 GRU preds for display
        for code in self.pool:
            if code in _GRU_MEM_CACHE and code in prices and prices[code] > 0:
                df_c = stock_data.get(code)
                df_len = len(df_c) if df_c is not None else day_idx + 1
                cache = _GRU_MEM_CACHE[code]
                cache_idx = self._get_cache_idx(len(cache), df_len, day_idx) if len(cache) > 0 else 0
                pred_start = GRU_HIDDEN_SIZE * len(TIME_WINDOWS)
                if cache_idx < len(cache):
                    preds_per_code[code] = cache[cache_idx, pred_start:pred_start + N_HORIZONS].tolist()
                else:
                    preds_per_code[code] = [0.0] * N_HORIZONS
            else:
                preds_per_code[code] = [0.0] * N_HORIZONS

        for code in self.pool:
            current = self.holdings.get(code, 0)
            target = target_shares.get(code, 0)
            price = prices.get(code, 0)
            limit_price = limit_prices.get(code, 0)
            if price <= 0 or limit_price <= 0:
                continue
            preds = preds_per_code.get(code, [0.0] * N_HORIZONS)

            if target > current:
                lots = target - current
                cost = lots * 1000 * limit_price
                # 交易限制
                if total_buy_value + cost > MAX_BUY_VALUE:
                    lots = int((MAX_BUY_VALUE - total_buy_value) / (1000 * limit_price))
                    if lots < MIN_LOTS:
                        continue
                    cost = lots * 1000 * limit_price
                if cost > remaining_cash:
                    lots = int(remaining_cash / (1000 * limit_price))
                    if lots < MIN_LOTS:
                        continue
                    cost = lots * 1000 * limit_price
                remaining_cash -= cost
                total_buy_value += cost
                buys.append({'code': code, 'lots': lots, 'price': limit_price, 'close': round(price, 1), 'preds': preds})

            elif current > target:
                lots = current - target
                sell_value = lots * 1000 * limit_price
                # 交易限制
                if total_sell_value + sell_value > MAX_SELL_VALUE:
                    lots = int((MAX_SELL_VALUE - total_sell_value) / (1000 * limit_price))
                    if lots < MIN_LOTS:
                        continue
                    sell_value = lots * 1000 * limit_price
                total_sell_value += sell_value
                cost_basis = self.cost_basis.get(code, price)
                sells.append({'code': code, 'lots': lots, 'price': limit_price, 'close': round(price, 1),
                              'preds': preds, 'cost': round(cost_basis, 1)})

        cash_after = remaining_cash
        log.info(f"  Remaining cash after all buys: {cash_after:.0f}")
        return Proposal(
            date=today.strftime('%Y-%m-%d'),
            cash=cash_after,
            port_value=port_value,
            cash_ratio=cash_ratio,
            buys=buys,
            sells=sells,
        )

    def execute_proposal(self, proposal, simulate=False):
        """Execute orders from a Proposal. Updates state. Returns trade count."""
        today = datetime.now()

        new_buys = {}
        new_sells = {}
        trades_today = 0

        for b in proposal.buys:
            code = b['code']; lots = b['lots']; limit_price = b['price']
            log.info(f"  BUY  {code}: {lots} lot @ limit {limit_price} (close={b['close']}) = {lots*1000*limit_price:.0f}")
            if not simulate:
                try:
                    ok, buy_msg = Buy_Stock(self.account, self.password, code, lots, limit_price)
                    if ok:
                        cost_buy = lots * 1000 * limit_price * (1 + COMMISSION_BUY)
                        self.cash -= cost_buy
                        new_buys[code] = {'shares': lots, 'price': limit_price, 'date': today.strftime('%Y-%m-%d')}
                        # B2: 記 cost_basis 給 V2 ROI 排序用
                        self.cost_basis[code] = limit_price
                        trades_today += 1
                        self.trades.append(('BUY', code, lots, limit_price, today.strftime('%Y-%m-%d')))
                    else:
                        log.warning(f"  BUY {code} API rejected: {buy_msg}")
                except Exception as e:
                    log.error(f"  BUY {code} failed: {e}")
            else:
                cost_buy = lots * 1000 * limit_price * (1 + COMMISSION_BUY)
                self.cash -= cost_buy
                new_buys[code] = {'shares': lots, 'price': limit_price, 'date': today.strftime('%Y-%m-%d')}
                # B2: 記 cost_basis（simulate 也記）
                self.cost_basis[code] = limit_price
                trades_today += 1
                self.trades.append(('BUY', code, lots, limit_price, today.strftime('%Y-%m-%d')+'(SIM)'))

        for s in proposal.sells:
            code = s['code']; lots = s['lots']; limit_price = s['price']
            log.info(f"  SELL {code}: {lots} lot @ limit {limit_price} (close={s['close']}) = {lots*1000*limit_price:.0f}")
            if not simulate:
                try:
                    ok, msg = Sell_Stock(self.account, self.password, code, lots, limit_price)
                    if ok:
                        new_sells[code] = {'shares': lots, 'price': limit_price, 'date': today.strftime('%Y-%m-%d')}
                        trades_today += 1
                        self.trades.append(('SELL', code, lots, limit_price, today.strftime('%Y-%m-%d')))
                    else:
                        log.warning(f"  SELL {code} API rejected: {msg}")
                except Exception as e:
                    log.error(f"  SELL {code} failed: {e}")
            else:
                new_sells[code] = {'shares': lots, 'price': limit_price, 'date': today.strftime('%Y-%m-%d')}
                trades_today += 1
                self.trades.append(('SELL', code, lots, limit_price, today.strftime('%Y-%m-%d')+'(SIM)'))

        self.pending_buys.update(new_buys)
        self.pending_sells.update(new_sells)
        self._save_portfolio_state()

        if self.trades:
            import pandas as pd
            today_str = datetime.now().strftime('%Y-%m-%d')
            df = pd.DataFrame(self.trades, columns=['type', 'code', 'lots', 'price', 'date'])
            df['status'] = 'confirmed'
            df.to_csv(f'logs/trades_{today_str}.csv', index=False)

        return trades_today

    def run(self, simulate=False):
        proposal = self.compute_proposal()
        if proposal is None:
            self._save_portfolio_state()
            return 0
        trades_today = self.execute_proposal(proposal, simulate)

        today_str = datetime.now().strftime('%Y-%m-%d')
        log.info(f"  Today: {trades_today} orders placed ({len(proposal.buys)} buys, {len(proposal.sells)} sells)")
        log.info(f"  Confirmed trades (cumulative): {self.confirmed_trades}")
        log.info(f"  Cash: {self.cash:.0f}")
        log.info(f"{'─'*60}\n")

        log_path = f'logs/trades_{today_str}.csv'
        df = pd.DataFrame(self.trades, columns=['type', 'code', 'lots', 'price', 'date'])
        df['status'] = 'confirmed'
        df.to_csv(log_path, index=False)
        return trades_today


# ─────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────

def is_trading_day():
    """Check if today is a weekday."""
    return datetime.now().weekday() < 5


def is_trading_hours():
    """Check if current time is within 09:30-13:30."""
    now = datetime.now()
    mins = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= mins <= 13 * 60 + 30


def get_state_path():
    return PIPELINE_STATE_FILE


def save_state(**kwargs):
    path = get_state_path()
    state = {}
    if os.path.exists(path):
        with open(path) as f:
            try:
                state = json.load(f)
            except Exception:
                pass
    state.update(kwargs)
    state['updated_at'] = datetime.now().isoformat()
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def load_state():
    path = get_state_path()
    if os.path.exists(path):
        with open(path) as f:
            try:
                return json.load(f)
            except Exception:
                pass
    return {}


def cmd_fetch():
    ok = phase_fetch()
    if ok:
        save_state(last_fetch=datetime.now().isoformat())
    return 0 if ok else 1


def cmd_gru():
    ok = phase_gru()
    if ok:
        save_state(last_gru=datetime.now().isoformat())
    return 0 if ok else 1


def cmd_trade():
    if not is_trading_day():
        log.info("Today is not a trading day (weekend). Skipping trade.")
        return 0
    if not is_trading_hours():
        log.warning("Outside trading hours (09:30-13:30). Submitting anyway as order...")
    try:
        account, password = _utils.get_credentials()
    except RuntimeError as e:
        log.error(e)
        return 1
    trader = LiveTrader(account=account, password=password, ppo_path='checkpoints/ppo_best.pth')
    trades = trader.run(
        simulate=os.environ.get('TRADING_SIMULATE', '0') == '1'
    )
    save_state(last_trade=datetime.now().isoformat(), total_trades=trader.confirmed_trades)
    log.info(f"Trade complete: {trades} today, {trader.confirmed_trades} cumulative confirmed")
    return 0


def reconcile_state(account=None, password=None):
    """Reconcile pending orders with API holdings. Updates state file. Returns summary dict."""
    from data.stock_api import Get_User_Stocks
    if not account or not password:
        try:
            account, password = _utils.get_credentials()
        except RuntimeError as e:
            log.error(e)
            return None

    if not os.path.exists(STATE_FILE):
        return {'cash': INITIAL_CAPITAL, 'num_holdings': 0, 'buy_confirmed': [],
                'buy_refunded': [], 'sell_confirmed': [], 'sell_refunded': [],
                'confirmed_trades': 0}

    with open(STATE_FILE, encoding='utf-8') as f:
        state = json.load(f)

    cash = state.get('cash', INITIAL_CAPITAL)
    pending_buys = state.get('pending_buys', {})
    pending_sells = state.get('pending_sells', {})
    confirmed_trades = state.get('confirmed_trades', 0)
    state_holdings = state.get('holdings', {})

    raw = Get_User_Stocks(account, password)
    api_holdings = {}
    api_cost = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                code = item.get('stock_code_id', '')
                shares = int(item.get('shares', 0))
                cost = float(item.get('beginning_price', 0))
                if code and shares > 0:
                    api_holdings[code] = shares
                    api_cost[code] = cost
    elif isinstance(raw, dict):
        api_holdings = {k: int(v) for k, v in raw.items() if int(v) > 0}
    if len(api_holdings) == 0 and (len(state_holdings) > 0 or len(pending_sells) > 0):
            log.warning(f"  ⚠️ API 回傳 0 持股但狀態有 {len(state_holdings)} 檔持股/{len(pending_sells)} 筆待賣單，可能 API 異常，跳過本次對帳")
            return {'cash': cash, 'num_holdings': len(state_holdings),
                    'buy_confirmed': [], 'buy_refunded': [],
                    'sell_confirmed': [], 'sell_still_pending': list(pending_sells.keys()),
                    'confirmed_trades': confirmed_trades, 'api_failed': True}

    buy_confirmed = []
    buy_refunded = []
    for code, order in list(pending_buys.items()):
        if code in api_holdings:
            buy_confirmed.append(code)
            confirmed_trades += 1
        else:
            shares = order['shares']
            price = order['price']
            refund = shares * 1000 * price * (1 + COMMISSION_BUY)
            cash += refund
            buy_refunded.append(code)

    # B2: 對帳後若 cash 變負（多次退款累計錯誤），發警告並 reset 為 initial
    if cash < 0:
        log.warning(f"  ⚠️ 對帳後現金變負 ({cash:.0f})，重置為初始資本 {INITIAL_CAPITAL:.0f}")
        cash = INITIAL_CAPITAL
    for code in buy_confirmed + buy_refunded:
        del pending_buys[code]

    sell_confirmed = []
    sell_still_pending = []
    for code, order in list(pending_sells.items()):
        if code not in api_holdings:
            shares = order['shares']
            price = order['price']
            cash += shares * 1000 * price * (1 - COMMISSION_SELL)
            sell_confirmed.append(code)
            confirmed_trades += 1
        else:
            # 還在 api_holdings → 沒成交，直接取消
            log.info(f"  ✗ SELL {code} NOT executed, cancelling pending")
            sell_still_pending.append(code)
    for code in sell_confirmed + sell_still_pending:
        del pending_sells[code]

    # NOTE: Do NOT use cost-basis drift correction here.
    # It ignores P&L from completed trades and cumulative commissions.
    # Cash must be tracked through sequential state updates only.

    state['cash'] = cash
    state['holdings'] = {k: v for k, v in api_holdings.items()}
    state['pending_buys'] = {}
    state['pending_sells'] = {k: v for k, v in pending_sells.items()}
    state['confirmed_trades'] = confirmed_trades
    state['cost_basis'] = {k: v for k, v in api_cost.items() if v > 0}
    state['updated_at'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    log.info(f"  Reconciliation: cash={cash:.0f}, holdings={len(api_holdings)}, confirmed={confirmed_trades}")
    log.info(f"    Buy confirmed: {buy_confirmed}, refunded: {len(buy_refunded)}")
    log.info(f"    Sell confirmed: {sell_confirmed}, cancelled: {len(sell_still_pending)}")
    return {
        'cash': cash,
        'num_holdings': len(api_holdings),
        'buy_confirmed': buy_confirmed,
        'buy_refunded': buy_refunded,
        'sell_confirmed': sell_confirmed,
        'sell_still_pending': sell_still_pending,
        'confirmed_trades': confirmed_trades,
    }


def cmd_reconcile():
    log.info("=" * 60)
    log.info("RECONCILIATION")
    log.info("=" * 60)
    try:
        result = reconcile_state()
        if result is None:
            return 1
    except Exception as e:
        log.error(f"Reconciliation failed: {e}", exc_info=True)
        return 1
    log.info("=" * 60)
    return 0


def cmd_daily():
    """Fetch + GRU + reconciliation (post-market update)."""
    rc = cmd_fetch()
    if rc != 0:
        log.warning("Fetch failed, continuing to GRU...")
    rc2 = cmd_gru()
    try:
        reconcile_state()
    except Exception as e:
        log.warning(f"Reconciliation failed: {e}")
    return max(rc, rc2) if rc2 != 0 else rc


def cmd_status():
    state = load_state()
    log.info("=" * 60)
    log.info("PIPELINE STATUS")
    log.info("=" * 60)
    for k, v in state.items():
        log.info(f"  {k}: {v}")
    log.info(f"  Today: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"  Trading day: {is_trading_day()}")
    log.info(f"  Trading hours: {is_trading_hours()}")
    log.info(f"  API cache: {len(os.listdir(_utils.CACHE_DIR))} stocks")
    log.info(f"  GRU cache: {len([f for f in os.listdir(GRU_CACHE_DIR) if f.endswith('.npy')])} stocks")
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Daily PPO Trading Pipeline')
    parser.add_argument('mode', nargs='?', default='status',
                        choices=['fetch', 'gru', 'trade', 'daily', 'reconcile', 'status', 'full'],
                        help='Pipeline phase')
    parser.add_argument('--simulate', action='store_true', help='Simulate trades only')
    parser.add_argument('--reset-cache', action='store_true', help='Delete API cache and re-fetch')
    args = parser.parse_args()

    if args.simulate:
        os.environ['TRADING_SIMULATE'] = '1'
    if args.reset_cache:
        import shutil
        if os.path.exists(_utils.CACHE_DIR):
            shutil.rmtree(_utils.CACHE_DIR)
            os.makedirs(_utils.CACHE_DIR, exist_ok=True)
        log.info("API cache cleared")

    cmds = {
        'fetch': cmd_fetch,
        'gru': cmd_gru,
        'trade': cmd_trade,
        'daily': cmd_daily,
        'reconcile': cmd_reconcile,
        'status': cmd_status,
    }

    if args.mode == 'full':
        log.info("=== FULL PIPELINE ===")
        rcs = [cmd_fetch(), cmd_gru(), cmd_trade()]
        sys.exit(max(rcs) if rcs else 0)
    else:
        sys.exit(cmds[args.mode]())
