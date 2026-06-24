"""Shared utility functions for pipeline modules."""
import os, json, pickle, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
from pipeline.config import (
    PRICE_DATA_DIR, CSV_FILE, GRU_CACHE_DIR, GRU_HIDDEN_SIZE, TIME_WINDOWS, N_PRED_TARGETS,
    FEATURE_COLS, GRU_PRETRAIN, MODEL_DIR,
)
from data.stock_api import get_taiwan_stock_data
from pipeline.data_processor import engineer_features
from model.multi_time_gru import MultiTimeGRU

FETCH_WORKERS = 50
FETCH_TIMEOUT = 15
FETCH_RETRY = 2
TOP_N = 60
CACHE_DIR = 'data/api_cache'

N_FEAT = len(FEATURE_COLS)
MAX_W = max(TIME_WINDOWS)
CACHE_FEAT_DIM = GRU_HIDDEN_SIZE * len(TIME_WINDOWS) + N_PRED_TARGETS

os.makedirs(CACHE_DIR, exist_ok=True)

log = logging.getLogger(__name__)


def setup_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(f'logs/{name}.log', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _fetch_one(code, start, end):
    last_err = None
    for attempt in range(FETCH_RETRY + 1):
        try:
            df = get_taiwan_stock_data(code, start, end)
            if df is not None and len(df) > 20:
                df.rename(columns={
                    'capacity': 'Trading_Volume',
                    'turnover': 'Trading_money',
                }, inplace=True)
                df['spread'] = (df['high'] - df['low']).fillna(0)
                return code, df
            return code, None
        except Exception as e:
            last_err = e
            if attempt < FETCH_RETRY:
                time.sleep(0.5 * (attempt + 1))
                continue
    log.debug(f"  fetch {code}: {last_err}")
    return code, None


def fetch_parallel(codes, days_back=120, timeout_per_job=FETCH_TIMEOUT):
    """Fetch stock data concurrently via ThreadPoolExecutor."""
    end = datetime.now().strftime('%Y/%m/%d')
    start = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
    result = {}
    ex = ThreadPoolExecutor(max_workers=FETCH_WORKERS)
    try:
        future_to_code = {ex.submit(_fetch_one, c, start, end): c for c in codes}
        for f in as_completed(future_to_code, timeout=timeout_per_job * (FETCH_RETRY + 40)):
            code = future_to_code[f]
            try:
                code2, df = f.result(timeout=timeout_per_job)
                if df is not None:
                    result[code2] = df
            except FuturesTimeoutError:
                log.warning(f"  fetch timeout: {code}")
            except Exception as e:
                log.debug(f"  fetch error {code}: {e}")
    except FuturesTimeoutError:
        log.warning(f"  fetch_parallel overall timeout after {timeout_per_job * (FETCH_RETRY + 2)}s, collected {len(result)}/{len(codes)}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return result


def append_to_price_data(code, new_df):
    """Append rows from new_df (must have date/open/high/low/close/volume) into
    data/price_data/{code}.csv, dedup by date, keep last.

    Returns:
        (path, appended_count) on success; (None, 0) on failure.
    """
    try:
        if new_df is None or len(new_df) == 0:
            return None, 0
        df = new_df.copy()
        if 'date' not in df.columns:
            for cand in ['Date', 'trade_date']:
                if cand in df.columns:
                    df = df.rename(columns={cand: 'date'})
                    break
        if 'date' not in df.columns:
            log.warning(f"  append_to_price_data {code}: no date col, skip")
            return None, 0

        col_map = {}
        for src, dst in [
            ('Open', 'open'), ('High', 'high'), ('Low', 'low'),
            ('Close', 'close'), ('Volume', 'volume'),
            ('Trading_Volume', 'volume'), ('capacity', 'volume'),
        ]:
            if src in df.columns and dst not in df.columns:
                col_map[src] = dst
        if col_map:
            df = df.rename(columns=col_map)

        df['date'] = pd.to_datetime(df['date'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c not in df.columns:
                df[c] = pd.NA

        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna(subset=['date'])

        path = os.path.join(CSV_FILE, f'{code}.csv')
        appended = 0
        if os.path.exists(path):
            try:
                old = pd.read_csv(path, skiprows=[1])
                old['date'] = pd.to_datetime(old['date'])
                old = old[['date', 'open', 'high', 'low', 'close', 'volume']]
            except Exception:
                old = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            before = len(old)
            combined = pd.concat([old, df]).drop_duplicates(subset='date', keep='last')
            combined = combined.sort_values('date').reset_index(drop=True)
            appended = len(combined) - before
        else:
            combined = df.copy()
            appended = len(combined)
            os.makedirs(CSV_FILE, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# Code: {code}\n")
            combined.to_csv(f, index=False)
        return path, appended
    except Exception as e:
        log.warning(f"  append_to_price_data {code}: {e}")
        return None, 0


def cached_fetch(codes, days_back=120):
    need_fetch = []
    result = {}
    for code in codes:
        cache_path = os.path.join(CACHE_DIR, f'{code}.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                result[code] = pickle.load(f)
        else:
            need_fetch.append(code)
    if need_fetch:
        log.info(f"  Fetching {len(need_fetch)} uncached stocks via API...")
        fresh = fetch_parallel(need_fetch, days_back)
        for code, df in fresh.items():
            result[code] = df
            with open(os.path.join(CACHE_DIR, f'{code}.pkl'), 'wb') as f:
                pickle.dump(df, f)
        log.info(f"  Got {len(fresh)} stocks from API")
    return result


def get_top_stocks(n=TOP_N):
    """Return list of available stock codes from price_data directory."""
    return sorted([f.replace('.csv', '') for f in os.listdir(CSV_FILE) if f.endswith('.csv')])[:n]


def load_industry_map():
    p = os.path.join('data', 'industry_map.json')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            raw = json.load(f)
        codes = sorted(set(raw.values()))
        return raw, {c: i for i, c in enumerate(codes)}
    return {}, {}


def load_gru_model(device):
    path = os.path.join(MODEL_DIR, 'gru_shared.pth')
    if not os.path.exists(path):
        log.error(f"  GRU model not found: {path}")
        return None
    mdl = MultiTimeGRU(N_FEAT, GRU_PRETRAIN['hidden_size'],
                        GRU_PRETRAIN['num_layers'], GRU_PRETRAIN['dropout'])
    mdl.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    mdl.eval()
    return mdl.to(device)


def compute_gru_for_stock(features, mdl, device):
    n = len(features)
    n_valid = n - MAX_W + 1
    if n_valid <= 0:
        return None
    feat_t = torch.from_numpy(features).to(device)
    x_list = []
    for W in TIME_WINDOWS:
        unfolded = feat_t.unfold(0, W, 1)
        start = MAX_W - W
        win = unfolded[start:start + n_valid]
        win = win.transpose(1, 2).contiguous()
        x_list.append(win)
    with torch.no_grad():
        pred, latent, _ = mdl(x_list)
    result = np.zeros((n, CACHE_FEAT_DIM), dtype=np.float32)
    result[MAX_W - 1:, :GRU_HIDDEN_SIZE * len(TIME_WINDOWS)] = latent.cpu().numpy()
    result[MAX_W - 1:, GRU_HIDDEN_SIZE * len(TIME_WINDOWS):] = pred.cpu().numpy()
    if n_valid > 0:
        first = result[MAX_W - 1]
        for k in range(MAX_W - 1):
            result[k] = first
    return result


def batch_gru_inference(stock_features_list, mdl, device, batch_size=32):
    """Batch GRU inference for multiple stocks in one model call.

    Args:
        stock_features_list: list of (code, features_ndarray) — features already scaled.
        mdl: MultiTimeGRU model.
        device: torch device.
        batch_size: max stocks per batch (to control GPU memory).

    Returns:
        {code: result_ndarray} — same format as compute_gru_for_stock.
    """
    results = {}
    for batch_start in range(0, len(stock_features_list), batch_size):
        if batch_start > 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        batch_items = stock_features_list[batch_start:batch_start + batch_size]

        x_list_batched = [[] for _ in TIME_WINDOWS]
        codes = []
        n_valids = []
        orig_ns = []

        for code, features in batch_items:
            n = len(features)
            n_valid = n - MAX_W + 1
            if n_valid <= 0:
                continue
            feat_t = torch.from_numpy(features).to(device)
            for i, W in enumerate(TIME_WINDOWS):
                unfolded = feat_t.unfold(0, W, 1)
                start = MAX_W - W
                win = unfolded[start:start + n_valid]
                win = win.transpose(1, 2).contiguous()
                x_list_batched[i].append(win)
            del feat_t
            codes.append(code)
            n_valids.append(n_valid)
            orig_ns.append(n)

        if not codes:
            continue

        x_list_cat = [torch.cat(x, dim=0) for x in x_list_batched]
        del x_list_batched

        with torch.no_grad():
            pred, latent, _ = mdl(x_list_cat)
        del x_list_cat, _

        offset = 0
        for j, code in enumerate(codes):
            nv = n_valids[j]
            n = orig_ns[j]
            stock_pred = pred[offset:offset + nv]
            stock_latent = latent[offset:offset + nv]
            offset += nv
            result = np.zeros((n, CACHE_FEAT_DIM), dtype=np.float32)
            result[MAX_W - 1:, :GRU_HIDDEN_SIZE * len(TIME_WINDOWS)] = stock_latent.cpu().numpy()
            result[MAX_W - 1:, GRU_HIDDEN_SIZE * len(TIME_WINDOWS):] = stock_pred.cpu().numpy()
            first = result[MAX_W - 1]
            for k in range(MAX_W - 1):
                result[k] = first
            results[code] = result
        del pred, latent

    return results


def get_stocks_needing_gru_update():
    needed = set()
    today = datetime.now().date()
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith('.pkl'):
            continue
        code = fname.replace('.pkl', '')
        try:
            with open(os.path.join(CACHE_DIR, fname), 'rb') as f:
                api_df = pickle.load(f)
            api_latest = pd.to_datetime(api_df['date'].max()).date()
        except Exception:
            continue

        gru_path = os.path.join(GRU_CACHE_DIR, f'{code}.npy')
        if not os.path.exists(gru_path):
            needed.add(code)
            continue

        try:
            with open(gru_path, 'rb') as f:
                data = np.load(f, allow_pickle=True).item()
            if isinstance(data, dict) and 'gru_date' in data:
                gru_latest = datetime.fromisoformat(data['gru_date']).date()
            else:
                gru_latest = datetime.min.date()
        except Exception:
            gru_latest = datetime.min.date()

        if api_latest > gru_latest or today > gru_latest:
            needed.add(code)
    return needed


def load_stock_combined_data(code):
    """Read stock data from price_data CSV (primary) + API cache (fallback)."""
    df_csv = pd.DataFrame()

    csv_path = os.path.join(CSV_FILE, f'{code}.csv')
    if os.path.exists(csv_path):
        try:
            df_csv = pd.read_csv(csv_path, skiprows=[0])
            df_csv = df_csv.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            })
            df_csv['date'] = pd.to_datetime(df_csv['date'])
            df_csv = df_csv.sort_values('date').reset_index(drop=True)
            df_csv = df_csv[df_csv['close'] > 0].copy()
            df_csv = df_csv.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
        except Exception as e:
            log.warning(f"  price_data read for {code}: {e}")
            df_csv = pd.DataFrame()

    cache_path = os.path.join(CACHE_DIR, f'{code}.pkl')
    df_api = pd.DataFrame()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                df_api = pickle.load(f)
            df_api['date'] = pd.to_datetime(df_api['date'])
            # Normalize API cache columns to match price_data format
            if 'Trading_Volume' in df_api.columns:
                df_api['volume'] = df_api['Trading_Volume']
            elif 'capacity' in df_api.columns:
                df_api['volume'] = df_api['capacity']
            for col in ['open', 'high', 'low', 'close']:
                if col not in df_api.columns:
                    df_api[col] = pd.NA
        except Exception as e:
            log.warning(f"  API cache read for {code}: {e}")

    if df_csv.empty and df_api.empty:
        return None

    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not df_csv.empty:
        df_csv = df_csv[[c for c in cols if c in df_csv.columns]].copy()
    if not df_api.empty:
        df_api = df_api[[c for c in cols if c in df_api.columns]].copy()

    combined = pd.concat([df_csv, df_api]).drop_duplicates(
        subset='date', keep='last').sort_values('date').reset_index(drop=True)
    combined = combined[combined['close'] > 0].copy()
    if len(combined) < MAX_W + 5:
        return None
    return combined


KAGGLE_DATASET = "ernie55ernie/historical-price-for-taiwan-listed-and-otc-stocks"


def fetch_kaggle_update(tracked_codes=None):
    """Download Kaggle dataset → overwrite price_data/*.csv + api_cache/*.pkl,
    then delete Kaggle cache (zero persistent disk usage).

    Args:
        tracked_codes: iterable of stock codes. If None, uses all files in price_data/.

    Returns:
        (saved_count, total_needed) where saved_count is # of stocks updated.
    """
    import kagglehub
    kaggle_dir = kagglehub.dataset_download(KAGGLE_DATASET)

    if tracked_codes is None:
        tracked_codes = set()
        for fname in os.listdir(CSV_FILE):
            if fname.endswith('.csv'):
                tracked_codes.add(fname.replace('.csv', ''))

    saved = 0
    for code in tracked_codes:
        kaggle_csv = os.path.join(kaggle_dir, f'{code}.csv')
        if not os.path.exists(kaggle_csv):
            continue

        try:
            kaggle_df = pd.read_csv(kaggle_csv)
            kaggle_df.columns = [c.lower() for c in kaggle_df.columns]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                kaggle_df[col] = pd.to_numeric(kaggle_df[col], errors='coerce')
            kaggle_df['date'] = pd.to_datetime(kaggle_df['date'])
            kaggle_max = kaggle_df['date'].max().date()
        except Exception:
            continue

        # Check local api_cache date to skip unchanged stocks
        if saved == 0:
            cache_path = os.path.join(CACHE_DIR, f'{code}.pkl')
            local_max = None
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        local_df = pickle.load(f)
                    local_max = pd.to_datetime(local_df['date'].max()).date()
                except Exception:
                    pass
            if local_max is not None and kaggle_max <= local_max:
                continue

        unified = kaggle_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        unified = unified[unified['close'] > 0].drop_duplicates(subset='date').sort_values('date')

        # Overwrite api_cache pkl
        with open(os.path.join(CACHE_DIR, f'{code}.pkl'), 'wb') as f:
            pickle.dump(unified, f)

        # Overwrite price_data CSV (comment + header + data)
        csv_path = os.path.join(CSV_FILE, f'{code}.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(f"# Code: {code}\n\n")
            unified.to_csv(f, index=False)
        saved += 1

    # Delete Kaggle cache to avoid disk bloat
    import shutil
    shutil.rmtree(kaggle_dir, ignore_errors=True)
    log.info(f"  Kaggle cache deleted ({kaggle_dir})")

    return saved, len(tracked_codes)


def get_credentials():
    account = os.environ.get('TRADING_ACCOUNT')
    password = os.environ.get('TRADING_PASSWORD')
    if not account or not password:
        raise RuntimeError("TRADING_ACCOUNT / TRADING_PASSWORD not set in .env")
    return account, password
