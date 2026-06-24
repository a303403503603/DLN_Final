"""多股票資料處理器"""
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import logging

from pipeline.config import TARGET_STOCKS, TRAIN_START, TRAIN_END, VAL_START, VAL_END, TIME_WINDOWS, FEATURE_COLS, CSV_FILE


def get_stock_df(stock_code: str, start=None, end=None) -> pd.DataFrame:
    start = start or TRAIN_START
    end = end or TRAIN_END
    # 股票代碼格式化：50 -> '0050'
    code = str(stock_code).zfill(4)
    
    if not os.path.exists(CSV_FILE):
        logging.error(f"CSV 檔案不存在: {CSV_FILE}")
        return pd.DataFrame()
    
    # 只讀取必要欄位，避免記憶體爆炸
    use_cols = ['date', 'stock_id', 'Trading_Volume', 'Trading_money', 'open', 'max', 'min', 'close', 'spread']
    try:
        df = pd.read_csv(CSV_FILE, usecols=use_cols)
    except ValueError as e:
        # usecols 失敗時改用完整讀取並過濾
        logging.warning(f"usecols 讀取失敗，改用完整讀取: {e}")
        full_df = pd.read_csv(CSV_FILE)
        available_cols = [c for c in use_cols if c in full_df.columns]
        df = full_df[available_cols] if available_cols else pd.DataFrame()
    df.rename(columns={'max': 'high', 'min': 'low'}, inplace=True)
    
    # stock_id 可能是整數，補零對齊
    df['stock_id'] = df['stock_id'].apply(lambda x: str(x).zfill(4))
    
    # 過濾目標股票 + 日期範圍
    mask = (df['stock_id'] == code) & (df['date'] >= start) & (df['date'] <= end)
    df = df[mask].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # 欄位對齊
    df.columns = ['date', 'stock_id', 'Trading_Volume', 'Trading_money', 'open', 'high', 'low', 'close', 'spread']
    
    # 轉數字類型
    for col in ['Trading_Volume', 'Trading_money', 'open', 'high', 'low', 'close', 'spread']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # dropna 移除無效日期
    df.dropna(subset=['high', 'low'], inplace=True)
    
    # 移除 close<=0 的壞資料（例：-10000 偽造記號）
    df = df[df['close'] > 0].copy()
    
    df = df.sort_values('date').reset_index(drop=True)
    # 同一天有多筆重複 row 時只保留第一筆
    df = df.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
    return df


def get_all_stocks_data(stocks: list = None, start: str = TRAIN_START, end: str = TRAIN_END) -> dict:
    """取得多筆股票資料"""
    stocks = stocks or TARGET_STOCKS
    all_data = {}
    for code in stocks:
        print(f'  → {code}')
        df = get_stock_df(code, start, end)
        if not df.empty:
            all_data[code] = df
    return all_data


# ========================
# 2. 特徵工程
# ========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算 ratio 特徵，消除價格 level。
    特徵：close_ret, open_ret, high_close_ratio, low_close_ratio, vol_ret
    接受大小寫欄位名（Date/Open/Close 或 date/open/close）
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('date').reset_index(drop=True)
    
    df['close_ret'] = df['close'] / df['close'].shift(1)
    df['open_ret'] = df['open'] / df['close'].shift(1)
    df['high_close_ratio'] = df['high'] / df['close']
    df['low_close_ratio'] = df['low'] / df['close']
    df['vol_ret'] = df['volume'] / df['volume'].shift(1).replace(0, pd.NA)
    
    # 去頭 NaN（shift 造成）
    df.dropna(subset=['close_ret', 'vol_ret'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


# ========================
# 3. 資料切塊
# ========================
def split_train_val(df: pd.DataFrame, val_ratio=0.15) -> tuple:
    """資料切分為訓練/驗證 set"""
    n = len(df)
    split = int(n * (1 - val_ratio))
    return df.iloc[:split], df.iloc[split:]


# ========================
# 4. GRU 多時序輸入
# ========================
def build_multi_sequence(df: pd.DataFrame, feature_cols: list, window: int) -> np.ndarray:
    """依指定 window 產生 GRU 輸入 (seq_len, n_features)"""
    data = df[feature_cols].values
    seq = data[-window:]
    # 若不夠 window 長度，pad
    if len(seq) < window:
        padded = np.pad(seq, ((window - len(seq), 0), (0, 0)), mode='constant')
        seq = padded
    return seq


def build_multi_time_input(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    回傳 shape (n_windows, max_window, n_features)
    每個 window 皆取最近 window 筆資料（不足 pad）
    """
    result = []
    for w in TIME_WINDOWS:
        seq = build_multi_sequence(df[:len(df)], feature_cols, w)
        result.append(seq)
    return np.array(result)  # (5, seq, n_feat)


# ========================
# 5. 資料預處理工具
# ========================
def normalize_series(series: pd.Series) -> pd.Series:
    """依 column 標準化"""
    return series.replace(np.inf, np.nan).fillna(0).replace(0, 1e-8)


def remove_outliers(df: pd.DataFrame, col: str, n_sigma=3):
    """移除異常值"""
    mean = df[col].mean()
    std = df[col].std()
    mask = (df[col] > mean - n_sigma * std) & (df[col] < mean + n_sigma * std)
    return df[mask]


def prepare_multi_stock_data(stocks=None, start=None, end=None):
    """準備多股票資料（與 train_gru.py 相容的介面）"""
    stocks = stocks or TARGET_STOCKS
    start = start or TRAIN_START
    end = end or TRAIN_END
    stock_data = {}
    for code in stocks:
        print(f'  → {code}')
        df = get_stock_df(code, start, end)
        if not df.empty:
            df = engineer_features(df)
            df.dropna(inplace=True)
            stock_data[code] = df
    return stock_data
