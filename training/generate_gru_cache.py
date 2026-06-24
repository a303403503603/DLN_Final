"""Regenerate GRU cache for all stocks in price_data dir using the new model."""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    FEATURE_COLS, TIME_WINDOWS, PRICE_DATA_DIR, GRU_CACHE_DIR,
    GRU_HIDDEN_SIZE, N_PRED_TARGETS,
)
from pipeline._utils import (
    load_gru_model, compute_gru_for_stock, setup_logger, CACHE_FEAT_DIM,
)
from pipeline.data_processor import engineer_features

N_FEAT = len(FEATURE_COLS)
MAX_W = max(TIME_WINDOWS)
os.makedirs(GRU_CACHE_DIR, exist_ok=True)
log = setup_logger('generate_gru_cache')


def load_one_csv(code):
    try:
        path = os.path.join(PRICE_DATA_DIR, f'{code}.csv')
        if not os.path.exists(path):
            return code, None, 'file_not_found'
        df = pd.read_csv(path, skiprows=[1])
        if len(df) < MAX_W + 5:
            return code, None, 'too_short'
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume',
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df[df['close'] > 0].copy()
        df = df.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
        if len(df) < MAX_W + 5:
            return code, None, 'too_short_after_clean'
        return code, df, None
    except Exception as e:
        return code, None, str(e)


def process_stock(code, df, mdl, device, today_str):
    try:
        df = engineer_features(df)
        df.dropna(subset=FEATURE_COLS, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) < MAX_W + 2:
            return 'skip', None

        features = df[FEATURE_COLS].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        features = scaler.fit_transform(features).astype(np.float32)

        cache = compute_gru_for_stock(features, mdl, device)
        if cache is None or cache.shape[1] != CACHE_FEAT_DIM:
            return 'skip', None

        out = {'cache': cache, 'gru_date': today_str}
        np.save(os.path.join(GRU_CACHE_DIR, f'{code}.npy'), out)
        return 'done', None
    except Exception as e:
        return 'error', str(e)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    log.info(f"Device: {device}")

    mdl = load_gru_model(device)
    if mdl is None:
        print("FAILED to load GRU model from checkpoints/gru_shared.pth")
        sys.exit(1)
    n_params = sum(p.numel() for p in mdl.parameters())
    print(f"Model loaded ({n_params:,} params)")
    log.info(f"Model loaded ({n_params:,} params)")

    codes = sorted([
        f.replace('.csv', '') for f in os.listdir(PRICE_DATA_DIR)
        if f.endswith('.csv')
    ])
    total = len(codes)
    print(f"Found {total} stock CSVs in {PRICE_DATA_DIR}")
    log.info(f"Found {total} stock CSVs")

    today_str = datetime.now().strftime('%Y-%m-%d')
    done = skipped = errors = 0
    n_workers = 8

    start_time = datetime.now()
    print(f"Processing with {n_workers} I/O threads...\n")
    log.info(f"Processing with {n_workers} I/O threads")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        in_flight = {}
        for code in codes[:n_workers]:
            in_flight[executor.submit(load_one_csv, code)] = code
        idx = n_workers

        while in_flight:
            f = next(as_completed(in_flight))
            code = in_flight.pop(f)
            code_back, df, err = f.result()

            if err:
                if err.startswith('too_short'):
                    skipped += 1
                else:
                    errors += 1
                    log.warning(f"  {code}: {err}")
            else:
                status, msg = process_stock(code, df, mdl, device, today_str)
                if status == 'done':
                    done += 1
                elif status == 'skip':
                    skipped += 1
                else:
                    errors += 1
                    log.warning(f"  {code}: {msg}")

            if idx < total:
                in_flight[executor.submit(load_one_csv, codes[idx])] = codes[idx]
                idx += 1

            processed = done + skipped + errors
            if processed % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"  [{processed:4d}/{total}] {done} done, {skipped} skip, {errors} err "
                      f"({rate:.1f}/s, ETA {eta:.0f}s)")
                log.info(f"  Progress: {processed}/{total} ({done} done, {skipped} skip, {errors} err)")

    elapsed = (datetime.now() - start_time).total_seconds()
    npy_count = len([f for f in os.listdir(GRU_CACHE_DIR) if f.endswith('.npy')])
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.0f}s")
    print(f"  {done} cache files written")
    print(f"  {skipped} skipped (too short)")
    print(f"  {errors} errors")
    print(f"  GRU cache: {npy_count} .npy files in {GRU_CACHE_DIR}")
    print(f"{'='*50}")
    log.info(f"Complete: {done} written, {skipped} skipped, {errors} errors, "
             f"{npy_count} total .npy")


if __name__ == '__main__':
    main()
