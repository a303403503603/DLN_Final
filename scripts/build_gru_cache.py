"""從 price_data 產生全部 GRU cache（給 PPO 訓練用）"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from pipeline import _utils as U
from pipeline.data_processor import engineer_features
from pipeline.config import PRICE_DATA_DIR, GRU_CACHE_DIR, FEATURE_COLS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

mdl = U.load_gru_model(device)
if mdl is None:
    print('Failed to load GRU model')
    sys.exit(1)

today_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d')
files = sorted([f for f in os.listdir(PRICE_DATA_DIR) if f.endswith('.csv')])
print(f'Generating GRU cache for {len(files)} stocks...')

done, skipped = 0, 0
for i, fname in enumerate(files):
    code = fname.replace('.csv', '')
    try:
        df = U.load_stock_combined_data(code)
        if df is None or len(df) < U.MAX_W + 5:
            skipped += 1
            continue

        df = engineer_features(df)
        df.dropna(subset=FEATURE_COLS, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) < U.MAX_W + 5:
            skipped += 1
            continue

        features = df[FEATURE_COLS].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler()
        features = scaler.fit_transform(features).astype(np.float32)

        cache = U.compute_gru_for_stock(features, mdl, device)
        if cache is not None:
            out_path = os.path.join(GRU_CACHE_DIR, f'{code}.npy')
            np.save(out_path, {'cache': cache, 'gru_date': today_str})
            done += 1
        else:
            skipped += 1
    except Exception as e:
        print(f'  Error {code}: {e}')
        skipped += 1

    if (i + 1) % 200 == 0:
        print(f'  [{i+1}/{len(files)}] done={done} skipped={skipped}')

mdl.cpu()
print(f'Done: {done} cached, {skipped} skipped')
