"""GRU 預訓練 - 共用 GRU + Multi-Horizon 預測 (優化版: pinned tensors + AMP)"""
import torch
import torch.nn as nn
import numpy as np
import os, sys, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ['OMP_NUM_THREADS'] = '1'

from pipeline.config import (
    GRU_PRETRAIN, MODEL_DIR, LOG_DIR, FEATURE_COLS, TIME_WINDOWS,
    PRED_HORIZONS, N_HORIZONS, N_PRED_TARGETS, HORIZON_WEIGHTS,
    PRICE_DATA_DIR
)
from model.multi_time_gru import MultiTimeGRU
from pipeline.data_processor import engineer_features
from sklearn.preprocessing import StandardScaler

N_FEAT = len(FEATURE_COLS)
MAX_W = max(TIME_WINDOWS)
MAX_H = max(PRED_HORIZONS)


def create_model():
    return MultiTimeGRU(N_FEAT, GRU_PRETRAIN['hidden_size'],
                        GRU_PRETRAIN['num_layers'], GRU_PRETRAIN['dropout'])


def _process_one_stock(args):
    code, df = args
    import pandas as pd

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['close'] > 0].copy()
    df = df.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
    if len(df) < MAX_W + MAX_H + 5:
        return None

    df = engineer_features(df)
    df.dropna(subset=FEATURE_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) < MAX_W + MAX_H + 5:
        return None

    n = len(df)
    features = df[FEATURE_COLS].values.astype(np.float32)
    close = df['close'].values.astype(np.float32)
    open_p = df['open'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)

    n_valid = n - MAX_W - MAX_H + 1
    y = np.zeros((n_valid, N_PRED_TARGETS), dtype=np.float32)
    t_start = MAX_W - 1
    for j, h in enumerate(PRED_HORIZONS):
        ref = close[t_start + np.arange(n_valid)]
        y[:, j * 4 + 0] = close[t_start + np.arange(n_valid) + h] / ref - 1
        y[:, j * 4 + 1] = open_p[t_start + np.arange(n_valid) + h] / ref - 1
        y[:, j * 4 + 2] = high[t_start + np.arange(n_valid) + h] / ref - 1
        y[:, j * 4 + 3] = low[t_start + np.arange(n_valid) + h] / ref - 1

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)

    return {'code': code, 'features': features, 'y': y,
            'n_valid': n_valid, 't_start': t_start}


def load_all_stocks():
    import pandas as pd

    all_files = sorted([f for f in os.listdir(PRICE_DATA_DIR) if f.endswith('.csv')])
    print(f"Found {len(all_files)} stock CSV files in {PRICE_DATA_DIR}")

    stocks_raw = []
    for i, fname in enumerate(all_files):
        if (i + 1) % 300 == 0:
            print(f"  Loaded {i + 1}/{len(all_files)}...")
        code = fname.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(PRICE_DATA_DIR, fname), skiprows=[0])
            if len(df) < 100:
                continue
            stocks_raw.append((code, df))
        except Exception as e:
            print(f"  Skip {code}: {e}")

    print(f"Loaded {len(stocks_raw)} DataFrames, processing...")

    n_workers = min(12, os.cpu_count() or 4)
    print(f"Processing with {n_workers} threads...")
    stocks = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_one_stock, s) for s in stocks_raw]
        done = 0
        for f in as_completed(futures):
            result = f.result()
            if result is not None:
                stocks.append(result)
            done += 1
            if done % 300 == 0:
                print(f"  {done}/{len(stocks_raw)} processed, "
                      f"{len(stocks)} loaded")

    print(f"Loaded {len(stocks)} stocks, "
          f"skipped {len(stocks_raw) - len(stocks)}")
    return stocks


def weighted_huber_loss(pred, target, delta=0.5):
    diff = (pred - target).abs()
    huber = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    loss_per_dim = huber.mean(dim=0)
    # Each horizon weight repeated 4 times for O/H/L/C
    horizon_w = torch.FloatTensor(HORIZON_WEIGHTS).repeat_interleave(4).to(pred.device)
    return (loss_per_dim * horizon_w).sum()


def prepare_stock_tensors(s, device):
    s['feat_t'] = torch.from_numpy(s['features']).to(device)
    s['y_t'] = torch.from_numpy(s['y']).to(device)
    return s


def train_shared(stocks, mdl, epochs, lr, bs, patience, device):
    total_samples = sum(s['n_valid'] for s in stocks)
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Model params: {sum(p.numel() for p in mdl.parameters()):,}")

    # Pre-convert all stock data to pinned cpu tensors
    print("Preparing tensor data...")
    for s in stocks:
        prepare_stock_tensors(s, device)
    print("  Done")

    n_val = max(1, int(len(stocks) * GRU_PRETRAIN['val_ratio']))
    np.random.seed(42)
    val_idx = np.random.choice(len(stocks), n_val, replace=False)
    train_stocks = [s for i, s in enumerate(stocks) if i not in set(val_idx)]
    val_stocks = [stocks[i] for i in val_idx]
    print(f"Train stocks: {len(train_stocks)}, Val stocks: {len(val_stocks)}")

    mdl = mdl.to(device)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    scaler = torch.amp.GradScaler()
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    # CSV logging
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'gru_training_log.csv')
    f_csv = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss'])
    epoch_log = {'train': [], 'val': [], 'best_val': []}

    for ep in range(epochs):
        print(f"\n--- Ep {ep}/{epochs} ---")
        mdl.train()
        np.random.shuffle(train_stocks)
        epoch_loss = 0.0
        total_seen = 0

        for si, s in enumerate(train_stocks):
            nv = s['n_valid']
            feat_t = s['feat_t']
            # unfold + transpose: views only, no memory copy
            X_view = feat_t.unfold(0, MAX_W, 1).transpose(1, 2)

            indices = torch.randperm(nv, device='cuda')
            for batch_start in range(0, nv, bs):
                bidx = indices[batch_start:batch_start + bs]

                bx = []
                for W in TIME_WINDOWS:
                    win = X_view[bidx, -W:, :]
                    bx.append(win)

                by = s['y_t'][bidx]

                opt.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    pred, _, _ = mdl(bx)
                    loss = weighted_huber_loss(pred, by)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                epoch_loss += loss.item() * len(bidx)
                total_seen += len(bidx)

            # Progress every 5%
            if (si + 1) % max(1, len(train_stocks) // 20) == 0:
                pct = (si + 1) / len(train_stocks) * 100
                print(f"  [{ep:3d}] {pct:.0f}% train "
                      f"({si+1}/{len(train_stocks)} stocks)")

        avg_train_loss = epoch_loss / max(total_seen, 1)

        # Validation
        mdl.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for s in val_stocks:
                nv = s['n_valid']
                X_view = s['feat_t'].unfold(0, MAX_W, 1).transpose(1, 2)

                for batch_start in range(0, nv, bs):
                    bidx = slice(batch_start, min(batch_start + bs, nv))
                    bx = []
                    for W in TIME_WINDOWS:
                        win = X_view[bidx, -W:, :]
                        bx.append(win)
                    by = s['y_t'][bidx]
                    pred, _, _ = mdl(bx)
                    batch_n = bidx.stop - bidx.start
                    val_loss += weighted_huber_loss(pred, by).item() * batch_n
                    val_seen += batch_n

        avg_val_loss = val_loss / max(val_seen, 1)

        # CSV log every epoch
        csv_writer.writerow([ep, f"{avg_train_loss:.8f}", f"{avg_val_loss:.8f}", f"{best_val_loss:.8f}"])
        f_csv.flush()
        epoch_log['train'].append(avg_train_loss)
        epoch_log['val'].append(avg_val_loss)
        epoch_log['best_val'].append(best_val_loss)

        if ep % 5 == 0 or ep == epochs - 1 or ep == 0:
            print(f"  Ep {ep:3d}  tr={avg_train_loss:.6f}  "
                  f"va={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in mdl.state_dict().items()}
            patience_counter = 0
        elif avg_val_loss > best_val_loss + 1e-6:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop @{ep}")
                break

    f_csv.close()

    # Save training history
    np.savez(os.path.join(log_dir, 'gru_training_history.npz'),
             train_loss=np.array(epoch_log['train']),
             val_loss=np.array(epoch_log['val']),
             best_val_loss=np.array(epoch_log['best_val']))

    mdl.load_state_dict(best_state)
    return mdl.cpu()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading stocks...")
    all_stocks = load_all_stocks()
    if len(all_stocks) == 0:
        print("No stocks loaded, abort.")
        sys.exit(1)

    print("Creating model...")
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\nTraining...")
    model = train_shared(
        all_stocks, model,
        epochs=GRU_PRETRAIN['epochs'],
        lr=GRU_PRETRAIN['lr'],
        bs=GRU_PRETRAIN['batch_size'],
        patience=GRU_PRETRAIN['patience'],
        device=device,
    )

    path = os.path.join(MODEL_DIR, 'gru_shared.pth')
    torch.save(model.state_dict(), path)
    print(f"Saved to {path}")
