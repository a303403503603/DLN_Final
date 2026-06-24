"""V2 N sweep: find best HOLD_POOL_SIZE.
Sweeps N = 5/10/15/20/30, each 50 episodes, picks best val_final.
"""
import subprocess
import sys
import re
import os

CONFIGS = [
    ("N05", 5),
    ("N10", 10),
    ("N15", 15),
    ("N20", 20),
    ("N30", 30),
]

EPISODES = 50
PYTHON = sys.executable
WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_metrics(log_text):
    """Extract last episode metrics from V2 log format."""
    metrics = {}
    # V2 format: ... hold=N hpool=X wpool=Y wentry=Z trades=...
    matches = re.findall(
        r"Ep\s+(\d+)/\d+\s+avg_r=([\-\d.]+)\s+total_r=([\-\d.]+)\s+"
        r"steps=\d+\s+val=\$([\d,]+)\s+cash=\$([\d,]+)\s+eq=\$([\d,]+)\s+"
        r"hold=(\d+)\s+hpool=(\d+)\s+wpool=(\d+)\s+wentry=(\d+)\s+"
        r"trades=(\d+)\s+tpd=([\d.]+)\s+avg_turnover=([\d.]+)",
        log_text
    )
    if matches:
        last = matches[-1]
        metrics['ep'] = int(last[0])
        metrics['avg_r'] = float(last[1])
        metrics['total_r'] = float(last[2])
        metrics['val'] = float(last[3].replace(',', ''))
        metrics['cash'] = float(last[4].replace(',', ''))
        metrics['eq'] = float(last[5].replace(',', ''))
        metrics['hold'] = int(last[6])
        metrics['hpool'] = int(last[7])
        metrics['wpool'] = int(last[8])
        metrics['wentry'] = int(last[9])
        metrics['trades'] = int(last[10])
        metrics['tpd'] = float(last[11])
        metrics['turnover'] = float(last[12])
    return metrics


def parse_validation(log_text):
    """Extract [Val] line."""
    matches = re.findall(
        r"\[Val\]\s+ret=([\-\d.]+)%.*?Sharpe=\s*([\-\d.]+).*?MaxDD=([\-\d.]+)%.*?"
        r"val_final=\$([\d,]+).*?held=\{([^}]*)\}",
        log_text
    )
    if matches:
        last = matches[-1]
        held_str = last[4].strip()
        return {
            'val_ret': float(last[0]),
            'val_sharpe': float(last[1]),
            'val_dd': float(last[2]),
            'val_final': float(last[3].replace(',', '')),
            'val_held': held_str,
        }
    return {}


def run_config(name, n_hold):
    print(f"\n{'='*60}")
    print(f"Running: {name}  hold_pool_size={n_hold}")
    print(f"{'='*60}")

    cmd = [
        PYTHON, "training/train_ppo.py",
        "--hold-pool-size", str(n_hold),
        "--max-episodes", str(EPISODES),
        "--run-name", name,
    ]

    try:
        result = subprocess.run(
            cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=3600
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 1h")
        return {'name': name, 'n_hold': n_hold, 'error': 'timeout'}

    log_text = result.stdout + result.stderr
    metrics = parse_metrics(log_text)
    val = parse_validation(log_text)

    print(f"  Final ep: {metrics.get('ep', '?')}  hold={metrics.get('hold','?')}  "
          f"hpool={metrics.get('hpool','?')}  wpool={metrics.get('wpool','?')}")
    print(f"  val=${metrics.get('val', 0)/1e6:.1f}M  tpd={metrics.get('tpd',0):.2f}")
    if val:
        print(f"  [Val] ret={val.get('val_ret',0):.1f}%  "
              f"Sharpe={val.get('val_sharpe',0):.2f}  "
              f"MaxDD={val.get('val_dd',0):.1f}%  "
              f"val_final=${val.get('val_final',0)/1e6:.1f}M")
        print(f"  val held: {val.get('val_held','')[:100]}")

    return {
        'name': name,
        'n_hold': n_hold,
        **metrics, **val,
    }


def main():
    results = []
    for name, n in CONFIGS:
        r = run_config(name, n)
        results.append(r)
        # Save partial after each run
        save_summary(results)

    print(f"\n{'='*80}")
    print("N SWEEP SUMMARY (V2)")
    print(f"{'='*80}")
    print(f"{'Name':<6} {'N':>3} {'hold':>5} {'hpool':>6} {'wpool':>6} "
          f"{'wentry':>7} {'tpd':>5} {'val$M':>8} "
          f"{'vret%':>7} {'vShp':>6} {'vDD%':>6}")
    print("-"*80)
    best = None
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<6} ERROR: {r['error']}")
            continue
        vret = r.get('val_ret', 0)
        vshp = r.get('val_sharpe', 0)
        vdd = r.get('val_dd', 0)
        print(f"{r['name']:<6} {r.get('n_hold',0):>3} {r.get('hold',0):>5} "
              f"{r.get('hpool',0):>6} {r.get('wpool',0):>6} "
              f"{r.get('wentry',0):>7} {r.get('tpd',0):>5.1f} "
              f"{r.get('val',0)/1e6:>7.1f}M "
              f"{vret:>6.1f}% {vshp:>6.2f} {vdd:>5.1f}%")
        # Pick best by val_sharpe
        if best is None or vshp > best.get('val_sharpe', -999):
            best = r

    if best:
        print(f"\nBest by Sharpe: {best['name']} (N={best['n_hold']})  "
              f"Sharpe={best.get('val_sharpe',0):.2f}  "
              f"val=${best.get('val_final',0)/1e6:.1f}M")


def save_summary(results):
    summary_path = os.path.join(WORKDIR, 'logs', 'sweep_n_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("V2 N Sweep Results (50 episodes each)\n\n")
        f.write(f"{'Name':<6} {'N':>3} {'hold':>5} {'hpool':>6} {'wpool':>6} "
                f"{'wentry':>7} {'tpd':>5} {'val$M':>8} "
                f"{'vret%':>7} {'vShp':>6} {'vDD%':>6}\n")
        f.write("-"*80 + "\n")
        for r in results:
            if 'error' in r:
                f.write(f"{r['name']:<6} ERROR: {r['error']}\n")
                continue
            f.write(f"{r['name']:<6} {r.get('n_hold',0):>3} {r.get('hold',0):>5} "
                    f"{r.get('hpool',0):>6} {r.get('wpool',0):>6} "
                    f"{r.get('wentry',0):>7} {r.get('tpd',0):>5.1f} "
                    f"{r.get('val',0)/1e6:>7.1f}M "
                    f"{r.get('val_ret',0):>6.1f}% "
                    f"{r.get('val_sharpe',0):>6.2f} "
                    f"{r.get('val_dd',0):>5.1f}%\n")


if __name__ == '__main__':
    main()
