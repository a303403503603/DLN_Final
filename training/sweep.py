"""Parameter sweep for PPO training.
Runs multiple parameter combinations and reports results.
Each run uses 30 episodes for quick comparison.
"""
import subprocess
import sys
import re
import os

# Sweep configurations
# Focus on the two problematic parameters:
#   - excess_cash_penalty (currently 1.0, too strong)
#   - holdings_reward_per_extra (currently 0.001, too weak)
# Each run: 30 episodes (~3-5 min)

CONFIGS = [
    # (name, excess_cash_penalty, holdings_reward_per_extra, holdings_reward_cap)
    ("A_weak_holdings",      0.1, 0.001, 0.05),  # baseline (light cash penalty)
    ("B_medium_holdings",    0.1, 0.005, 0.10),  # stronger holdings reward
    ("C_strong_holdings",    0.1, 0.010, 0.20),  # very strong holdings reward
    ("D_balanced",           0.3, 0.005, 0.10),  # mid cash penalty + mid reward
    ("E_no_cash_penalty",    0.0, 0.005, 0.10),  # no cash penalty
    ("F_strong_all",         0.05, 0.010, 0.20), # light cash + strong holdings
]

EPISODES = 30
PYTHON = sys.executable
WORKDIR = r"C:\Users\a3034\Desktop\DL_Final\grullr_stock"


def parse_final_metrics(log_text):
    """Extract final episode metrics from log."""
    metrics = {}
    # Find last "Ep NNN/200" line
    matches = re.findall(
        r"Ep\s+(\d+)/\d+\s+avg_r=([\-\d.]+)\s+total_r=([\-\d.]+)\s+"
        r"steps=\d+\s+val=\$([\d,]+)\s+cash=\$([\d,]+)\s+eq=\$([\d,]+)\s+"
        r"hold=(\d+)\s+trades=(\d+)\s+tpd=([\d.]+)\s+avg_turnover=([\d.]+)",
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
        metrics['trades'] = int(last[7])
        metrics['tpd'] = float(last[8])
        metrics['turnover'] = float(last[9])
    return metrics


def parse_reward_components(log_text):
    """Extract last reward breakdown line."""
    matches = re.findall(
        r"reward: sr=([\-\d.]+)\s+profit=([\-\d.]+)\s+dd=([\-\d.]+)\s+"
        r"turnover=([\-\d.]+)\s+invalid=([\-\d.]+)\s+term=([\-\d.]+)\s+"
        r"xcash=([\-\d.]+)\s+mtrades=([\-\d.]+)\s+hold=([\-\d.]+)",
        log_text
    )
    if matches:
        last = matches[-1]
        return {
            'sr': float(last[0]),
            'profit': float(last[1]),
            'dd': float(last[2]),
            'turnover': float(last[3]),
            'invalid': float(last[4]),
            'term': float(last[5]),
            'xcash': float(last[6]),
            'mtrades': float(last[7]),
            'hold_r': float(last[8]),
        }
    return {}


def parse_validation(log_text):
    """Extract [Val] line."""
    matches = re.findall(
        r"\[Val\]\s+ret=([\-\d.]+)%.*?Sharpe=\s*([\-\d.]+).*?MaxDD=([\-\d.]+)%.*?"
        r"val_final=\$([\d,]+)",
        log_text
    )
    if matches:
        last = matches[-1]
        return {
            'val_ret': float(last[0]),
            'val_sharpe': float(last[1]),
            'val_dd': float(last[2]),
            'val_final': float(last[3].replace(',', '')),
        }
    return {}


def run_config(name, ecp, hrpe, hrc):
    print(f"\n{'='*60}")
    print(f"Running: {name}  ecp={ecp}  hrpe={hrpe}  hrc={hrc}")
    print(f"{'='*60}")

    cmd = [
        PYTHON, "training/train_ppo.py",
        "--excess-cash-penalty", str(ecp),
        "--holdings-reward-per-extra", str(hrpe),
        "--holdings-reward-cap", str(hrc),
        "--max-episodes", str(EPISODES),
        "--run-name", name,
    ]

    result = subprocess.run(
        cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=1800
    )

    log_text = result.stdout + result.stderr
    metrics = parse_final_metrics(log_text)
    rewards = parse_reward_components(log_text)
    val = parse_validation(log_text)

    print(f"  Final ep: {metrics.get('ep', '?')}")
    print(f"  hold={metrics.get('hold','?')}  tpd={metrics.get('tpd','?')}  "
          f"val=${metrics.get('val', 0)/1e6:.1f}M")
    print(f"  rewards: profit={rewards.get('profit',0):.4f}  "
          f"hold_r={rewards.get('hold_r',0):.4f}  "
          f"xcash={rewards.get('xcash',0):.4f}  "
          f"turnover={rewards.get('turnover',0):.4f}")
    if val:
        print(f"  [Val] ret={val.get('val_ret',0):.1f}%  "
              f"Sharpe={val.get('val_sharpe',0):.2f}  "
              f"MaxDD={val.get('val_dd',0):.1f}%")

    return {
        'name': name,
        'ecp': ecp, 'hrpe': hrpe, 'hrc': hrc,
        **metrics, **rewards, **val,
    }


def main():
    results = []
    for cfg in CONFIGS:
        try:
            r = run_config(*cfg)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'name': cfg[0], 'error': str(e)})

    # Summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<25} {'hold':>5} {'tpd':>6} {'val$M':>8} "
          f"{'profit':>8} {'hold_r':>8} {'xcash':>8} {'turn':>8} "
          f"{'vret%':>8} {'vShp':>6}")
    print("-"*100)
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<25} ERROR: {r['error']}")
            continue
        print(f"{r['name']:<25} {r.get('hold',0):>5} {r.get('tpd',0):>6.1f} "
              f"{r.get('val',0)/1e6:>7.1f}M "
              f"{r.get('profit',0):>8.4f} {r.get('hold_r',0):>8.4f} "
              f"{r.get('xcash',0):>8.4f} {r.get('turnover',0):>8.4f} "
              f"{r.get('val_ret',0):>7.1f}% {r.get('val_sharpe',0):>6.2f}")

    # Save to file
    summary_path = os.path.join(WORKDIR, 'logs', 'sweep_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Sweep Results (30 episodes each)\n\n")
        f.write(f"{'Name':<25} {'hold':>5} {'tpd':>6} {'val$M':>8} "
                f"{'profit':>8} {'hold_r':>8} {'xcash':>8} {'turn':>8} "
                f"{'vret%':>8} {'vShp':>6}\n")
        f.write("-"*100 + "\n")
        for r in results:
            if 'error' in r:
                f.write(f"{r['name']:<25} ERROR: {r['error']}\n")
                continue
            f.write(f"{r['name']:<25} {r.get('hold',0):>5} {r.get('tpd',0):>6.1f} "
                    f"{r.get('val',0)/1e6:>7.1f}M "
                    f"{r.get('profit',0):>8.4f} {r.get('hold_r',0):>8.4f} "
                    f"{r.get('xcash',0):>8.4f} {r.get('turnover',0):>8.4f} "
                    f"{r.get('val_ret',0):>7.1f}% {r.get('val_sharpe',0):>6.2f}\n")
    print(f"\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
