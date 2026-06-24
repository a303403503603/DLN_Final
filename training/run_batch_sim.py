"""多時段回測 + 詳細圖表報告"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))  # for sibling imports

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

from pipeline.config import (
    MODEL_DIR, INITIAL_CAPITAL, OBS_TOTAL, OUTPUT_DIM,
    CANDIDATE_POOL_SIZE, GRU_CACHE_DIR,
)
from rl_env.trading_env import MultiStockTradingEnv
from model.ppo_agent import PPOAgent
from training.train_ppo import _load_stock_data

import logging
logging.basicConfig(level=logging.WARNING)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 120,
})


PERIODS = [
    # 2023
    ('2023-01-03', '2023-01-31'),
    ('2023-03-01', '2023-03-31'),
    ('2023-06-01', '2023-06-30'),
    ('2023-09-01', '2023-09-29'),
    # 2024
    ('2024-01-02', '2024-01-31'),
    ('2024-04-01', '2024-04-30'),
    ('2024-07-01', '2024-07-31'),
    ('2024-10-01', '2024-10-31'),
    # 2025
    ('2025-01-02', '2025-01-31'),
    ('2025-04-01', '2025-04-30'),
    ('2025-07-01', '2025-07-31'),
    ('2025-10-01', '2025-10-31'),
    # 2026
    ('2026-01-05', '2026-01-30'),
    ('2026-04-01', '2026-04-30'),
]


def sim_one(agent, start, end, max_stocks=50, seed=None):
    stock_data = _load_stock_data(
        min_days=20, max_stocks=max_stocks,
        date_range=(start, end),
    )
    if not stock_data:
        return None
    codes = list(stock_data.keys())

    env = MultiStockTradingEnv(
        all_stocks=codes, data=stock_data,
        gru_cache_dir=GRU_CACHE_DIR,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        initial_capital=INITIAL_CAPITAL,
    )

    obs, _ = env.reset(seed=seed)
    done = False
    portfolio = []
    cash = []
    holdings_cnt = []
    steps = 0

    while not done:
        action, _, _ = agent.get_action(obs, stochastic=False)
        obs, reward, terminated, truncated, info = env.step(action.flatten())
        portfolio.append(info['portfolio_value'])
        cash.append(info['cash'])
        holdings_cnt.append(len(info['holdings']))
        steps += 1
        done = terminated or truncated

    # Metrics
    init = INITIAL_CAPITAL
    final = portfolio[-1]
    ret_pct = (final / init - 1) * 100
    daily_ret = np.diff(portfolio) / np.maximum(portfolio[:-1], 1)
    sharpe = float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252))
    peak = np.maximum.accumulate(portfolio)
    dd = (peak - portfolio) / peak
    max_dd = float(np.max(dd))

    trade_count = len(env.trades)
    buy_vol = sum(t[2]*t[3]*1000 for t in env.trades if 'BUY' in t[0])
    sell_vol = sum(t[2]*t[3]*1000 for t in env.trades if 'SELL' in t[0])

    return {
        'start': start, 'end': end, 'steps': steps,
        'return_pct': ret_pct, 'sharpe': sharpe, 'max_dd': max_dd,
        'trade_count': trade_count, 'buy_volume': buy_vol, 'sell_volume': sell_vol,
        'portfolio': portfolio, 'cash': cash,
        'holdings_cnt': holdings_cnt, 'trades': env.trades,
        'holdings_final': {k: int(v) for k, v in env.holdings.items() if v > 0},
    }


def plot_period(r, ax_eq, ax_dd, ax_hold):
    p = r['portfolio']
    init = INITIAL_CAPITAL
    ax_eq.plot(p, linewidth=1.2)
    ax_eq.axhline(init, color='gray', ls='--', alpha=0.4)
    ret_str = f"{r['return_pct']:+.2f}%"
    dd_str = f"{r['max_dd']:.2%}"
    sr_str = f"{r['sharpe']:.2f}"
    tr_str = f"{r['trade_count']}"
    label = f"{r['start']}~{r['end'][5:]}  {ret_str}  SR={sr_str}  DD={dd_str}  T={tr_str}"
    ax_eq.text(len(p)-1, p[-1], f" {ret_str}", fontsize=7, va='center')
    return label


def make_report(results):
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    n = len(results)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.30)

    # ── Page 1: Per-period equity curves ──
    for idx, r in enumerate(results):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        p = r['portfolio']
        init = INITIAL_CAPITAL
        ax.plot(p, linewidth=1.2)
        ax.axhline(init, color='gray', ls='--', alpha=0.4)
        ret = r['return_pct']
        ax.set_title(f"{r['start']}~{r['end'][5:]}  {ret:+.2f}%", fontsize=9)
        ax.set_ylabel('Portfolio' if col == 0 else '')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0e'))
        ax.grid(True, alpha=0.25)
        # Annotate final value
        ax.annotate(f"${p[-1]:,.0f}", xy=(len(p)-1, p[-1]),
                    fontsize=7, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))

    fig.suptitle('PPO Simulation — Monthly Periods (Equity Curves)', fontsize=14, y=1.01)
    p1 = f'results/batch_equity_{ts}.png'
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {p1}")

    # ── Page 2: Performance summary table + stats ──
    fig2 = plt.figure(figsize=(18, 10))
    gs2 = GridSpec(2, 3, figure=fig2, hspace=0.30, wspace=0.30)

    # Summary table
    ax_tab = fig2.add_subplot(gs2[0, :])
    ax_tab.axis('off')
    cols = ['Period', 'Days', 'Return%', 'Sharpe', 'MaxDD%', 'Trades', 'AvgTrade$']
    rows_data = []
    for r in results:
        avg_trd = (r['buy_volume']+r['sell_volume'])/max(r['trade_count'],1)
        rows_data.append([
            f"{r['start']}~{r['end'][5:]}",
            str(r['steps']),
            f"{r['return_pct']:+.2f}",
            f"{r['sharpe']:.2f}",
            f"{r['max_dd']:.2%}",
            str(r['trade_count']),
            f"${avg_trd:,.0f}",
        ])
    # Averages row
    avg_ret = np.mean([r['return_pct'] for r in results])
    avg_sr = np.mean([r['sharpe'] for r in results])
    avg_dd = np.mean([r['max_dd'] for r in results])
    avg_tr = np.mean([r['trade_count'] for r in results])
    avg_at = np.mean([(r['buy_volume']+r['sell_volume'])/max(r['trade_count'],1) for r in results])
    rows_data.append([
        'AVERAGE', f"{np.mean([r['steps'] for r in results]):.0f}",
        f"{avg_ret:+.2f}", f"{avg_sr:.2f}", f"{avg_dd:.2%}",
        f"{avg_tr:.0f}", f"${avg_at:,.0f}",
    ])
    table = ax_tab.table(cellText=rows_data, colLabels=cols,
                          cellLoc='center', loc='center',
                          colWidths=[0.16, 0.08, 0.10, 0.10, 0.10, 0.10, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for key, cell in table.get_celld().items():
        if key[0] == len(rows_data)-1:
            cell.set_facecolor('#e6f3ff')
        elif key[0] == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
    ax_tab.set_title('Performance Summary by Period', fontsize=13, pad=20)

    # Return distribution
    ax_ret = fig2.add_subplot(gs2[1, 0])
    returns = [r['return_pct'] for r in results]
    colors = ['green' if v >= 0 else 'red' for v in returns]
    ax_ret.bar(range(len(returns)), returns, color=colors, alpha=0.7)
    ax_ret.axhline(0, color='gray', ls='--')
    ax_ret.set_xticks(range(len(results)))
    ax_ret.set_xticklabels([f"{i+1}" for i in range(len(results))], fontsize=6)
    ax_ret.set_ylabel('Return %')
    ax_ret.set_title('Per-Period Returns')
    ax_ret.grid(True, alpha=0.25)

    # Sharpe distribution
    ax_sr = fig2.add_subplot(gs2[1, 1])
    srs = [r['sharpe'] for r in results]
    colors_sr = ['green' if v >= 1 else ('orange' if v >= 0 else 'red') for v in srs]
    ax_sr.bar(range(len(srs)), srs, color=colors_sr, alpha=0.7)
    ax_sr.axhline(0, color='gray', ls='--')
    ax_sr.axhline(1, color='green', ls=':', alpha=0.5)
    ax_sr.set_xticks(range(len(results)))
    ax_sr.set_xticklabels([f"{i+1}" for i in range(len(results))], fontsize=6)
    ax_sr.set_ylabel('Sharpe')
    ax_sr.set_title('Sharpe Ratio')
    ax_sr.grid(True, alpha=0.25)

    # Trade count
    ax_tr = fig2.add_subplot(gs2[1, 2])
    trades = [r['trade_count'] for r in results]
    ax_tr.bar(range(len(trades)), trades, color='steelblue', alpha=0.7)
    ax_tr.axhline(np.mean(trades), color='red', ls='--', alpha=0.6, label=f'Avg={np.mean(trades):.0f}')
    ax_tr.set_xticks(range(len(results)))
    ax_tr.set_xticklabels([f"{i+1}" for i in range(len(results))], fontsize=6)
    ax_tr.set_ylabel('Trades')
    ax_tr.set_title('Trade Count')
    ax_tr.legend(fontsize=7)
    ax_tr.grid(True, alpha=0.25)

    fig2.suptitle('PPO Simulation — Cross-Period Summary', fontsize=14, y=1.01)
    p2 = f'results/batch_summary_{ts}.png'
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {p2}")

    # ── Page 3: Detailed trade analysis for best/worst/avg periods ──
    sorted_idx = np.argsort([r['return_pct'] for r in results])
    best = results[sorted_idx[-1]]
    worst = results[sorted_idx[0]]
    median_idx = sorted_idx[len(sorted_idx)//2]
    median = results[median_idx]

    fig3 = plt.figure(figsize=(18, 10))
    gs3 = GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.30)

    for col_i, (label, r) in enumerate([
        (f'Worst: {worst["start"]}~{worst["end"][5:]} ({worst["return_pct"]:+.2f}%)', worst),
        (f'Median: {median["start"]}~{median["end"][5:]} ({median["return_pct"]:+.2f}%)', median),
        (f'Best: {best["start"]}~{best["end"][5:]} ({best["return_pct"]:+.2f}%)', best),
    ]):
        # Equity
        ax_eq = fig3.add_subplot(gs3[0, col_i])
        p = r['portfolio']
        ax_eq.plot(p, linewidth=1.5, color='darkblue')
        ax_eq.axhline(INITIAL_CAPITAL, color='gray', ls='--', alpha=0.4)
        ax_eq.set_title(label, fontsize=9)
        ax_eq.grid(True, alpha=0.25)

        # Drawdown
        ax_dd = fig3.add_subplot(gs3[1, col_i])
        peak = np.maximum.accumulate(p)
        dd = (peak - p) / peak * 100
        ax_dd.fill_between(range(len(dd)), dd, 0, color='red', alpha=0.3)
        ax_dd.set_ylabel('Drawdown %')
        ax_dd.set_xlabel('Step')
        ax_dd.grid(True, alpha=0.25)

    fig3.suptitle('PPO Simulation — Best / Worst / Median Periods Detail', fontsize=14, y=1.01)
    p3 = f'results/batch_detail_{ts}.png'
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {p3}")

    # ── Page 4: Trade distribution across all periods ──
    all_trade_prices = []
    all_trade_sizes = []
    for r in results:
        for t in r['trades']:
            all_trade_prices.append(t[2])
            all_trade_sizes.append(t[3])

    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_hist_p, ax_hist_s, ax_scatter, ax_blank = axes.flatten()

    ax_hist_p.hist(all_trade_prices, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax_hist_p.set_xlabel('Trade Price')
    ax_hist_p.set_ylabel('Frequency')
    ax_hist_p.set_title(f'Trade Price Distribution (n={len(all_trade_prices)})')
    ax_hist_p.grid(True, alpha=0.25)

    ax_hist_s.hist(all_trade_sizes, bins=50, color='coral', alpha=0.7, edgecolor='white')
    ax_hist_s.set_xlabel('Trade Size (lots)')
    ax_hist_s.set_ylabel('Frequency')
    ax_hist_s.set_title(f'Trade Size Distribution (lots)')
    ax_hist_s.grid(True, alpha=0.25)

    ax_scatter.scatter(all_trade_prices, all_trade_sizes, alpha=0.3, s=5, c='purple')
    ax_scatter.set_xlabel('Price')
    ax_scatter.set_ylabel('Size (lots)')
    ax_scatter.set_title('Trade Price vs Size')
    ax_scatter.grid(True, alpha=0.25)

    ax_blank.axis('off')
    # Summary text
    total_trades = sum(r['trade_count'] for r in results)
    total_steps = sum(r['steps'] for r in results)
    all_ret = [r['return_pct'] for r in results]
    win_rate = sum(1 for v in all_ret if v > 0) / len(all_ret) * 100
    summary_text = (
        f"Total Periods: {len(results)}\n"
        f"Total Trades: {total_trades}\n"
        f"Avg Trades/Period: {total_trades/len(results):.0f}\n"
        f"Avg Steps/Period: {total_steps/len(results):.1f}\n"
        f"Avg Return: {np.mean(all_ret):+.2f}%\n"
        f"Win Rate (months): {win_rate:.0f}%\n"
        f"Avg Sharpe: {np.mean(srs):.2f}\n"
        f"Avg MaxDD: {avg_dd:.2%}\n"
    )
    ax_blank.text(0.1, 0.5, summary_text, fontsize=11, va='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig4.suptitle('PPO Simulation — Trade Analysis Across All Periods', fontsize=14, y=1.01)
    p4 = f'results/batch_trades_{ts}.png'
    fig4.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {p4}")

    return [p1, p2, p3, p4]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppo', default='checkpoints/ppo_best.pth')
    parser.add_argument('--stocks', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Batch Simulation: {len(PERIODS)} monthly periods")
    print(f"PPO: {args.ppo}, Universe: {args.stocks} stocks")
    print(f"{'='*60}")

    # Load agent once
    agent = PPOAgent(input_dim=OBS_TOTAL, output_dim=OUTPUT_DIM)
    agent.load(args.ppo)
    agent.actor_critic.eval()
    print("Agent loaded.\n")

    results = []
    for i, (start, end) in enumerate(PERIODS):
        seed = args.seed + i
        print(f"[{i+1}/{len(PERIODS)}] {start} ~ {end}  (seed={seed})")
        r = sim_one(agent, start, end, max_stocks=args.stocks, seed=seed)
        if r:
            print(f"  → Return: {r['return_pct']:+.2f}% | "
                  f"Sharpe: {r['sharpe']:.2f} | "
                  f"MaxDD: {r['max_dd']:.2%} | "
                  f"Trades: {r['trade_count']}")
            results.append(r)
        else:
            print(f"  → SKIPPED (no data)")
        print()

    if not results:
        print("No periods completed!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Generating report ({len(results)} periods)...")
    paths = make_report(results)

    # Print final summary
    returns = [r['return_pct'] for r in results]
    srs = [r['sharpe'] for r in results]
    dds = [r['max_dd'] for r in results]
    trs = [r['trade_count'] for r in results]
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY ({len(results)} periods)")
    print(f"{'='*60}")
    print(f"  Avg Return:    {np.mean(returns):+.2f}%  (min={min(returns):+.2f}%, max={max(returns):+.2f}%)")
    print(f"  Win Rate:      {sum(1 for v in returns if v>0)/len(returns)*100:.0f}%")
    print(f"  Avg Sharpe:    {np.mean(srs):.2f}")
    print(f"  Avg MaxDD:     {np.mean(dds):.2%}")
    print(f"  Avg Trades:    {np.mean(trs):.0f}  (total={sum(trs)})")
    print(f"{'='*60}")
    print(f"  Reports saved:")
    for p in paths:
        print(f"    {p}")
    print(f"{'='*60}")
