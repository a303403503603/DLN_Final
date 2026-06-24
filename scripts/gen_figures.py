import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'slide_output', 'figures')
os.makedirs(OUT, exist_ok=True)

COLORS = {
    'data': '#4472C4',
    'gru': '#548235',
    'ppo': '#BF8F00',
    'exec': '#C00000',
    'ctrl': '#7030A0',
    'bg': '#F2F2F2',
}


def box(ax, x, y, w, h, label, color, sub=None, lw=1.5):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.08", facecolor=color,
                          edgecolor='#333333', linewidth=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + 0.02, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=4)
    if sub:
        ax.text(x, y - 0.22, sub, ha='center', va='center', fontsize=6.5,
                color='white', alpha=0.85, zorder=4)


def arrow(ax, x1, y1, x2, y2, label='', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color='#555555',
                                lw=1.5, connectionstyle='arc3,rad=0'), zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.08
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=7,
                color='#555555', fontstyle='italic')


def varrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=1.5, connectionstyle='arc3,rad=0'), zorder=2)
    if label:
        mx, my = (x1 + x2) / 2 + 0.1, (y1 + y2) / 2
        ax.text(mx, my, label, ha='left', va='center', fontsize=7,
                color='#555555', fontstyle='italic')


# ── fig01: System Architecture ──
def gen_fig01():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    ax.set_facecolor('white')

    # Title
    ax.text(5, 6.3, 'System Architecture', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#333333')

    # Layer 4: Discord Bot (top)
    box(ax, 5, 5.6, 6, 0.5, 'Discord Bot Control Panel', COLORS['ctrl'],
        sub='Schedule / Manual Confirm / Dashboard')
    # Layer 3: Execution
    box(ax, 5, 4.5, 6, 0.5, 'Trade Execution — NCKU Simulated API', COLORS['exec'],
        sub='Place Order / Query Quote / Portfolio Reconciliation')
    # Layer 2: PPO
    box(ax, 5, 3.4, 6, 0.5, 'PPO Actor-Critic — Decision Engine', COLORS['ppo'],
        sub='Dirichlet(51) → Target Weights / Position Limit / Stop-Loss')
    # Layer 2b: GRU
    box(ax, 5, 2.3, 6, 0.5, 'MultiTimeGRU — Price Prediction', COLORS['gru'],
        sub='8 Parallel Windows / Attention Fusion / 28-Dim Forecast')
    # Layer 1: Data
    box(ax, 5, 1.2, 6, 0.5, 'Data Layer — Storage & Cache', COLORS['data'],
        sub='Kaggle CSV / API Cache / GRU Cache / Portfolio State JSON')

    # Vertical arrows between layers
    varrow(ax, 5, 1.45, 5, 2.05, 'Load features + latent')
    varrow(ax, 5, 2.55, 5, 3.15, 'Prediction (28d) + latent (2048)')
    varrow(ax, 5, 3.65, 5, 4.25, 'Target weights (51-dim)')
    varrow(ax, 5, 4.75, 5, 5.35, 'Execution report / State update')
    varrow(ax, 7.5, 5.35, 7.5, 4.75, 'Manual confirm / Schedule')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig01_system_architecture.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig01 done')


# ── fig02: Daily Pipeline ──
def gen_fig02():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.text(5, 4.8, 'Daily Pipeline — Timeline', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#333333')

    # Swimlane: Data / Decision / Execution
    # Lane labels
    ax.text(0.3, 3.7, 'Data', fontsize=9, fontweight='bold', color=COLORS['data'], va='center')
    ax.text(0.3, 2.5, 'Decision', fontsize=9, fontweight='bold', color=COLORS['ppo'], va='center')
    ax.text(0.3, 1.3, 'Execution', fontsize=9, fontweight='bold', color=COLORS['exec'], va='center')

    # Horizontal lines for lanes
    ax.axhline(y=3.2, xmin=0.05, xmax=0.95, color='#DDDDDD', lw=1, zorder=1)
    ax.axhline(y=2.0, xmin=0.05, xmax=0.95, color='#DDDDDD', lw=1, zorder=1)
    ax.axhline(y=0.8, xmin=0.05, xmax=0.95, color='#DDDDDD', lw=1, zorder=1)

    # Time markers
    for tx, tl in [(1.5, '09:30'), (4.0, '10:00'), (6.5, '10:01~'), (8.5, '16:30')]:
        ax.text(tx, 4.3, tl, ha='center', fontsize=8, fontweight='bold', color='#666666')
        ax.axvline(x=tx, ymin=0.1, ymax=0.85, color='#DDDDDD', lw=0.8, ls='--', zorder=1)

    # Boxes on lanes
    box(ax, 2.8, 3.7, 2.6, 0.45, 'Load Cache + GRU', COLORS['data'],
        sub='price_data / gru_cache / api_cache')
    box(ax, 5.5, 3.7, 2.6, 0.45, 'Kaggle Update', COLORS['data'],
        sub='Download / Extract / Overwrite')

    box(ax, 4.0, 2.5, 2.0, 0.45, 'Compute Proposal', COLORS['ppo'],
        sub='PPO forward + position limit')
    box(ax, 6.5, 2.5, 2.0, 0.45, 'User Confirm', COLORS['ctrl'],
        sub='Discord Embed + Button')

    box(ax, 5.5, 1.3, 2.6, 0.45, 'Batch Market Orders', COLORS['exec'],
        sub='NCKU API / Update State')
    box(ax, 8.5, 1.3, 2.6, 0.45, 'Reconcile', COLORS['exec'],
        sub='API vs State / Trade Log')

    # Arrows: data flow
    arrow(ax, 4.1, 3.7, 4.1, 2.95, 'features + cache')
    arrow(ax, 5.0, 2.5, 5.0, 1.75, 'target weights')
    arrow(ax, 7.5, 2.5, 7.5, 2.95, 'confirm')
    arrow(ax, 7.5, 3.15, 7.2, 3.7, 'trigger update')
    arrow(ax, 6.8, 1.3, 8.2, 1.3, 'post-trade')
    arrow(ax, 2.8, 1.3, 2.8, 0.55, 'update portfolio_state.json')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig02_daily_pipeline.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig02 done')


# ── fig03: GRU Architecture ──
def gen_fig03():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.text(5, 5.8, 'MultiTimeGRU — Network Architecture', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#333333')

    windows = [1, 3, 5, 7, 14, 30, 60, 120]
    colors = plt.cm.YlGn([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Input layer
    box(ax, 5, 5.2, 8, 0.35, 'Input: 5 Ratio Features (close_ret, open_ret, high_close, low_close, vol_ret)', COLORS['data'],
        sub='StandardScaler normalized')

    # 8 parallel GRUs
    for i, (w, c) in enumerate(zip(windows, colors)):
        bx = 1.0 + i * 1.0
        rect = FancyBboxPatch((bx - 0.35, 3.4), 0.7, 0.8,
                              boxstyle="round,pad=0.06", facecolor=c,
                              edgecolor='#333333', linewidth=1, zorder=3)
        ax.add_patch(rect)
        ax.text(bx, 3.95, f'W={w}', ha='center', va='center', fontsize=7,
                fontweight='bold', color='#333333', zorder=4)
        ax.text(bx, 3.55, 'GRU', ha='center', va='center', fontsize=6,
                color='#333333', alpha=0.7, zorder=4)
        # Arrow from input to each GRU
        ax.annotate('', xy=(bx, 3.4), xytext=(bx, 4.85),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8), zorder=2)

    # Hidden state row
    ax.text(5, 3.1, 'Hidden States (256-dim each)', ha='center', va='center',
            fontsize=8, color='#666666', fontstyle='italic')

    # Fusion block
    rect = FancyBboxPatch((3.0, 1.8), 4, 0.9,
                          boxstyle="round,pad=0.08", facecolor=COLORS['gru'],
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(5, 2.35, 'Fusion Pipeline', ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=4)
    ax.text(5, 2.05, 'MLP(2048→512→1024→2048) + MultiTimeAttention(8)', ha='center', va='center',
            fontsize=6.5, color='white', alpha=0.85, zorder=4)

    # Arrow from GRUs to fusion
    for i in range(8):
        bx = 1.0 + i * 1.0
        ax.annotate('', xy=(bx, 2.25), xytext=(bx, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8), zorder=2)

    # Latent block
    rect = FancyBboxPatch((3.5, 1.0), 3, 0.6,
                          boxstyle="round,pad=0.06", facecolor='#8B5CF6',
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(5, 1.3, 'Latent: 256-dim (fused representation)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)

    # Output prediction
    rect = FancyBboxPatch((3.0, 0.2), 4, 0.6,
                          boxstyle="round,pad=0.06", facecolor=COLORS['exec'],
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(5, 0.5, 'Prediction: 28-dim (7 Horizons × 4 OHLC)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)

    arrow(ax, 5, 1.0, 5, 0.8, 'Linear(256→28)')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig03_gru_architecture.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig03 done')


# ── fig04: Pool Flow ──
def gen_fig04():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.text(4.5, 4.8, 'Stock Pool Flow — From Universe to Portfolio', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#333333')

    # Level 1: Full universe
    rect = FancyBboxPatch((0.5, 3.6), 8, 0.7,
                          boxstyle="round,pad=0.08", facecolor=COLORS['data'],
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(4.5, 3.95, 'Stock Universe: 1,941 Taiwan Stocks', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)
    ax.text(4.5, 3.65, 'Kaggle dataset (all listed stocks with sufficient history)', ha='center', va='center',
            fontsize=7, color='white', alpha=0.85, zorder=4)

    # Arrow
    arrow(ax, 4.5, 3.6, 4.5, 3.0, 'GRU score ranking')

    # Level 2: Candidate pool
    rect = FancyBboxPatch((1.5, 2.4), 6, 0.6,
                          boxstyle="round,pad=0.08", facecolor=COLORS['gru'],
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(4.5, 2.7, 'Candidate Pool: Top 50 (GRU Score)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=4)
    ax.text(4.5, 2.45, 'GRU_score = pred_7d_close + 0.5 × pred_30d_close + Tech Boost × 1.15',
            ha='center', va='center', fontsize=6.5, color='white', alpha=0.85, zorder=4)

    # Split arrow
    ax.annotate('', xy=(3.0, 2.4), xytext=(3.0, 1.8),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5), zorder=2)
    ax.annotate('', xy=(6.0, 2.4), xytext=(6.0, 1.8),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5), zorder=2)

    ax.text(3.0, 2.1, 'Top 10 by ROI', ha='center', fontsize=7, color='#666666')
    ax.text(6.0, 2.1, 'Remaining 40', ha='center', fontsize=7, color='#666666')

    # Level 3a: Hold pool
    rect = FancyBboxPatch((0.5, 1.2), 3, 0.6,
                          boxstyle="round,pad=0.08", facecolor=COLORS['ppo'],
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(2.0, 1.5, 'Hold Pool: Top N (N=10)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=4)
    ax.text(2.0, 1.25, 'Long positions tracked with cost basis', ha='center', va='center',
            fontsize=6.5, color='white', alpha=0.85, zorder=4)

    # Level 3b: Watch pool
    rect = FancyBboxPatch((5.5, 1.2), 3, 0.6,
                          boxstyle="round,pad=0.08", facecolor=COLORS['exec'],
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(7.0, 1.5, 'Watch Pool: 40 Stocks', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=4)
    ax.text(7.0, 1.25, 'No position, GRU prediction only', ha='center', va='center',
            fontsize=6.5, color='white', alpha=0.85, zorder=4)

    # Arrows to PPO
    ax.annotate('', xy=(2.0, 1.2), xytext=(2.0, 0.5),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5), zorder=2)
    ax.annotate('', xy=(7.0, 1.2), xytext=(7.0, 0.5),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5), zorder=2)

    # PPO decision
    rect = FancyBboxPatch((1.5, 0.0), 6, 0.5,
                          boxstyle="round,pad=0.08", facecolor=COLORS['ctrl'],
                          edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(4.5, 0.25, 'PPO Dirichlet(51) → Rebalance: Cash (1) + Hold (10) + Watch (40)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)

    # Cycle back arrow
    ax.annotate('', xy=(8.2, 3.95), xytext=(8.2, 0.25),
                arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1, ls='--',
                                connectionstyle='arc3,rad=0.3'), zorder=2)
    ax.text(8.5, 2.0, 'daily\nupdate', ha='center', va='center',
            fontsize=6.5, color='#999999', fontstyle='italic')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig04_pool_flow.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig04 done')


if __name__ == '__main__':
    gen_fig01()
    gen_fig02()
    gen_fig03()
    gen_fig04()
    print('All figures generated.')
