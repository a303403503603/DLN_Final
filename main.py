"""PPO Stock Trading — 單一入口點"""
import sys, subprocess
from pipeline.run_daily_pipeline import cmd_fetch, cmd_gru, cmd_trade, cmd_daily, cmd_reconcile, cmd_status
from pipeline.config import DISCORD_TOKEN, DISCORD_CHANNEL_ID


def cmd_full():
    from pipeline.run_daily_pipeline import cmd_fetch, cmd_gru, cmd_trade
    rcs = [cmd_fetch(), cmd_gru(), cmd_trade()]
    return max(rcs) if rcs else 0


def cmd_bot():
    if not DISCORD_TOKEN or not DISCORD_CHANNEL_ID:
        print("DISCORD_TOKEN / DISCORD_CHANNEL_ID not set in .env")
        return 1
    from pipeline.discord_bot import TradeBot
    bot = TradeBot()
    bot.run(DISCORD_TOKEN, log_handler=None)
    return 0


CMDS = {
    'fetch': cmd_fetch,
    'gru': cmd_gru,
    'trade': cmd_trade,
    'daily': cmd_daily,
    'reconcile': cmd_reconcile,
    'status': cmd_status,
    'full': cmd_full,
    'bot': cmd_bot,
    'train-gru': lambda: subprocess.call([sys.executable, '-m', 'training.train_gru'] + sys.argv[2:]),
    'train-ppo': lambda: subprocess.call([sys.executable, '-m', 'training.train_ppo'] + sys.argv[2:]),
    'batch-sim': lambda: subprocess.call([sys.executable, '-m', 'training.run_batch_sim'] + sys.argv[2:]),
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python main.py <command> [args...]")
        print()
        print("Pipeline:")
        print("  daily         Fetch + GRU + reconcile (盤後執行)")
        print("  fetch         API data fetch only")
        print("  gru           GRU cache update only")
        print("  trade         PPO 交易執行")
        print("  reconcile     對帳（核對掛單 vs API）")
        print("  status        顯示 pipeline 狀態")
        print("  full          fetch + gru + trade")
        print()
        print("Bot:")
        print("  bot           啟動 Discord bot（前景）")
        print()
        print("Training:")
        print("  train-gru     GRU 預訓練")
        print("  train-ppo     PPO 強化學習訓練 [options]")
        print("  batch-sim     多時段回測")
        sys.exit(0 if sys.argv[1:] else 1)

    cmd = sys.argv[1]
    fn = CMDS.get(cmd)
    if fn:
        sys.exit(fn())
    print(f"Unknown command: {cmd}")
    print("Use 'python main.py --help' for available commands")
    sys.exit(1)
