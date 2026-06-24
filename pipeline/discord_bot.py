"""Discord bot — 交易確認、控制面板、投資組合查詢"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import discord
from discord import ui
from discord.ext import commands, tasks
from datetime import datetime, timedelta
import asyncio
import pandas as pd

from pipeline.config import (
    DISCORD_TOKEN, DISCORD_CHANNEL_ID, INITIAL_CAPITAL,
    GRU_CACHE_DIR, GRU_HIDDEN_SIZE, TIME_WINDOWS, N_HORIZONS, PRED_HORIZONS,
    COMMISSION_BUY, PORTFOLIO_STATE_FILE,
)
from data.stock_api import Get_User_Stocks, get_taiwan_stock_data, Buy_Stock, Sell_Stock, get_all_stock_list
from pipeline import _utils
from pipeline.run_daily_pipeline import LiveTrader, Proposal, phase_fetch, phase_gru, reconcile_state
import numpy as np

try:
    import twstock
    HAS_TWSTOCK = True
except ImportError:
    HAS_TWSTOCK = False
    twstock = None

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('discord_bot')


async def _safe_followup(interaction, content=None, **kwargs):
    """Discord followup.send wrapper that swallows 401/HTTP errors gracefully.

    接受 content= 或 embed= (or file= etc.) 兩種用法。
    """
    try:
        if content is not None or kwargs:
            return await interaction.followup.send(content, **kwargs)
        return None
    except discord.errors.HTTPException as e:
        if e.code == 50027 or e.status == 401:
            log.error(f"Webhook token 過期 (401)，請更新 DISCORD_TOKEN")
        else:
            log.error(f"followup.send HTTPException: {e}")
        return None
    except Exception as e:
        log.error(f"followup.send error: {e}")
        return None


def _handle_http_exception(e, context=""):
    """Handle HTTPException from discord interactions."""
    if isinstance(e, discord.errors.HTTPException):
        if e.code == 50027 or e.status == 401:
            log.error(f"[{context}] Webhook token 過期 (401)：{e}")
            return True
    log.error(f"[{context}] {type(e).__name__}: {e}", exc_info=True)
    return False


async def _safe_channel_send(channel, *args, **kwargs):
    """channel.send wrapper that swallows 401/HTTP errors gracefully."""
    try:
        return await channel.send(*args, **kwargs)
    except discord.errors.HTTPException as e:
        if e.code == 50027 or e.status == 401:
            log.error(f"[channel.send] Webhook token 過期 (401)，請更新 DISCORD_TOKEN")
        else:
            log.error(f"[channel.send] HTTPException: {e}")
        return None
    except Exception as e:
        log.error(f"[channel.send] error: {e}", exc_info=True)
        return None


def _premium_str(price, close):
    if not close or close <= 0:
        return ""
    pct = (price - close) / close * 100
    sign = "+" if pct >= 0 else ""
    return f"較現價{sign}{pct:.1f}%"


def _buy_sell_field(title_prefix, entries, sign="+", mode="buy"):
    if not entries:
        return []
    chunk_size = 15
    fields = []
    total = 0
    for i in range(0, len(entries), chunk_size):
        chunk = entries[i:i + chunk_size]
        lines = []
        for b in chunk:
            cost = b['lots'] * 1000 * b['price']
            total += cost
            code = b['code']
            close = b.get('close', 0)
            close_str = f" 現${close}" if close else ""

            if mode == "buy" and b.get('preds'):
                pred_h1 = b['preds'][0] * 100
                sign_p = "+" if pred_h1 >= 0 else ""
                lines.append(f"{code}  {b['lots']}張 @${b['price']:.1f}{close_str}  GRU預測{sign_p}{pred_h1:.1f}%")
            elif mode == "sell" and b.get('cost') and b['cost'] > 0:
                roi = (b['price'] - b['cost']) / b['cost'] * 100
                sign_r = "+" if roi >= 0 else ""
                lines.append(f"{code}  {b['lots']}張 @${b['price']:.1f}{close_str}  投報{sign_r}{roi:.1f}%（成本${b['cost']:.1f}）")
            elif close:
                premium = _premium_str(b['price'], close)
                if premium:
                    lines.append(f"{code}  {b['lots']}張 @${b['price']:.1f}{close_str}  ({premium})")
                else:
                    lines.append(f"{code}  {b['lots']}張 @${b['price']:.1f}{close_str}")
            else:
                lines.append(f"{code}  {b['lots']}張 @${b['price']:.1f}")
        text = "\n".join(lines)
        if i + chunk_size < len(entries):
            field_name = f"{title_prefix} ({i + 1}-{i + len(chunk)})"
        else:
            field_name = f"{title_prefix} ({len(entries)})"
            text += f"\n**{sign}總額：${total:,}**"
        fields.append((field_name, text))
    return fields


def build_proposal_embed(proposal):
    embed = discord.Embed(
        title=f"交易提案 {proposal.date}",
        color=0x00ff00,
    )
    embed.add_field(name="現金", value=f"${proposal.cash:,.0f}", inline=True)
    embed.add_field(name="總資產", value=f"${proposal.port_value:,.0f}", inline=True)
    embed.add_field(name="現金比", value=f"{proposal.cash_ratio:.1%}", inline=True)

    buy_fields = _buy_sell_field("買進", proposal.buys, sign="+", mode="buy")
    for name, text in buy_fields:
        embed.add_field(name=name, value=text, inline=False)

    sell_fields = _buy_sell_field("賣出", proposal.sells, sign="+", mode="sell")
    for name, text in sell_fields:
        embed.add_field(name=name, value=text, inline=False)

    if not proposal.buys and not proposal.sells:
        embed.description = "無交易提案。"

    embed.set_footer(text="確認 → 執行 | 調整 → 修改 | 取消")
    return embed


def parse_adjustments(text, proposal):
    buy_map = {b['code']: dict(b) for b in proposal.buys}
    sell_map = {s['code']: dict(s) for s in proposal.sells}

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            return None

        code = parts[0].strip()
        if code not in buy_map and code not in sell_map:
            continue

        lots = None
        price = None

        for part in parts[1:]:
            part = part.strip()
            if part.startswith('@'):
                try:
                    price = float(part[1:])
                except ValueError:
                    return None
            else:
                try:
                    lots = int(float(part))
                except ValueError:
                    return None

        if code in buy_map:
            if lots == 0:
                del buy_map[code]
            else:
                if lots is not None:
                    buy_map[code]['lots'] = lots
                if price is not None:
                    buy_map[code]['price'] = int(price)
                    buy_map[code]['close'] = int(price)

        if code in sell_map:
            if lots == 0:
                del sell_map[code]
            else:
                if lots is not None:
                    sell_map[code]['lots'] = lots
                if price is not None:
                    sell_map[code]['price'] = int(price)
                    sell_map[code]['close'] = int(price)

    return list(buy_map.values()), list(sell_map.values())


class AdjustModal(ui.Modal):
    def __init__(self, bot, proposal, message):
        super().__init__(title="調整交易提案")
        self.bot = bot
        self.proposal = proposal
        self.message = message
        self.adjustments = ui.TextInput(
            label="調整：代碼 張數 [@價格]",
            style=discord.TextStyle.paragraph,
            placeholder=(
                "一行一檔，範例：\n"
                "2438 80 @26   代碼, 張數, 價格\n"
                "3041 150      只改張數\n"
                "2392 0        移除該檔\n"
                "留空則不修改。"
            ),
            required=False,
        )
        self.add_item(self.adjustments)

    async def on_submit(self, interaction: discord.Interaction):
        text = self.adjustments.value.strip()
        if not text:
            await interaction.response.send_message("無任何變更。", ephemeral=True)
            return

        result = parse_adjustments(text, self.proposal)
        if result is None:
            await interaction.response.send_message(
                "格式錯誤。請使用：`代碼 張數 [@價格]`\n例如：`2438 80 @26`", ephemeral=True)
            return

        new_buys, new_sells = result
        self.proposal.buys = new_buys
        self.proposal.sells = new_sells

        embed = build_proposal_embed(self.proposal)
        view = TradeProposalView(self.bot, self.proposal)
        await interaction.response.edit_message(embed=embed, view=view)


class ManualBuyModal(ui.Modal):
    def __init__(self, bot):
        super().__init__(title="手動買入")
        self.bot = bot
        self.stock_code = ui.TextInput(
            label="股票代碼",
            placeholder="例如 2330",
            min_length=4,
            max_length=4,
            required=True,
        )
        self.add_item(self.stock_code)
        self.lots = ui.TextInput(
            label="張數",
            placeholder="例如 10",
            required=True,
        )
        self.add_item(self.lots)
        self.price = ui.TextInput(
            label="限價（留空 = 市價）",
            placeholder="例如 25.0，或留空",
            required=False,
        )
        self.add_item(self.price)

    async def on_submit(self, interaction: discord.Interaction):
        code = self.stock_code.value.strip().zfill(4)
        try:
            lots = int(self.lots.value.strip())
            if lots <= 0:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("張數格式錯誤，請輸入正整數。", ephemeral=True)
            return

        price_raw = self.price.value.strip()
        if price_raw:
            try:
                limit_price = int(float(price_raw))
                if limit_price <= 0:
                    raise ValueError
            except ValueError:
                await interaction.response.send_message("價格格式錯誤，請輸入正數。", ephemeral=True)
                return
        else:
            limit_price = 0

        await interaction.response.defer(ephemeral=True)

        embed = discord.Embed(title=f"手動買入 {code}", color=0x3498db)

        # 即時價（不分時段都試 twstock）
        last_close = 0
        rt_prices = await asyncio.to_thread(_get_realtime_batch, [code])
        rt_info = rt_prices.get(code)
        if rt_info and rt_info.get('close', 0) > 0:
            last_close = rt_info['close']
            embed.add_field(name="即時現價", value=f"${last_close:.2f}", inline=True)

        if last_close == 0:
            end = datetime.now().strftime('%Y/%m/%d')
            start = (datetime.now() - timedelta(days=10)).strftime('%Y/%m/%d')
            try:
                df = await asyncio.to_thread(get_taiwan_stock_data, code, start, end)
                if df is not None and len(df) > 0:
                    last_close = float(df.iloc[-1]['close'])
                    embed.add_field(name="昨收", value=f"${last_close:.2f}", inline=True)
            except Exception:
                pass

        gru_path = os.path.join(GRU_CACHE_DIR, f'{code}.npy')
        if os.path.exists(gru_path):
            try:
                cache = np.load(gru_path).astype(np.float32)
                if len(cache) > 0:
                    pred_start = GRU_HIDDEN_SIZE * len(TIME_WINDOWS)
                    preds = cache[-1, pred_start:pred_start + N_HORIZONS]
                    lines = []
                    for h, p in zip(PRED_HORIZONS, preds[:3]):
                        pct = p * 100
                        sign = "+" if pct >= 0 else ""
                        lines.append(f"t+{h}  {sign}{pct:.1f}%")
                    embed.add_field(name="GRU 預測", value="\n".join(lines), inline=True)
            except Exception:
                pass

        try:
            acct, pw = _utils.get_credentials()
            raw = Get_User_Stocks(acct, pw)
            held_lots = 0
            held_cost = 0
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get('stock_code_id', '') == code:
                        held_lots = int(item.get('shares', 0))
                        held_cost = float(item.get('beginning_price', 0))
                        break
            if held_lots > 0:
                embed.add_field(name="目前持有", value=f"{held_lots} 張（成本 ${held_cost:.0f}）", inline=True)
        except Exception:
            pass

        try:
            if limit_price == 0:
                limit_price = int(last_close) if last_close > 0 else 0
                if limit_price == 0:
                    await _safe_followup(
                        interaction,
                        "無法取得參考價格，請手動輸入限價。", ephemeral=True)
                    return

            acct, pw = _utils.get_credentials()
            result = await asyncio.to_thread(
                Buy_Stock, acct, pw, code, lots, limit_price,
            )
            ok = result[0] if isinstance(result, tuple) else result
            if ok:
                cost_total = lots * 1000 * limit_price * (1 + COMMISSION_BUY)
                embed.color = 0x00ff00
                embed.add_field(
                    name="已送出買單",
                    value=f"{lots} 張 @ ${limit_price} = ${lots*1000*limit_price:,}\n（含手續費 ${cost_total:,.0f}）",
                    inline=False,
                )

                # 更新 portfolio_state.json
                state_path = PORTFOLIO_STATE_FILE
                state = {}
                if os.path.exists(state_path):
                    try:
                        with open(state_path, encoding='utf-8') as f:
                            state = json.load(f)
                    except Exception:
                        pass
                state['cash'] = state.get('cash', INITIAL_CAPITAL) - cost_total
                if 'pending_buys' not in state:
                    state['pending_buys'] = {}
                state['pending_buys'][code] = {
                    'shares': lots,
                    'price': limit_price,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                }
                state['updated_at'] = datetime.now().isoformat()
                with open(state_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)

                embed.add_field(name="剩餘現金", value=f"${state['cash']:,.0f}", inline=False)
            else:
                embed.color = 0xff0000
                embed.add_field(name="下單失敗", value="API 回傳失敗", inline=False)
        except Exception as e:
            log.error(f"手動買入 {code} 錯誤: {e}", exc_info=True)
            embed.color = 0xff0000
            embed.add_field(name="下單失敗", value=str(e), inline=False)

        await _safe_followup(interaction, embed=embed, ephemeral=True)


class ManualSellModal(ui.Modal):
    def __init__(self, bot):
        super().__init__(title="手動賣出")
        self.bot = bot
        self.stock_code = ui.TextInput(
            label="股票代碼",
            placeholder="例如 2330",
            min_length=4,
            max_length=4,
            required=True,
        )
        self.add_item(self.stock_code)
        self.lots = ui.TextInput(
            label="張數",
            placeholder="例如 10",
            required=True,
        )
        self.add_item(self.lots)
        self.price = ui.TextInput(
            label="限價（留空 = 市價）",
            placeholder="例如 25.0，或留空",
            required=False,
        )
        self.add_item(self.price)

    async def on_submit(self, interaction: discord.Interaction):
        code = self.stock_code.value.strip().zfill(4)
        try:
            lots = int(self.lots.value.strip())
            if lots <= 0:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("張數格式錯誤，請輸入正整數。", ephemeral=True)
            return

        price_raw = self.price.value.strip()
        if price_raw:
            try:
                limit_price = int(float(price_raw))
                if limit_price <= 0:
                    raise ValueError
            except ValueError:
                await interaction.response.send_message("價格格式錯誤，請輸入正數。", ephemeral=True)
                return
        else:
            limit_price = 0

        await interaction.response.defer(ephemeral=True)

        embed = discord.Embed(title=f"手動賣出 {code}", color=0x3498db)

        # 即時價（不分時段都試 twstock）
        last_close = 0
        rt_prices = await asyncio.to_thread(_get_realtime_batch, [code])
        rt_info = rt_prices.get(code)
        if rt_info and rt_info.get('close', 0) > 0:
            last_close = rt_info['close']
            embed.add_field(name="即時現價", value=f"${last_close:.2f}", inline=True)

        if last_close == 0:
            end = datetime.now().strftime('%Y/%m/%d')
            start = (datetime.now() - timedelta(days=10)).strftime('%Y/%m/%d')
            try:
                df = await asyncio.to_thread(get_taiwan_stock_data, code, start, end)
                if df is not None and len(df) > 0:
                    last_close = float(df.iloc[-1]['close'])
                    embed.add_field(name="昨收", value=f"${last_close:.2f}", inline=True)
            except Exception:
                pass

        gru_path = os.path.join(GRU_CACHE_DIR, f'{code}.npy')
        if os.path.exists(gru_path):
            try:
                cache = np.load(gru_path).astype(np.float32)
                if len(cache) > 0:
                    pred_start = GRU_HIDDEN_SIZE * len(TIME_WINDOWS)
                    preds = cache[-1, pred_start:pred_start + N_HORIZONS]
                    lines = []
                    for h, p in zip(PRED_HORIZONS, preds[:3]):
                        pct = p * 100
                        sign = "+" if pct >= 0 else ""
                        lines.append(f"t+{h}  {sign}{pct:.1f}%")
                    embed.add_field(name="GRU 預測", value="\n".join(lines), inline=True)
            except Exception:
                pass

        try:
            acct, pw = _utils.get_credentials()
            raw = Get_User_Stocks(acct, pw)
            held_lots = 0
            held_cost = 0
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get('stock_code_id', '') == code:
                        held_lots = int(item.get('shares', 0))
                        held_cost = float(item.get('beginning_price', 0))
                        break
            if held_lots > 0:
                embed.add_field(name="目前持有", value=f"{held_lots} 張（成本 ${held_cost:.0f}）", inline=True)
        except Exception:
            pass

        try:
            if limit_price == 0:
                limit_price = int(last_close) if last_close > 0 else 0
                if limit_price == 0:
                    await _safe_followup(interaction, "無法取得參考價格，請手動輸入限價。", ephemeral=True)
                    return

            acct, pw = _utils.get_credentials()
            ok, sell_msg = await asyncio.to_thread(
                Sell_Stock, acct, pw, code, lots, limit_price,
            )
            if ok:
                proceeds = lots * 1000 * limit_price * (1 - 0.003)
                embed.color = 0x00ff00
                embed.add_field(
                    name="已送出賣單",
                    value=f"{lots} 張 @ ${limit_price} = ${lots*1000*limit_price:,}\n（扣除手續費 ${proceeds:,.0f}）",
                    inline=False,
                )

                state_path = PORTFOLIO_STATE_FILE
                state = {}
                if os.path.exists(state_path):
                    try:
                        with open(state_path, encoding='utf-8') as f:
                            state = json.load(f)
                    except Exception:
                        pass
                if 'pending_sells' not in state:
                    state['pending_sells'] = {}
                state['pending_sells'][code] = {
                    'shares': lots,
                    'price': limit_price,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                }
                state['updated_at'] = datetime.now().isoformat()
                with open(state_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
            else:
                embed.color = 0xff0000
                embed.add_field(name="下單失敗", value=f"API: {sell_msg}", inline=False)
        except Exception as e:
            log.error(f"手動賣出 {code} 錯誤: {e}", exc_info=True)
            embed.color = 0xff0000
            embed.add_field(name="下單失敗", value=str(e), inline=False)

        await _safe_followup(interaction, embed=embed, ephemeral=True)


class PriceModal(ui.Modal):
    def __init__(self):
        super().__init__(title="查詢股價")
        self.stock_code = ui.TextInput(
            label="股票代碼",
            placeholder="例如 2330",
            min_length=4,
            max_length=4,
            required=True,
        )
        self.add_item(self.stock_code)

    async def on_submit(self, interaction: discord.Interaction):
        code = self.stock_code.value.strip().zfill(4)
        print(f'[PriceModal:on_submit] code={code}', flush=True)
        try:
            await interaction.response.defer(ephemeral=True)
            print(f'[PriceModal:on_submit] deferred', flush=True)
        except Exception as e:
            print(f'[PriceModal:on_submit] defer fail: {e}', flush=True)
            return

        try:
            embed = discord.Embed(title=f"{code} 最新股價", color=0x3498db)

            # 即時價（不分時段都試 twstock）
            rt_info = None
            rt_prices = await asyncio.to_thread(_get_realtime_batch, [code])
            rt_info = rt_prices.get(code)

            if rt_info and rt_info.get('close', 0) > 0:
                c = rt_info['close']
                o = rt_info.get('open', c)
                h = rt_info.get('high', c)
                lo = rt_info.get('low', c)
                embed.add_field(name="即時現價", value=f"${c:.2f}", inline=True)
                embed.add_field(name="開盤", value=f"${o:.2f}", inline=True)
                embed.add_field(name="最高", value=f"${h:.2f}", inline=True)
                embed.add_field(name="最低", value=f"${lo:.2f}", inline=True)
                embed.add_field(name="資料來源", value="twstock 即時", inline=True)
            else:
                # fallback 到 api_cache
                end = datetime.now().strftime('%Y/%m/%d')
                start = (datetime.now() - timedelta(days=10)).strftime('%Y/%m/%d')
                df = await asyncio.to_thread(get_taiwan_stock_data, code, start, end)
                if df is not None and len(df) > 0:
                    last = df.iloc[-1]
                    date_val = last.get('date', 'N/A')
                    if hasattr(date_val, 'date'):
                        date_str = date_val.date().isoformat()
                    else:
                        date_str = str(date_val).split(' ')[0]
                    embed.add_field(name="日期", value=date_str, inline=True)
                    embed.add_field(name="收盤", value=f"${last.get('close', 0):.2f}", inline=True)
                    embed.add_field(name="最高", value=f"${last.get('high', 0):.2f}", inline=True)
                    embed.add_field(name="最低", value=f"${last.get('low', 0):.2f}", inline=True)
                    embed.add_field(name="開盤", value=f"${last.get('open', 0):.2f}", inline=True)
                    embed.add_field(name="成交量", value=str(int(last.get('Trading_Volume', last.get('capacity', 0)))), inline=True)
                else:
                    embed.add_field(name="股價", value="無資料", inline=False)

            try:
                acct, pw = _utils.get_credentials()
                raw = Get_User_Stocks(acct, pw)
                held_lots = 0
                held_cost = 0
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict) and item.get('stock_code_id', '') == code:
                            held_lots = int(item.get('shares', 0))
                            held_cost = float(item.get('beginning_price', 0))
                            break
                if held_lots > 0:
                    embed.add_field(name="持有", value=f"{held_lots} 張（成本 ${held_cost:.0f}）", inline=False)
            except Exception:
                pass

            gru_path = os.path.join(GRU_CACHE_DIR, f'{code}.npy')
            if os.path.exists(gru_path):
                try:
                    data = np.load(gru_path, allow_pickle=True)
                    if isinstance(data, np.ndarray) and data.dtype == np.object_:
                        data = data.item()
                    if isinstance(data, dict):
                        cache = data['cache'].astype(np.float32)
                    else:
                        cache = data.astype(np.float32)
                    if len(cache) > 0:
                        pred_start = GRU_HIDDEN_SIZE * len(TIME_WINDOWS)
                        preds = cache[-1, pred_start:pred_start + N_HORIZONS]
                        lines = []
                        for h, p in zip(PRED_HORIZONS, preds):
                            pct = p * 100
                            sign = "+" if pct >= 0 else ""
                            lines.append(f"t+{h}  {sign}{pct:.1f}%")
                        embed.add_field(name="GRU 預測報酬", value="\n".join(lines), inline=False)
                except Exception as e:
                    log.warning(f"讀取 GRU cache {code}: {e}")

            await _safe_followup(interaction, embed=embed, ephemeral=True)
            print(f'[PriceModal:on_submit] followup sent', flush=True)
        except Exception as e:
            print(f'[PriceModal:on_submit] top-level err: {type(e).__name__}: {e}', flush=True)
            _handle_http_exception(e, f"股價查詢 {code}")
            await _safe_followup(interaction, f"查詢 {code} 失敗：{e}", ephemeral=True)


class TradeQueryModal(ui.Modal):
    def __init__(self):
        super().__init__(title="查詢交易紀錄")
        self.date_input = ui.TextInput(
            label="日期（MMDD / YYYYMMDD / YYYY-MM-DD）",
            placeholder="例如 0603 或 2026-06-02",
            required=True,
        )
        self.add_item(self.date_input)

    async def on_submit(self, interaction: discord.Interaction):
        raw = self.date_input.value.strip().replace('-', '')
        if not raw.isdigit():
            await interaction.response.send_message("日期格式錯誤", ephemeral=True)
            return
        if len(raw) == 4:
            raw = datetime.now().strftime('%Y') + raw
        if len(raw) != 8:
            await interaction.response.send_message("日期格式錯誤，請輸入 MMDD / YYYYMMDD / YYYY-MM-DD", ephemeral=True)
            return
        fmt_dash = f'{raw[:4]}-{raw[4:6]}-{raw[6:]}'
        fmt_flat = raw

        csv_path = None
        for fname in [f'logs/trades_{fmt_dash}.csv', f'logs/trades_{fmt_flat}.csv']:
            if os.path.exists(fname):
                csv_path = fname
                break

        if not csv_path:
            await interaction.response.send_message(f"❌ {fmt_dash} 無交易紀錄", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            embed = discord.Embed(title=f"交易紀錄 {fmt_dash}", color=0x3498db)

            has_status = 'status' in df.columns
            state_pending_buys = {}
            state_pending_sells = {}
            if not has_status:
                state_path = PORTFOLIO_STATE_FILE
                if os.path.exists(state_path):
                    with open(state_path) as f:
                        s = json.load(f)
                    state_pending_buys = s.get('pending_buys', {})
                    state_pending_sells = s.get('pending_sells', {})

            def _tag(row):
                code = str(row['code']).zfill(4)
                if has_status:
                    st = row['status']
                    if st == 'confirmed':
                        return f"✅ {code}"
                    elif st == 'refunded':
                        return f"❌ {code}"
                    elif st == 'pending':
                        return f"⏳ {code}"
                    return f"{code}"
                if row['type'] == 'BUY' and code in state_pending_buys:
                    return f"⏳ {code}"
                if row['type'] == 'SELL' and code in state_pending_sells:
                    return f"⏳ {code}"
                return f"✅ {code}"

            buys = df[df['type'] == 'BUY']
            sells = df[df['type'] == 'SELL']

            if len(buys) > 0:
                lines = []
                buy_total = 0
                for _, r in buys.iterrows():
                    cost = r['lots'] * 1000 * r['price']
                    buy_total += cost
                    lines.append(f"{_tag(r)}  {r['lots']}張 @${r['price']:.0f}")
                text = '\n'.join(lines)
                if len(text) > 1000:
                    text = text[:1000] + '\n...'
                embed.add_field(name=f"買進 ({len(buys)})", value=text, inline=False)
                embed.add_field(name="買進總額", value=f"${buy_total:,.0f}", inline=True)

            if len(sells) > 0:
                lines = []
                sell_total = 0
                for _, r in sells.iterrows():
                    proceeds = r['lots'] * 1000 * r['price']
                    sell_total += proceeds
                    lines.append(f"{_tag(r)}  {r['lots']}張 @${r['price']:.0f}")
                text = '\n'.join(lines)
                if len(text) > 1000:
                    text = text[:1000] + '\n...'
                embed.add_field(name=f"賣出 ({len(sells)})", value=text, inline=False)
                embed.add_field(name="賣出總額", value=f"${sell_total:,.0f}", inline=True)

            embed.add_field(name="總筆數", value=f"{len(df)} 筆", inline=True)

            await _safe_followup(interaction, embed=embed, ephemeral=True)
        except Exception as e:
            _handle_http_exception(e, "交易查詢")
            await _safe_followup(interaction, f"讀取失敗：{e}", ephemeral=True)


class TradeProposalView(ui.View):
    def __init__(self, bot, proposal):
        super().__init__(timeout=None)
        self.bot = bot
        self.proposal = proposal

    @discord.ui.button(label="確認，執行", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=False)
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)

        try:
            acct, pw = _utils.get_credentials()
            trader = LiveTrader(account=acct, password=pw)
            trader._load_portfolio_state()
            trades = await asyncio.to_thread(trader.execute_proposal, self.proposal, False)

            embed = discord.Embed(
                title=f"交易已執行 {self.proposal.date}",
                description=f"已送出 {trades} 筆訂單（買 {len(self.proposal.buys)} / 賣 {len(self.proposal.sells)}）",
                color=0x00ff00,
            )
            embed.add_field(name="剩餘現金", value=f"${trader.cash:,.0f}", inline=True)
            embed.add_field(name="累計成交", value=str(trader.confirmed_trades), inline=True)
        except Exception as e:
            log.error(f"執行錯誤: {e}", exc_info=True)
            embed = discord.Embed(
                title="執行失敗",
                description=str(e),
                color=0xff0000,
            )

        await interaction.message.edit(embed=embed, view=None)
        self.bot.pending_proposal = None

    @discord.ui.button(label="調整", style=discord.ButtonStyle.secondary)
    async def adjust(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = AdjustModal(self.bot, self.proposal, interaction.message)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="取消", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True

        embed = discord.Embed(
            title="交易已取消",
            description=f"{self.proposal.date} 的交易提案已取消。",
            color=0xff0000,
        )
        await interaction.response.edit_message(embed=embed, view=None)
        self.bot.pending_proposal = None


def _get_yesterday_close(code):
    cache_path = os.path.join('data', 'api_cache', f'{code}.pkl')
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            import pickle
            df = pickle.load(f)
        if len(df) == 0:
            return None
        return float(df.iloc[-1]['close'])
    except Exception:
        return None


def _is_market_hours():
    """判斷是否在交易時間 09:00 ~ 13:30（週一到週五）"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1330


def _get_realtime_batch(codes):
    """批次抓即時股價，回傳 {code: {close, open, high, low}}"""
    if not HAS_TWSTOCK or not codes:
        return {}
    result = {}
    batch_size = 40
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        try:
            data = twstock.realtime.get(batch)
            if not data.get('success', False):
                continue
            for code, v in data.items():
                if code == 'success':
                    continue
                rt = v.get('realtime', {})

                def _to_float(val):
                    if val is None or val == '' or val == '-':
                        return None
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None

                o = _to_float(rt.get('open') or rt.get('o'))
                h = _to_float(rt.get('high') or rt.get('h'))
                lo = _to_float(rt.get('low') or rt.get('l'))
                # latest_trade_price → best_bid → open fallback
                c = _to_float(rt.get('latest_trade_price') or rt.get('z'))
                if c is None or c <= 0:
                    bids = rt.get('best_bid_price', [])
                    if bids and bids[0] and bids[0] != '-':
                        c = _to_float(bids[0])
                if c is None or c <= 0:
                    c = o
                if c and c > 0:
                    result[code] = {
                        'close': c,
                        'open': o if o and o > 0 else c,
                        'high': h if h and h > 0 else c,
                        'low': lo if lo and lo > 0 else c,
                    }
        except Exception as e:
            log.warning(f"twstock batch error: {e}")
    return result


class ControlPanelView(ui.View):
    def __init__(self, bot=None):
        super().__init__(timeout=None)
        # bot 參數保留向後相容；callback 用 interaction.client 拿
        self.bot = bot

    @discord.ui.button(label="狀態", style=discord.ButtonStyle.primary, custom_id="cp:status")
    async def status(self, interaction: discord.Interaction, button: discord.ui.Button):
        print(f'[CB:status] enter', flush=True)
        try:
            await interaction.response.defer(ephemeral=True)
            print(f'[CB:status] deferred', flush=True)
        except Exception as e:
            print(f'[CB:status] defer fail: {e}', flush=True)
            return
        try:
            state_path = PORTFOLIO_STATE_FILE
            cash = INITIAL_CAPITAL
            confirmed = 0
            pending_b = 0
            pending_s = 0
            cost_basis_map = {}
            if os.path.exists(state_path):
                with open(state_path) as f:
                    s = json.load(f)
                    cash = s.get('cash', INITIAL_CAPITAL)
                    confirmed = s.get('confirmed_trades', 0)
                    pending_b = len(s.get('pending_buys', {}))
                    pending_s = len(s.get('pending_sells', {}))
                    cost_basis_map = s.get('cost_basis', {})

            embed = discord.Embed(title="投資組合", color=0x3498db)
            embed.add_field(name="現金", value=f"${cash:,.0f}", inline=True)
            embed.add_field(name="累計成交", value=str(confirmed), inline=True)
            embed.add_field(name="待成交買/賣", value=f"{pending_b} / {pending_s}", inline=True)

            total_cost_value = 0.0
            if cost_basis_map:
                for code, shares in s.get('holdings', {}).items():
                    cost_per_lot = cost_basis_map.get(code, 0)
                    total_cost_value += shares * 1000 * cost_per_lot

            if cost_basis_map:
                lines = []
                for code, shares in s.get('holdings', {}).items():
                    cost_per_lot = cost_basis_map.get(code, 0)
                    lines.append(f"**{code}**  {shares}張  成本${cost_per_lot:.2f}" if cost_per_lot else f"**{code}**  {shares}張")
                text = "\n".join(lines)
                if len(text) > 1000:
                    text = text[:1000] + "\n..."
                embed.add_field(name=f"持股 ({len(s.get('holdings',{}))})", value=text, inline=False)
            else:
                embed.add_field(name="持股", value="無持股", inline=False)

            embed.add_field(name="股票總成本", value=f"${total_cost_value:,.0f}", inline=True)

            embed.set_footer(text=f"📈 僅供參考 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            await _safe_followup(interaction, embed=embed, ephemeral=True)
            print(f'[CB:status] followup sent', flush=True)
        except Exception as e:
            print(f'[CB:status] err: {type(e).__name__}: {e}', flush=True)
            _handle_http_exception(e, "狀態查詢")
            await _safe_followup(interaction, f"錯誤：{e}", ephemeral=True)

    @discord.ui.button(label="股價查詢", style=discord.ButtonStyle.secondary, custom_id="cp:price")
    async def price(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = PriceModal()
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="盤後更新", style=discord.ButtonStyle.secondary, custom_id="cp:daily")
    async def daily(self, interaction: discord.Interaction, button: discord.ui.Button):
        print(f'[CB:daily] enter', flush=True)
        await interaction.response.defer(ephemeral=True)
        await _safe_followup(interaction, "正在執行盤後更新（抓取所有股價 → GRU 預測 → 對帳）...", ephemeral=True)
        try:
            fetch_ok, fetch_count = await asyncio.to_thread(_fetch_all_prices)
            if fetch_count == 0:
                await _safe_followup(interaction, "所有股票資料已是最新，無需更新。", ephemeral=True)
            else:
                await _safe_followup(interaction, f"✅ 股價更新完成：{fetch_count} 檔", ephemeral=True)

            gru_ok = await asyncio.to_thread(phase_gru)
            if gru_ok:
                gru_count = len(_utils.get_stocks_needing_gru_update())
                done = sum(1 for _ in os.listdir(GRU_CACHE_DIR) if _.endswith('.npy'))
                await _safe_followup(interaction, f"✅ GRU 預測完成：{done} 檔已更新", ephemeral=True)

            result = await asyncio.to_thread(reconcile_state)
            if result:
                embed = discord.Embed(title="對帳結果", color=0x00ff00)
                embed.add_field(name="現金", value=f"${result['cash']:,.0f}", inline=True)
                embed.add_field(name="持股", value=f"{result['num_holdings']} 檔", inline=True)
                embed.add_field(name="累計成交", value=str(result['confirmed_trades']), inline=True)
                details = []
                if result['buy_confirmed']:
                    details.append(f"買進成交：{', '.join(result['buy_confirmed'])}")
                if result['buy_refunded']:
                    details.append(f"買進未成交（退款）：{len(result['buy_refunded'])} 筆")
                if result['sell_confirmed']:
                    details.append(f"賣出成交：{', '.join(result['sell_confirmed'])}")
                if result['sell_still_pending']:
                    details.append(f"賣出取消：{', '.join(result['sell_still_pending'])}")
                if details:
                    embed.add_field(name="明細", value="\n".join(details), inline=False)
                await _safe_followup(interaction, embed=embed, ephemeral=True)
        except Exception as e:
            _handle_http_exception(e, "盤後更新")
            await _safe_followup(interaction, f"盤後更新失敗：{e}", ephemeral=True)

    @discord.ui.button(label="查詢交易", style=discord.ButtonStyle.secondary, custom_id="cp:query")
    async def query_trades(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = TradeQueryModal()
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="GRU 排行", style=discord.ButtonStyle.secondary, custom_id="cp:gru_rank")
    async def gru_rank(self, interaction: discord.Interaction, button: discord.ui.Button):
        print(f'[CB:gru_rank] enter', flush=True)
        await interaction.response.defer(ephemeral=True)
        try:
            result = await asyncio.to_thread(_compute_gru_rankings)
            if not result:
                await _safe_followup(interaction, "GRU 快取中無資料。請先執行盤後更新。", ephemeral=True)
                return

            view = GRURankView(result)
            embed = view.build_embed(horizon_idx=0)
            await _safe_followup(interaction, embed=embed, view=view, ephemeral=True)
        except Exception as e:
            _handle_http_exception(e, "GRU 排行")
            await _safe_followup(interaction, f"錯誤：{e}", ephemeral=True)

    @discord.ui.button(label="手動買入", style=discord.ButtonStyle.success, custom_id="cp:buy")
    async def manual_buy(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = ManualBuyModal(interaction.client)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="手動賣出", style=discord.ButtonStyle.danger, custom_id="cp:sell")
    async def manual_sell(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = ManualSellModal(interaction.client)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="PPO 提案", style=discord.ButtonStyle.primary, custom_id="cp:trade")
    async def trade(self, interaction: discord.Interaction, button: discord.ui.Button):
        bot = interaction.client
        if getattr(bot, 'pending_proposal', None) is not None:
            await interaction.response.send_message(
                "已有待確認的交易提案，請先到該提案按取消。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        await bot.do_trade_flow(interaction)


def _fetch_all_prices():
    """從 Kaggle dataset 更新所有股價（無 API 呼叫、無超時）。"""
    saved, total = _utils.fetch_kaggle_update()
    return True, saved


class GRURankView(ui.View):
    """GRU 排行 View：含 horizon 下拉選單。"""
    def __init__(self, rankings, top_n=20, timeout=300):
        super().__init__(timeout=timeout)
        self.rankings = rankings
        self.top_n = top_n
        self.current_horizon_idx = 0
        options = [
            discord.SelectOption(
                label=f"t+{h}",
                value=str(i),
                description=f"預測 {h} 日後報酬率",
                default=(i == 0),
            )
            for i, h in enumerate(PRED_HORIZONS)
        ]
        self.horizon_select = ui.Select(
            placeholder="選擇預測區間 (t+?)",
            options=options,
            custom_id="gru_rank:horizon",
        )
        self.horizon_select.callback = self.on_horizon_change
        self.add_item(self.horizon_select)

    def build_embed(self, horizon_idx):
        horizon = PRED_HORIZONS[horizon_idx]
        sorted_ranks = sorted(
            self.rankings, key=lambda x: -x[1][horizon_idx])[:self.top_n]
        header = f"代碼  t+{horizon}"
        lines = [f"{code}  {preds[horizon_idx]*100:+.2f}%"
                 for code, preds in sorted_ranks]
        embed = discord.Embed(
            title=f"GRU 預測排行（t+{horizon}，前 {len(sorted_ranks)} 名）",
            description=f"共 {len(self.rankings)} 檔股票\n```\n{header}\n" + "\n".join(lines) + "\n```",
            color=0x9b59b6,
        )
        return embed

    async def on_horizon_change(self, interaction: discord.Interaction):
        self.current_horizon_idx = int(self.horizon_select.values[0])
        # 更新 default
        for opt in self.horizon_select.options:
            opt.default = (opt.value == self.horizon_select.values[0])
        embed = self.build_embed(self.current_horizon_idx)
        await interaction.response.edit_message(embed=embed, view=self)


def _compute_gru_rankings(force=False):
    """計算（或讀 cache）GRU 預測排行。

    Cache 寫到 logs/.gru_rank_cache.json，5 分鐘內重用。
    第一次計算用 ThreadPoolExecutor 平行載入 (8 workers)。
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache_path = os.path.join('logs', '.gru_rank_cache.json')
    if not force and os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            if time.time() - mtime < 300:
                with open(cache_path, encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass

    if not os.path.isdir(GRU_CACHE_DIR):
        return []
    pred_start = GRU_HIDDEN_SIZE * len(TIME_WINDOWS)
    files = [f for f in os.listdir(GRU_CACHE_DIR) if f.endswith('.npy')]

    def _load_one(fname):
        code = fname[:-4]
        path = os.path.join(GRU_CACHE_DIR, fname)
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == np.object_:
                data = data.item()
            if isinstance(data, dict):
                cache = data['cache'].astype(np.float32)
            else:
                cache = data.astype(np.float32)
            if len(cache) == 0:
                return None
            preds = cache[-1, pred_start:pred_start + N_HORIZONS].tolist()
            return (code, preds)
        except Exception:
            return None

    rankings = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_load_one, files):
            if r is not None:
                rankings.append(r)
    rankings.sort(key=lambda x: -x[1][0])

    try:
        os.makedirs('logs', exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(rankings, f)
    except Exception as e:
        log.warning(f"gru_rank cache write fail: {e}")
    return rankings


class TradeBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(command_prefix='!', intents=intents)
        self.pending_proposal = None
        self.proposal_msg = None
        self.last_trade_date = None
        self.last_post_update_date = None
        fh = logging.FileHandler('logs/discord_bot.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logging.getLogger().addHandler(fh)

    async def on_ready(self):
        print(f'[READY] logged in as {self.user} (ID: {self.user.id})', flush=True)
        log.info(f'已登入為 {self.user} (ID: {self.user.id})')
        # 清掉舊的控制面板（它們的 callback 已經失效），再送新的
        await self._cleanup_old_panels()
        await self._send_control_panel()
        print(f'[READY] control panel sent', flush=True)

    async def on_interaction(self, interaction: discord.Interaction):
        """全域 interaction logger：記錄所有 component 點擊，幫助 debug。"""
        try:
            data = interaction.data or {}
            custom_id = data.get('custom_id', '?')
            user = interaction.user
            print(f'[INTERACTION] type={interaction.type} custom_id={custom_id} user={user} channel={interaction.channel_id}', flush=True)
            log.info(
                f"INTERACTION type={interaction.type} custom_id={custom_id} "
                f"user={user} channel={interaction.channel_id}"
            )
        except Exception as e:
            print(f'[INTERACTION-LOG-ERR] {e}', flush=True)
            log.warning(f"on_interaction log error: {e}")

    async def _cleanup_old_panels(self):
        """刪除 channel 中由本 bot 送的、標題為 'PPO 交易機器人' 的舊訊息，
        避免重啟後使用者按舊按鈕沒有反應。
        """
        channel = self.get_channel(DISCORD_CHANNEL_ID)
        if channel is None:
            return
        try:
            self_user_id = self.user.id if self.user else None
            if self_user_id is None:
                return
            deleted = 0
            async for msg in channel.history(limit=20):
                if msg.author.id != self_user_id:
                    continue
                if msg.embeds and msg.embeds[0].title == "PPO 交易機器人":
                    try:
                        await msg.delete()
                        deleted += 1
                    except discord.HTTPException as e:
                        log.debug(f"  delete old panel failed: {e}")
            if deleted:
                log.info(f"  已刪除 {deleted} 個舊控制面板")
        except discord.HTTPException as e:
            log.warning(f"  _cleanup_old_panels HTTP error: {e}")
        except Exception as e:
            log.warning(f"  _cleanup_old_panels error: {e}")

    async def on_error(self, event, *args, **kwargs):
        log.error(f'Unhandled error in {event}', exc_info=True)
        # 401 token expired — log loudly so operator notices
        import traceback
        tb = traceback.format_exc()
        if '401' in tb or '50027' in tb:
            log.error('🔴 DISCORD WEBHOOK TOKEN 過期！請更新 .env 的 DISCORD_TOKEN 並重啟 bot')

    async def _send_control_panel(self):
        channel = self.get_channel(DISCORD_CHANNEL_ID)
        if channel is None:
            log.error(f'找不到頻道 {DISCORD_CHANNEL_ID}！請檢查 DISCORD_CHANNEL_ID。')
            return
        embed = discord.Embed(
            title="PPO 交易機器人",
            description=(
                "自動化股票交易系統的控制面板。\n\n"
                "**狀態** — 現金、累計成交、待成交、各股成本與昨收投報率\n"
                "**股價查詢** — 查詢各股最新股價 + GRU 預測\n"
                "**盤後更新** — 更新全部股價 → GRU 預測 → 對帳\n"
                "**查詢交易** — 查詢特定日期的交易紀錄\n"
                "**GRU 排行** — 依 GRU t+1 預測報酬率排行\n"
                "**手動買入** — 手動下單買入股票\n"
                "**手動賣出** — 手動下單賣出股票\n"
                "**PPO 提案** — PPO 計算 → 提案 → 確認後下單"
            ),
            color=0x00ff00,
        )
        await _safe_channel_send(channel, embed=embed, view=ControlPanelView(self))
        log.info('控制面板已送出')

    async def setup_hook(self):
        self.check_trade_time.start()
        self.auto_post_market_update.start()
        # 註冊 persistent view（讓重啟後舊 message 上的按鈕仍有效）
        self.add_view(ControlPanelView())
        print(f'[SETUP] added ControlPanelView as persistent', flush=True)
        await self.tree.sync()
        print(f'[SETUP] tree synced', flush=True)

    @tasks.loop(minutes=1)
    async def check_trade_time(self):
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        if now.weekday() < 5 and now.hour == 10 and now.minute == 0:
            if self.last_trade_date != today:
                log.info('10:00 自動觸發交易流程')
                channel = self.get_channel(DISCORD_CHANNEL_ID)
                if channel:
                    await _safe_channel_send(channel, "10:00 — 正在計算今日交易提案...")
                await self.do_trade_flow()
                self.last_trade_date = today

    @tasks.loop(minutes=1)
    async def auto_post_market_update(self):
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        if now.weekday() < 5 and now.hour == 16 and now.minute == 30:
            if self.last_post_update_date != today:
                self.last_post_update_date = today
                channel = self.get_channel(DISCORD_CHANNEL_ID)
                if channel:
                    await _safe_channel_send(channel, "⏰ 16:30 — 自動盤後更新開始（Kaggle 股價 → GRU → 對帳）")
                    try:
                        saved, total = await asyncio.to_thread(_utils.fetch_kaggle_update)
                        await _safe_channel_send(channel, f"✅ 股價更新完成：{saved}/{total} 檔（Kaggle）")

                        await _safe_channel_send(channel, "🧠 執行 GRU 預測...")
                        gru_ok = await asyncio.to_thread(phase_gru)
                        if gru_ok:
                            done = len([f for f in os.listdir(GRU_CACHE_DIR) if f.endswith('.npy')])
                            await _safe_channel_send(channel, f"✅ GRU 預測完成：{done} 檔")

                        result = await asyncio.to_thread(reconcile_state)
                        if result:
                            embed = discord.Embed(title="📋 對帳結果", color=0x00ff00)
                            embed.add_field(name="現金", value=f"${result['cash']:,.0f}", inline=True)
                            embed.add_field(name="持股", value=f"{result['num_holdings']} 檔", inline=True)
                            embed.add_field(name="累計成交", value=str(result['confirmed_trades']), inline=True)
                            details = []
                            if result['buy_confirmed']:
                                details.append(f"買進成交：{', '.join(result['buy_confirmed'])}")
                            if result['buy_refunded']:
                                details.append(f"買進未成交退款：{len(result['buy_refunded'])} 筆")
                            if result['sell_confirmed']:
                                details.append(f"賣出成交：{', '.join(result['sell_confirmed'])}")
                            if result['sell_still_pending']:
                                details.append(f"賣出取消：{', '.join(result['sell_still_pending'])}")
                            if details:
                                embed.add_field(name="明細", value="\n".join(details), inline=False)
                            await _safe_channel_send(channel, embed=embed)
                        await _safe_channel_send(channel, "✅ 盤後更新完成")
                    except discord.errors.HTTPException as e:
                        if e.status == 401 or e.code == 50027:
                            log.error("🔴 channel.send 401 — DISCORD_TOKEN 過期，請更新")
                        else:
                            log.error(f"channel.send HTTPException: {e}", exc_info=True)
                    except Exception as e:
                        log.error(f"盤後更新錯誤: {e}", exc_info=True)
                        await _safe_channel_send(channel, f"❌ 盤後更新失敗：{e}")

    async def do_trade_flow(self, interaction=None):
        try:
            acct, pw = _utils.get_credentials()
            trader = LiveTrader(account=acct, password=pw)
            trader._load_portfolio_state()
            proposal = await asyncio.to_thread(trader.compute_proposal)

            if proposal is None:
                msg = "今日無交易提案（尚有未成交訂單或無資料）。"
                if interaction:
                    await _safe_followup(interaction, msg, ephemeral=True)
                else:
                    channel = self.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await _safe_channel_send(channel, msg)
                return

            embed = build_proposal_embed(proposal)
            view = TradeProposalView(self, proposal)

            if interaction:
                await _safe_followup(interaction, embed=embed, view=view)
            else:
                channel = self.get_channel(DISCORD_CHANNEL_ID)
                if channel:
                    self.proposal_msg = await _safe_channel_send(channel, embed=embed, view=view)

            self.pending_proposal = proposal

        except Exception as e:
            _handle_http_exception(e, "交易流程")
            err = f"交易流程失敗：{e}"
            if interaction:
                await _safe_followup(interaction, err, ephemeral=True)
            else:
                channel = self.get_channel(DISCORD_CHANNEL_ID)
                if channel:
                    await _safe_channel_send(channel, err)


if __name__ == '__main__':
    if not DISCORD_TOKEN:
        log.error('DISCORD_TOKEN 未設定！請編輯 .env 或設定環境變數。')
        sys.exit(1)
    if not DISCORD_CHANNEL_ID:
        log.error('DISCORD_CHANNEL_ID 未設定！請編輯 .env 或設定環境變數。')
        sys.exit(1)

    # Add file logging so crashes are captured
    fh = logging.FileHandler('logs/discord_bot.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(fh)

    bot = TradeBot()
    bot.run(DISCORD_TOKEN, log_handler=None)
