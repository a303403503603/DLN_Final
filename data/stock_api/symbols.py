import json
from pathlib import Path
import requests


def _get_symbol_map_path() -> Path:
    return Path(__file__).resolve().parent / "stock_symbol_map.json"


def load_symbol_map() -> dict:
    path = _get_symbol_map_path()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("stock_symbol_map.json 格式錯誤，應為 dict")

    return data


def get_stock_info(stock_code: str) -> dict:
    symbol_map = load_symbol_map()
    stock_code = str(stock_code)

    info = symbol_map.get(stock_code)
    if info is None:
        raise ValueError(f"找不到股票代號: {stock_code}")

    if not isinstance(info, dict):
        raise ValueError(f"股票代號 {stock_code} 的資料格式錯誤")

    return info


def get_raw_market(stock_code: str) -> str:
    """
    回傳 map 原始市場別：
    ETF / TWSE / OTC / ESB
    """
    info = requests.get(
        f"https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/stock_type?stock_code={stock_code}"
    ).json()

    market = info.get("type")

    if market not in {"ETF", "TWSE", "OTC", "ESB"}:
        raise ValueError(f"未知市場別: {market}")

    return market


def normalize_market(raw_market: str) -> str:
    """
    將原始市場別轉成內部抓資料用的市場別
    ETF  -> TWSE
    TWSE -> TWSE
    OTC  -> TPEX
    ESB  -> ESB
    """
    market_map = {
        "ETF": "TWSE",
        "TWSE": "TWSE",
        "OTC": "TPEX",
        "ESB": "ESB",
    }

    if raw_market not in market_map:
        raise ValueError(f"無法正規化市場別: {raw_market}")

    return market_map[raw_market]


def get_stock_market(stock_code: str) -> str:
    """
    回傳內部抓資料用的市場別：
    TWSE / TPEX / ESB
    """
    raw_market = get_raw_market(stock_code)
    return normalize_market(raw_market)