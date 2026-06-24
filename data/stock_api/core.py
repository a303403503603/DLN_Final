from .symbols import get_stock_market
from .fetchers import (
    get_twse_stock_data,
    get_tpex_stock_data,
    get_esb_stock_data,
)
import requests
import logging

_log = logging.getLogger('stock_api')

def to_legacy_schema(df):
    legacy_columns = [
        "date",
        "capacity",
        "turnover",
        "high",
        "low",
        "close",
        "change",
        "transaction_volume",
        "stock_code_id",
        "open",
    ]
    return df[legacy_columns].copy()


def get_all_stock_list():
    data = requests.get(
        "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/stock_list",
        timeout=15,
    ).json()

    stock_codes_list = list(data.keys())

    return stock_codes_list


def get_taiwan_stock_data(stock_code: str, start_date: str, end_date: str):
    """取得股票資訊"""
    market = get_stock_market(stock_code)

    if market == "TWSE":
        return to_legacy_schema(get_twse_stock_data(stock_code, start_date, end_date))
    elif market == "TPEX":
        return to_legacy_schema(get_tpex_stock_data(stock_code, start_date, end_date))
    elif market == "ESB":
        return to_legacy_schema(get_esb_stock_data(stock_code, start_date, end_date))
    else:
        raise ValueError(f"不支援的市場別: {market}")


BASE_URL = "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api"

def Get_User_Stocks(account: str, password: str):
    """取得持有股票"""
    data = {'account': account,
            'password': password
            }
    response = requests.post(f"{BASE_URL}/get_user_stocks", data=data, timeout=15)
    result = response.json()
    if(result['result'] == 'success'):
        return result['data']
    return dict([])


def get_account_info(account: str, password: str):
    """取得完整帳戶資訊（含可用餘額等）"""
    data = {'account': account, 'password': password}
    response = requests.post(f"{BASE_URL}/get_user_stocks", data=data, timeout=15)
    result = response.json()
    return result if isinstance(result, dict) else {'result': 'fail'}

# 預約購入股票
def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    """預約購入股票

    Returns:
        (ok: bool, msg: str) 統一介面，跟 Sell_Stock 一致。
    """
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}

    response = requests.post(f"{BASE_URL}/buy", data=data, timeout=15)
    try:
        result = response.json()
    except Exception as e:
        _log.error(f"Buy_Stock {stock_code}: JSON decode error: {e}")
        return False, f"json decode error: {e}"
    ok = result.get('result') == 'success'
    msg = result.get('status', 'unknown')
    _log.info(f"Buy_Stock {stock_code} {stock_shares}張 @{stock_price}: ok={ok}, status={msg}")
    return ok, msg

# 預約售出股票
def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    """預約售出股票，回傳 (成功與否, 狀態訊息)"""
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}

    response = requests.post(f"{BASE_URL}/sell", data=data, timeout=15)
    try:
        result = response.json()
    except Exception as e:
        _log.error(f"Sell_Stock {stock_code}: JSON decode error: {e}")
        return False, f"json decode error: {e}"
    ok = result.get('result') == 'success'
    msg = result.get('status', 'unknown')
    _log.info(f"Sell_Stock {stock_code} {stock_shares}張 @{stock_price}: ok={ok}, status={msg}")
    return ok, msg