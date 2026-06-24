import time
import pandas as pd

from .utils import month_starts, roc_to_ad, safe_get_json, clean_numeric

# TWSE
def get_twse_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    all_df = []

    for month_start in month_starts(start_date, end_date):
        params = {
            "response": "json",
            "date": pd.Timestamp(month_start).strftime("%Y%m%d"),
            "stockNo": stock_code,
        }

        raw = safe_get_json(url, params=params)

        if raw.get("stat") != "OK" or not raw.get("data"):
            time.sleep(2.0)
            continue

        df = pd.DataFrame(raw["data"], columns=raw["fields"])

        df = df.rename(columns={
            "日期": "date",
            "成交股數": "capacity",
            "成交金額": "turnover",
            "開盤價": "open",
            "最高價": "high",
            "最低價": "low",
            "收盤價": "close",
            "漲跌價差": "change",
            "成交筆數": "transaction_volume",
        })

        df["date"] = df["date"].apply(roc_to_ad)

        numeric_cols = [
            "capacity", "turnover", "open", "high",
            "low", "close", "change", "transaction_volume"
        ]
        for col in numeric_cols:
            df[col] = clean_numeric(df[col])

        df["stock_code_id"] = stock_code
        df["market"] = "TWSE"
        df["close_is_proxy"] = False

        all_df.append(df)
        time.sleep(2.0)

    if not all_df:
        return pd.DataFrame(columns=[
            "date", "stock_code_id", "market",
            "capacity", "turnover", "open", "high", "low", "close", "change",
            "transaction_volume", "close_is_proxy"
        ])

    result = pd.concat(all_df, ignore_index=True)

    result = result[
        (result["date"] >= pd.Timestamp(start_date)) &
        (result["date"] <= pd.Timestamp(end_date))
    ].copy()

    result = result.sort_values("date").reset_index(drop=True)

    return result[[
        "date", "stock_code_id", "market",
        "capacity", "turnover", "open", "high", "low", "close", "change",
        "transaction_volume", "close_is_proxy"
    ]]
    
# TPEX
def get_tpex_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
    all_df = []

    for month_start in month_starts(start_date, end_date):
        params = {
            "code": stock_code,
            "date": month_start,
            "id": "",
            "response": "json",
        }

        raw = safe_get_json(url, params=params)

        if raw.get("stat", "").lower() != "ok":
            time.sleep(2.0)
            continue

        tables = raw.get("tables", [])
        if not tables:
            time.sleep(2.0)
            continue

        table = tables[0]
        data = table.get("data", [])
        fields = table.get("fields", [])

        if not data or not fields:
            time.sleep(2.0)
            continue

        df = pd.DataFrame(data, columns=fields)

        df = df.rename(columns={
            "日 期": "date",
            "成交張數": "capacity",
            "成交仟元": "turnover",
            "開盤": "open",
            "最高": "high",
            "最低": "low",
            "收盤": "close",
            "漲跌": "change",
            "筆數": "transaction_volume",
        })

        df["date"] = df["date"].apply(roc_to_ad)

        numeric_cols = [
            "capacity", "turnover", "open", "high",
            "low", "close", "change", "transaction_volume"
        ]
        for col in numeric_cols:
            df[col] = clean_numeric(df[col])

        # TPEX 的成交張數、成交仟元換算成股數與元
        df["capacity"] = df["capacity"] * 1000
        df["turnover"] = df["turnover"] * 1000

        df["stock_code_id"] = stock_code
        df["market"] = "TPEX"
        df["close_is_proxy"] = False

        all_df.append(df)
        time.sleep(2.0)

    if not all_df:
        return pd.DataFrame(columns=[
            "date", "stock_code_id", "market",
            "capacity", "turnover", "open", "high", "low", "close", "change",
            "transaction_volume", "close_is_proxy"
        ])

    result = pd.concat(all_df, ignore_index=True)

    result = result[
        (result["date"] >= pd.Timestamp(start_date)) &
        (result["date"] <= pd.Timestamp(end_date))
    ].copy()

    result = result.sort_values("date").reset_index(drop=True)

    return result[[
        "date", "stock_code_id", "market",
        "capacity", "turnover", "open", "high", "low", "close", "change",
        "transaction_volume", "close_is_proxy"
    ]]
    
# ESB
def get_esb_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://www.tpex.org.tw/www/zh-tw/emerging/historical"
    all_df = []

    for month_start in month_starts(start_date, end_date):
        params = {
            "type": "Monthly",
            "date": month_start,
            "code": stock_code,
            "id": "",
            "response": "json",
        }

        raw = safe_get_json(url, params=params)

        if raw.get("stat", "").lower() != "ok":
            time.sleep(2.0)
            continue

        tables = raw.get("tables", [])
        if not tables:
            time.sleep(2.0)
            continue

        table = tables[0]
        data = table.get("data", [])

        if not data:
            time.sleep(2.0)
            continue

        # ESB 固定 13 欄，避免重名 fields 造成 DataFrame 麻煩，直接指定欄名
        columns = [
            "date",
            "capacity_1", "turnover_1", "high_1", "low_1", "avg_1", "txn_1",
            "capacity_2", "turnover_2", "high_2", "low_2", "avg_2", "txn_2",
        ]

        df = pd.DataFrame(data, columns=columns)

        df["date"] = df["date"].apply(roc_to_ad)

        numeric_cols = [
            "capacity_1", "turnover_1", "high_1", "low_1", "avg_1", "txn_1",
            "capacity_2", "turnover_2", "high_2", "low_2", "avg_2", "txn_2",
        ]
        for col in numeric_cols:
            df[col] = clean_numeric(df[col]).fillna(0)

        df["capacity"] = df["capacity_1"] + df["capacity_2"]
        df["turnover"] = df["turnover_1"] + df["turnover_2"]
        df["transaction_volume"] = df["txn_1"] + df["txn_2"]

        df["high"] = df[["high_1", "high_2"]].replace(0, pd.NA).max(axis=1, skipna=True)
        df["low"] = df[["low_1", "low_2"]].replace(0, pd.NA).min(axis=1, skipna=True)

        # 以加權平均成交價作為 close 代理值
        df["close"] = df["turnover"] / df["capacity"]
        df.loc[df["capacity"] == 0, "close"] = pd.NA

        # ESB 無標準 open
        df["open"] = pd.NA

        df["stock_code_id"] = stock_code
        df["market"] = "ESB"
        df["close_is_proxy"] = True

        all_df.append(df)
        time.sleep(2.0)

    if not all_df:
        return pd.DataFrame(columns=[
            "date", "stock_code_id", "market",
            "capacity", "turnover", "open", "high", "low", "close", "change",
            "transaction_volume", "close_is_proxy"
        ])

    result = pd.concat(all_df, ignore_index=True)

    result = result[
        (result["date"] >= pd.Timestamp(start_date)) &
        (result["date"] <= pd.Timestamp(end_date))
    ].copy()

    result = result.sort_values("date").reset_index(drop=True)

    # 用代理 close 算日變動
    result["change"] = result["close"].diff()

    return result[[
        "date", "stock_code_id", "market",
        "capacity", "turnover", "open", "high", "low", "close", "change",
        "transaction_volume", "close_is_proxy"
    ]]