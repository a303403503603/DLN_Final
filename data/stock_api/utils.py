import time
import random
import requests
import pandas as pd


def month_starts(start_date: str, end_date: str) -> list[str]:
    """
    回傳起始日到結束日之間，每個月的月初字串 YYYY/MM/DD
    例如: ["2026/01/01", "2026/02/01", "2026/03/01"]
    """
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)
    months = pd.date_range(start=start, end=end, freq="MS")
    return [d.strftime("%Y/%m/%d") for d in months]


def roc_to_ad(date_str: str) -> pd.Timestamp:
    """
    將民國日期 115/03/02 轉成西元 Timestamp(2026-03-02)
    """
    y, m, d = date_str.split("/")
    ad_year = int(y) + 1911
    return pd.Timestamp(f"{ad_year}-{m}-{d}")


def safe_get_json(url: str, params: dict | None = None, timeout: int = 30, max_retries: int = 5) -> dict:
    """
    安全取得 JSON，內建簡單重試
    """
    session = requests.Session()

    for i in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if i == max_retries - 1:
                raise
            sleep_sec = (2 ** i) + random.uniform(0.3, 1.0)
            time.sleep(sleep_sec)


def clean_numeric(series: pd.Series) -> pd.Series:
    """
    將含逗號、空白、特殊字元的欄位轉數字
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("X", "", regex=False)
        .str.replace("除權", "", regex=False)
        .str.replace("除息", "", regex=False)
        .str.replace("----", "", regex=False)
        .str.strip(),
        errors="coerce"
    )