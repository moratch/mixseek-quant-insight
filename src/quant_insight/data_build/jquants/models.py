"""J-Quants API models and enumerations."""

from enum import Enum


class JQuantsPlan(str, Enum):
    """J-Quants APIプラン.

    各プランでRate Limitと取得可能なデータ範囲が異なる。
    """

    FREE = "free"
    LIGHT = "light"
    STANDARD = "standard"
    PREMIUM = "premium"


class JQuantsUniverse(str, Enum):
    """対象銘柄ユニバース.

    市場コード（Mkt）に対応:
    - prime: プライム市場 (Mkt='0111')
    - standard: スタンダード市場 (Mkt='0112')
    - growth: グロース市場 (Mkt='0113')
    - all: 全銘柄
    """

    PRIME = "prime"
    STANDARD = "standard"
    GROWTH = "growth"
    ALL = "all"


# 市場コードマッピング
UNIVERSE_TO_MARKET_CODE: dict[JQuantsUniverse, str | None] = {
    JQuantsUniverse.PRIME: "0111",
    JQuantsUniverse.STANDARD: "0112",
    JQuantsUniverse.GROWTH: "0113",
    JQuantsUniverse.ALL: None,  # フィルタなし
}


# OHLCVカラムマッピング (J-Quants → 標準スキーマ + 追加カラム)
OHLCV_COLUMN_MAPPING: dict[str, str] = {
    "Date": "datetime",
    "Code": "symbol",
    # 調整後価格 (標準スキーマ)
    "AdjO": "open",
    "AdjH": "high",
    "AdjL": "low",
    "AdjC": "close",
    "AdjVo": "volume",
    "AdjFactor": "adj_factor",
    # 生価格
    "O": "raw_open",
    "H": "raw_high",
    "L": "raw_low",
    "C": "raw_close",
    "Vo": "raw_volume",
    # 値幅制限
    "UL": "upper_limit",
    "LL": "lower_limit",
    # 売買代金
    "Va": "turnover",
    # 前場
    "MO": "morning_open",
    "MH": "morning_high",
    "ML": "morning_low",
    "MC": "morning_close",
    "MUL": "morning_upper_limit",
    "MLL": "morning_lower_limit",
    "MVo": "morning_volume",
    "MVa": "morning_turnover",
    "MAdjO": "morning_adj_open",
    "MAdjH": "morning_adj_high",
    "MAdjL": "morning_adj_low",
    "MAdjC": "morning_adj_close",
    "MAdjVo": "morning_adj_volume",
    # 後場
    "AO": "afternoon_open",
    "AH": "afternoon_high",
    "AL": "afternoon_low",
    "AC": "afternoon_close",
    "AUL": "afternoon_upper_limit",
    "ALL": "afternoon_lower_limit",
    "AVo": "afternoon_volume",
    "AVa": "afternoon_turnover",
    "AAdjO": "afternoon_adj_open",
    "AAdjH": "afternoon_adj_high",
    "AAdjL": "afternoon_adj_low",
    "AAdjC": "afternoon_adj_close",
    "AAdjVo": "afternoon_adj_volume",
}

# Masterカラムマッピング
MASTER_COLUMN_MAPPING: dict[str, str] = {
    "Date": "datetime",
    "Code": "symbol",
    "CoName": "company_name",
    "CoNameEn": "company_name_en",
    "S17": "sector17_code",
    "S17Nm": "sector17_name",
    "S33": "sector33_code",
    "S33Nm": "sector33_name",
    "ScaleCat": "scale_category",
    "Mkt": "market_code",
    "MktNm": "market_name",
    "Mrgn": "margin_code",
    "MrgnNm": "margin_name",
}
