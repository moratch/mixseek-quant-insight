"""BOJ macro data models and column mappings."""

# BOJ Parquet long-format column names → standard schema mapping
BOJ_PARQUET_COLUMNS = {
    "date": "日付",
    "series_code": "系列コード",
    "db": "データベース",
    "value": "値",
    "unit": "単位",
    "frequency": "頻度",
    "last_update": "最終更新",
    "fetch_ts": "取得タイムスタンプ",
}

# Phase 1 series code → human-readable name mapping
SERIES_NAME_MAPPING: dict[str, str] = {
    # FM01 - Call Rate
    "FM01_STRDCLUCON": "call_rate_avg",
    "FM01_STRDCLUCONH": "call_rate_high",
    "FM01_STRDCLUCONL": "call_rate_low",
    "FM01_STRDCLUCV": "call_rate_volume",
    # FM08 - FX
    "FM08_FXERD01": "usdjpy_spot_9am",
    "FM08_FXERD02": "usdjpy_high",
    "FM08_FXERD03": "usdjpy_low",
    "FM08_FXERD04": "usdjpy_spot_5pm",
    "FM08_FXERD05": "usdjpy_central",
    "FM08_FXERD06": "usdjpy_turnover_spot",
    "FM08_FXERD07": "usdjpy_turnover_swap",
    "FM08_FXERD31": "eurusd_9am",
    "FM08_FXERD32": "eurusd_high",
    "FM08_FXERD33": "eurusd_low",
    "FM08_FXERD34": "eurusd_5pm",
    # IR01 - Discount Rate
    "IR01_MADR1Z@D": "discount_rate",
    # MD01 - Monetary Base
    "MD01_MABS1AN11": "monetary_base_total",
    "MD01_MABS1AN11@": "monetary_base_yoy",
    "MD01_MABS2AN116": "banknotes_outstanding",
    "MD01_MABS1AN113": "current_account_balances",
    # PR01 - PPI
    "PR01_PRCG20_2200000000": "ppi_all",
    "PR01_PRCG20_2200010001": "ppi_manufacturing",
    "PR01_PRCG20_2200520001": "ppi_chemicals",
    "PR01_PRCG20_2200620001": "ppi_petroleum",
    "PR01_PRCG20_2200920001": "ppi_iron_steel",
    "PR01_PRCG20_2201120001": "ppi_machinery",
    "PR01_PRCG20_2201220001": "ppi_electronics",
    "PR01_PRCG20_2201620001": "ppi_transport_equip",
    # CO - Tankan
    "CO_TK99F0000601GCQ00000": "tankan_all_all_actual",
    "CO_TK99F0000601GCQ10000": "tankan_all_all_forecast",
    "CO_TK99F1000601GCQ00000": "tankan_all_mfg_actual",
    "CO_TK99F1000601GCQ10000": "tankan_all_mfg_forecast",
    "CO_TK99F2000601GCQ00000": "tankan_all_nonmfg_actual",
    "CO_TK99F2000601GCQ10000": "tankan_all_nonmfg_forecast",
    "CO_TK99F0000601GCQ01000": "tankan_large_all_actual",
    "CO_TK99F0000601GCQ11000": "tankan_large_all_forecast",
    "CO_TK99F1000601GCQ01000": "tankan_large_mfg_actual",
    "CO_TK99F1000601GCQ11000": "tankan_large_mfg_forecast",
    "CO_TK99F2000601GCQ01000": "tankan_large_nonmfg_actual",
    "CO_TK99F2000601GCQ11000": "tankan_large_nonmfg_forecast",
}

# Column prefix for BOJ macro indicators
BOJ_COLUMN_PREFIX = "BOJ_"
