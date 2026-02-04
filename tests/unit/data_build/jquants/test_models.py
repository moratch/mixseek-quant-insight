"""Unit tests for J-Quants models and enumerations."""

import pytest

from quant_insight.data_build.jquants.models import (
    MASTER_COLUMN_MAPPING,
    OHLCV_COLUMN_MAPPING,
    UNIVERSE_TO_MARKET_CODE,
    JQuantsPlan,
    JQuantsUniverse,
)


class TestJQuantsPlan:
    """JQuantsPlan Enumのテスト."""

    def test_plan_values(self) -> None:
        """全てのプラン値が正しく定義されていること."""
        assert JQuantsPlan.FREE.value == "free"
        assert JQuantsPlan.LIGHT.value == "light"
        assert JQuantsPlan.STANDARD.value == "standard"
        assert JQuantsPlan.PREMIUM.value == "premium"

    def test_plan_count(self) -> None:
        """4つのプランが定義されていること."""
        assert len(JQuantsPlan) == 4

    def test_plan_from_string(self) -> None:
        """文字列からEnumに変換できること."""
        assert JQuantsPlan("free") == JQuantsPlan.FREE
        assert JQuantsPlan("light") == JQuantsPlan.LIGHT
        assert JQuantsPlan("standard") == JQuantsPlan.STANDARD
        assert JQuantsPlan("premium") == JQuantsPlan.PREMIUM

    def test_plan_invalid_value(self) -> None:
        """無効な値でValueErrorが発生すること."""
        with pytest.raises(ValueError):
            JQuantsPlan("invalid")


class TestJQuantsUniverse:
    """JQuantsUniverse Enumのテスト."""

    def test_universe_values(self) -> None:
        """全てのユニバース値が正しく定義されていること."""
        assert JQuantsUniverse.PRIME.value == "prime"
        assert JQuantsUniverse.STANDARD.value == "standard"
        assert JQuantsUniverse.GROWTH.value == "growth"
        assert JQuantsUniverse.ALL.value == "all"

    def test_universe_count(self) -> None:
        """4つのユニバースが定義されていること."""
        assert len(JQuantsUniverse) == 4

    def test_universe_from_string(self) -> None:
        """文字列からEnumに変換できること."""
        assert JQuantsUniverse("prime") == JQuantsUniverse.PRIME
        assert JQuantsUniverse("standard") == JQuantsUniverse.STANDARD
        assert JQuantsUniverse("growth") == JQuantsUniverse.GROWTH
        assert JQuantsUniverse("all") == JQuantsUniverse.ALL


class TestUniverseToMarketCode:
    """UNIVERSE_TO_MARKET_CODEマッピングのテスト."""

    def test_prime_market_code(self) -> None:
        """プライム市場のコードが正しいこと."""
        assert UNIVERSE_TO_MARKET_CODE[JQuantsUniverse.PRIME] == "0111"

    def test_standard_market_code(self) -> None:
        """スタンダード市場のコードが正しいこと."""
        assert UNIVERSE_TO_MARKET_CODE[JQuantsUniverse.STANDARD] == "0112"

    def test_growth_market_code(self) -> None:
        """グロース市場のコードが正しいこと."""
        assert UNIVERSE_TO_MARKET_CODE[JQuantsUniverse.GROWTH] == "0113"

    def test_all_market_code(self) -> None:
        """全銘柄はNone（フィルタなし）であること."""
        assert UNIVERSE_TO_MARKET_CODE[JQuantsUniverse.ALL] is None

    def test_all_universes_covered(self) -> None:
        """全てのユニバースがマッピングに含まれていること."""
        for universe in JQuantsUniverse:
            assert universe in UNIVERSE_TO_MARKET_CODE


class TestOhlcvColumnMapping:
    """OHLCVカラムマッピングのテスト."""

    def test_standard_schema_columns(self) -> None:
        """標準スキーマカラムが含まれていること."""
        assert OHLCV_COLUMN_MAPPING["Date"] == "datetime"
        assert OHLCV_COLUMN_MAPPING["Code"] == "symbol"
        assert OHLCV_COLUMN_MAPPING["AdjO"] == "open"
        assert OHLCV_COLUMN_MAPPING["AdjH"] == "high"
        assert OHLCV_COLUMN_MAPPING["AdjL"] == "low"
        assert OHLCV_COLUMN_MAPPING["AdjC"] == "close"
        assert OHLCV_COLUMN_MAPPING["AdjVo"] == "volume"

    def test_raw_price_columns(self) -> None:
        """生価格カラムが含まれていること."""
        assert OHLCV_COLUMN_MAPPING["O"] == "raw_open"
        assert OHLCV_COLUMN_MAPPING["H"] == "raw_high"
        assert OHLCV_COLUMN_MAPPING["L"] == "raw_low"
        assert OHLCV_COLUMN_MAPPING["C"] == "raw_close"
        assert OHLCV_COLUMN_MAPPING["Vo"] == "raw_volume"

    def test_turnover_column(self) -> None:
        """売買代金カラムが含まれていること."""
        assert OHLCV_COLUMN_MAPPING["Va"] == "turnover"

    def test_session_columns(self) -> None:
        """前場・後場カラムが含まれていること."""
        # 前場
        assert OHLCV_COLUMN_MAPPING["MO"] == "morning_open"
        assert OHLCV_COLUMN_MAPPING["MH"] == "morning_high"
        assert OHLCV_COLUMN_MAPPING["ML"] == "morning_low"
        assert OHLCV_COLUMN_MAPPING["MC"] == "morning_close"
        # 後場
        assert OHLCV_COLUMN_MAPPING["AO"] == "afternoon_open"
        assert OHLCV_COLUMN_MAPPING["AH"] == "afternoon_high"
        assert OHLCV_COLUMN_MAPPING["AL"] == "afternoon_low"
        assert OHLCV_COLUMN_MAPPING["AC"] == "afternoon_close"

    def test_column_count(self) -> None:
        """期待通りのカラム数であること.

        Date, Code (2)
        + Adj OHLCV (5): AdjO, AdjH, AdjL, AdjC, AdjVo
        + AdjFactor (1)
        + Raw OHLCV (5): O, H, L, C, Vo
        + Value limits (2): UL, LL
        + Turnover (1): Va
        + Morning (4): MO, MH, ML, MC
        + Morning limits (2): MUL, MLL
        + Morning volume/turnover (2): MVo, MVa
        + Morning Adj (5): MAdjO, MAdjH, MAdjL, MAdjC, MAdjVo
        + Afternoon (4): AO, AH, AL, AC
        + Afternoon limits (2): AUL, ALL
        + Afternoon volume/turnover (2): AVo, AVa
        + Afternoon Adj (5): AAdjO, AAdjH, AAdjL, AAdjC, AAdjVo
        = 42 columns
        """
        assert len(OHLCV_COLUMN_MAPPING) == 42


class TestMasterColumnMapping:
    """Masterカラムマッピングのテスト."""

    def test_basic_columns(self) -> None:
        """基本カラムが含まれていること."""
        assert MASTER_COLUMN_MAPPING["Date"] == "datetime"
        assert MASTER_COLUMN_MAPPING["Code"] == "symbol"
        assert MASTER_COLUMN_MAPPING["CoName"] == "company_name"
        assert MASTER_COLUMN_MAPPING["CoNameEn"] == "company_name_en"

    def test_sector_columns(self) -> None:
        """セクターカラムが含まれていること."""
        assert MASTER_COLUMN_MAPPING["S17"] == "sector17_code"
        assert MASTER_COLUMN_MAPPING["S17Nm"] == "sector17_name"
        assert MASTER_COLUMN_MAPPING["S33"] == "sector33_code"
        assert MASTER_COLUMN_MAPPING["S33Nm"] == "sector33_name"

    def test_market_columns(self) -> None:
        """市場カラムが含まれていること."""
        assert MASTER_COLUMN_MAPPING["Mkt"] == "market_code"
        assert MASTER_COLUMN_MAPPING["MktNm"] == "market_name"

    def test_other_columns(self) -> None:
        """その他のカラムが含まれていること."""
        assert MASTER_COLUMN_MAPPING["ScaleCat"] == "scale_category"
        assert MASTER_COLUMN_MAPPING["Mrgn"] == "margin_code"
        assert MASTER_COLUMN_MAPPING["MrgnNm"] == "margin_name"

    def test_column_count(self) -> None:
        """期待通りのカラム数であること."""
        assert len(MASTER_COLUMN_MAPPING) == 13
