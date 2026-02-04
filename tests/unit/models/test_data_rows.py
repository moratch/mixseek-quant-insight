"""Unit tests for data row models."""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_insight.models.data_rows import OHLCVRow, ReturnRow, SignalRow


class TestOHLCVRow:
    """Test OHLCVRow model."""

    def test_valid_ohlcv_row(self) -> None:
        """Test valid OHLCVRow."""
        row = OHLCVRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            open=Decimal("100.5"),
            high=Decimal("102.0"),
            low=Decimal("100.0"),
            close=Decimal("101.5"),
            volume=1000000,
        )
        assert row.datetime == datetime(2023, 1, 1)
        assert row.symbol == "AAPL"
        assert row.open == Decimal("100.5")
        assert row.volume == 1000000

    def test_ohlcv_volume_must_be_non_negative(self) -> None:
        """Test volume must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            OHLCVRow(
                datetime=datetime(2023, 1, 1),
                symbol="AAPL",
                open=Decimal("100.0"),
                high=Decimal("100.0"),
                low=Decimal("100.0"),
                close=Decimal("100.0"),
                volume=-100,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_ohlcv_zero_volume_allowed(self) -> None:
        """Test volume can be 0."""
        row = OHLCVRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            open=Decimal("100.0"),
            high=Decimal("100.0"),
            low=Decimal("100.0"),
            close=Decimal("100.0"),
            volume=0,
        )
        assert row.volume == 0


class TestReturnRow:
    """Test ReturnRow model."""

    def test_valid_return_row(self) -> None:
        """Test valid ReturnRow."""
        row = ReturnRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            return_value=0.05,
        )
        assert row.datetime == datetime(2023, 1, 1)
        assert row.symbol == "AAPL"
        assert row.return_value == 0.05

    def test_return_row_negative_value(self) -> None:
        """Test ReturnRow allows negative returns."""
        row = ReturnRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            return_value=-0.03,
        )
        assert row.return_value == -0.03

    def test_return_row_zero_value(self) -> None:
        """Test ReturnRow allows zero returns."""
        row = ReturnRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            return_value=0.0,
        )
        assert row.return_value == 0.0


class TestSignalRow:
    """Test SignalRow model."""

    def test_valid_signal_row(self) -> None:
        """Test valid SignalRow."""
        row = SignalRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            signal=0.8,
        )
        assert row.datetime == datetime(2023, 1, 1)
        assert row.symbol == "AAPL"
        assert row.signal == 0.8

    def test_signal_row_negative_value(self) -> None:
        """Test SignalRow allows negative signals."""
        row = SignalRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            signal=-0.5,
        )
        assert row.signal == -0.5

    def test_signal_row_zero_value(self) -> None:
        """Test SignalRow allows zero signals."""
        row = SignalRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            signal=0.0,
        )
        assert row.signal == 0.0


class TestDataRowsSerialization:
    """Test data row serialization."""

    def test_ohlcv_row_model_dump(self) -> None:
        """Test OHLCVRow serialization."""
        row = OHLCVRow(
            datetime=datetime(2023, 1, 1, 10, 0, 0),
            symbol="AAPL",
            open=Decimal("100.0"),
            high=Decimal("101.0"),
            low=Decimal("99.0"),
            close=Decimal("100.5"),
            volume=500000,
        )
        data = row.model_dump()
        assert data["symbol"] == "AAPL"
        assert data["volume"] == 500000

    def test_return_row_model_dump(self) -> None:
        """Test ReturnRow serialization."""
        row = ReturnRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            return_value=0.025,
        )
        data = row.model_dump()
        assert data["symbol"] == "AAPL"
        assert data["return_value"] == 0.025

    def test_signal_row_model_dump(self) -> None:
        """Test SignalRow serialization."""
        row = SignalRow(
            datetime=datetime(2023, 1, 1),
            symbol="AAPL",
            signal=0.75,
        )
        data = row.model_dump()
        assert data["symbol"] == "AAPL"
        assert data["signal"] == 0.75
