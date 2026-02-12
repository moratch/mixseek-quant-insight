"""Unit tests for EnsembleBuilder (P2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_insight.ensemble.builder import (
    EnsembleBuilder,
    EnsembleConfig,
    EnsembleResult,
    OptimizationMethod,
)
from quant_insight.pipeline.result_models import (
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screening_result(
    strategy_name: str,
    passed: bool = True,
    **overrides: Any,
) -> ScreeningResult:
    """Create a ScreeningResult with minimal defaults."""
    defaults: dict[str, Any] = {
        "execution_id": "exec-1",
        "team_id": "team-1",
        "team_name": "Team",
        "round_number": 1,
        "strategy_name": strategy_name,
        "screened_at": datetime(2025, 1, 1),
        "mode": "fast",
        "mixseek_score": 1.0,
        "wfa": WFASummary(
            n_cycles=6,
            mean_oos_sharpe=0.5,
            std_oos_sharpe=0.1,
            mean_wfe=0.7,
            std_wfe=0.05,
            consistency_score=0.8,
            degradation_rate=0.0,
            degradation_pvalue=0.5,
            trend_direction="stable",
            cycles=[],
        ),
        "cpcv": CPCVSummary(
            n_splits=6,
            purge_length=5,
            embargo_pct=0.01,
            mean_test_sharpe=0.4,
            std_test_sharpe=0.1,
            pbo=0.2,
            pbo_pvalue=0.1,
            deflated_sharpe=0.35,
            sharpe_haircut=0.1,
            consistency_ratio=0.75,
            rank_correlation=0.6,
        ),
        "verdict": ScreeningVerdict(
            passed=passed,
            criteria={"min_oos_sharpe": True},
            reasoning="All criteria passed." if passed else "Failed.",
        ),
        "adapter_config": {},
        "wfa_config": {},
        "cpcv_config": {},
    }
    defaults.update(overrides)
    return ScreeningResult(**defaults)


def _make_strategy_returns(
    name: str,
    n_days: int = 100,
    mean: float = 0.001,
    std: float = 0.02,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic daily return series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns = pd.Series(
        rng.normal(mean, std, n_days),
        index=dates,
        name=name,
    )
    return returns


def _mock_portfolio_result(
    strategy_names: list[str],
) -> MagicMock:
    """Build a mock PortfolioResult."""
    n = len(strategy_names)
    weights = {name: 1.0 / n for name in strategy_names}
    risk_contributions = {name: 1.0 / n for name in strategy_names}

    corr_df = pd.DataFrame(
        np.eye(n),
        index=strategy_names,
        columns=strategy_names,
    )

    result = MagicMock()
    result.weights = weights
    result.expected_return = 0.12
    result.volatility = 0.15
    result.sharpe_ratio = 0.8
    result.max_drawdown = -0.10
    result.diversification_ratio = 1.2
    result.risk_contributions = risk_contributions
    result.correlation_matrix = corr_df
    return result


def _setup_qal_mock(
    strategy_names: list[str],
    method: str = "sharpe",
) -> tuple[MagicMock, MagicMock]:
    """Create mock PortfolioOptimizer module and class.

    Returns:
        (mock_module, mock_optimizer_instance)
    """
    mock_portfolio_result = _mock_portfolio_result(strategy_names)

    mock_optimizer_instance = MagicMock()
    mock_optimizer_instance.add_strategy.side_effect = lambda name, returns, n_trades=0: MagicMock(name=name)
    mock_optimizer_instance.optimize_sharpe.return_value = mock_portfolio_result
    mock_optimizer_instance.optimize_risk_parity.return_value = mock_portfolio_result

    mock_optimizer_cls = MagicMock(return_value=mock_optimizer_instance)

    mock_module = MagicMock()
    mock_module.PortfolioOptimizer = mock_optimizer_cls

    return mock_module, mock_optimizer_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestEnsembleConfig:
    """Test EnsembleConfig defaults and frozen behavior."""

    def test_defaults(self) -> None:
        config = EnsembleConfig()
        assert config.optimization_method == OptimizationMethod.SHARPE
        assert config.max_weight == 0.5
        assert config.min_weight == 0.05
        assert config.risk_free_rate == 0.0

    def test_frozen(self) -> None:
        config = EnsembleConfig()
        with pytest.raises(AttributeError):
            config.max_weight = 0.9  # type: ignore[misc]


@pytest.mark.unit()
class TestEnsembleBuild:
    """Test EnsembleBuilder.build() with mocked PortfolioOptimizer."""

    def test_sharpe_optimization(self) -> None:
        """Build with sharpe method calls optimize_sharpe."""
        names = ["StratA R1", "StratB R1"]
        mock_module, mock_instance = _setup_qal_mock(names)

        results = [_make_screening_result(n) for n in names]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        config = EnsembleConfig(optimization_method=OptimizationMethod.SHARPE)
        builder = EnsembleBuilder(config)

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.portfolio_optimizer": mock_module},
        ):
            ensemble = builder.build(results, returns)

        assert isinstance(ensemble, EnsembleResult)
        assert ensemble.optimization_method == "sharpe"
        assert len(ensemble.weights) == 2
        mock_instance.optimize_sharpe.assert_called_once()
        mock_instance.optimize_risk_parity.assert_not_called()

    def test_risk_parity_optimization(self) -> None:
        """Build with risk_parity method calls optimize_risk_parity."""
        names = ["StratA R1", "StratB R1"]
        mock_module, mock_instance = _setup_qal_mock(names, method="risk_parity")

        results = [_make_screening_result(n) for n in names]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        config = EnsembleConfig(optimization_method=OptimizationMethod.RISK_PARITY)
        builder = EnsembleBuilder(config)

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.portfolio_optimizer": mock_module},
        ):
            ensemble = builder.build(results, returns)

        assert isinstance(ensemble, EnsembleResult)
        assert ensemble.optimization_method == "risk_parity"
        mock_instance.optimize_risk_parity.assert_called_once()
        mock_instance.optimize_sharpe.assert_not_called()

    def test_filters_failed_strategies(self) -> None:
        """Only passed strategies are included in ensemble."""
        names = ["StratA R1", "StratB R1", "StratC R1"]
        mock_module, mock_instance = _setup_qal_mock(names[:2])

        results = [
            _make_screening_result("StratA R1", passed=True),
            _make_screening_result("StratB R1", passed=False),  # Failed
            _make_screening_result("StratC R1", passed=True),
        ]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        builder = EnsembleBuilder(EnsembleConfig())

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.portfolio_optimizer": mock_module},
        ):
            builder.build(results, returns)

        # Only 2 passed strategies should be added
        assert mock_instance.add_strategy.call_count == 2
        call_names = [call.kwargs["name"] for call in mock_instance.add_strategy.call_args_list]
        assert "StratB R1" not in call_names

    def test_raises_on_empty_passed(self) -> None:
        """Raises ValueError when no strategies passed screening."""
        results = [_make_screening_result("Fail", passed=False)]
        returns = {"Fail": _make_strategy_returns("Fail")}

        builder = EnsembleBuilder(EnsembleConfig())

        with pytest.raises(ValueError, match="No screening-passed"):
            builder.build(results, returns)

    def test_raises_on_insufficient_strategies(self) -> None:
        """Raises ValueError when fewer than 2 strategies have return data."""
        results = [_make_screening_result("Solo R1", passed=True)]
        returns = {"Solo R1": _make_strategy_returns("Solo R1")}

        builder = EnsembleBuilder(EnsembleConfig())

        with pytest.raises(ValueError, match="at least 2"):
            builder.build(results, returns)

    def test_skips_missing_return_data(self) -> None:
        """Strategies without matching return data are skipped with warning."""
        names = ["HasData R1", "NoData R1", "HasData2 R1"]
        mock_module, mock_instance = _setup_qal_mock(["HasData R1", "HasData2 R1"])

        results = [_make_screening_result(n) for n in names]
        # Only provide returns for 2 of 3
        returns = {
            "HasData R1": _make_strategy_returns("HasData R1", seed=0),
            "HasData2 R1": _make_strategy_returns("HasData2 R1", seed=2),
        }

        builder = EnsembleBuilder(EnsembleConfig())

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.portfolio_optimizer": mock_module},
        ):
            builder.build(results, returns)

        assert mock_instance.add_strategy.call_count == 2

    def test_result_fields_populated(self) -> None:
        """All EnsembleResult fields are populated from PortfolioResult."""
        names = ["StratA R1", "StratB R1"]
        mock_module, _ = _setup_qal_mock(names)

        results = [_make_screening_result(n) for n in names]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        builder = EnsembleBuilder(EnsembleConfig())

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.portfolio_optimizer": mock_module},
        ):
            ensemble = builder.build(results, returns)

        assert ensemble.expected_return == pytest.approx(0.12)
        assert ensemble.volatility == pytest.approx(0.15)
        assert ensemble.sharpe_ratio == pytest.approx(0.8)
        assert ensemble.max_drawdown == pytest.approx(-0.10)
        assert ensemble.diversification_ratio == pytest.approx(1.2)
        assert ensemble.n_strategies_input == 2
        assert len(ensemble.component_strategies) == 2
        assert len(ensemble.risk_contributions) == 2
        assert len(ensemble.correlation_matrix) == 2

    def test_config_passed_to_optimizer(self) -> None:
        """EnsembleConfig values are forwarded to PortfolioOptimizer."""
        names = ["StratA R1", "StratB R1"]
        mock_module, _ = _setup_qal_mock(names)

        results = [_make_screening_result(n) for n in names]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        config = EnsembleConfig(
            max_weight=0.6,
            min_weight=0.1,
            risk_free_rate=0.02,
        )
        builder = EnsembleBuilder(config)

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.portfolio_optimizer": mock_module},
        ):
            builder.build(results, returns)

        # Verify constructor was called with our config values
        mock_module.PortfolioOptimizer.assert_called_once_with(
            risk_free_rate=0.02,
            max_weight=0.6,
            min_weight=0.1,
        )

    def test_raises_on_infeasible_min_weight(self) -> None:
        """n * min_weight > 1 raises ValueError before optimization."""
        # 3 strategies with min_weight=0.4 → 3*0.4 = 1.2 > 1.0
        names = ["S1", "S2", "S3"]
        results = [_make_screening_result(n) for n in names]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        config = EnsembleConfig(min_weight=0.4, max_weight=0.5)
        builder = EnsembleBuilder(config)

        with pytest.raises(ValueError, match="infeasible.*min_weight"):
            builder.build(results, returns)

    def test_raises_on_infeasible_max_weight(self) -> None:
        """n * max_weight < 1 raises ValueError before optimization."""
        # 5 strategies with max_weight=0.15 → 5*0.15 = 0.75 < 1.0
        names = [f"S{i}" for i in range(5)]
        results = [_make_screening_result(n) for n in names]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        config = EnsembleConfig(min_weight=0.0, max_weight=0.15)
        builder = EnsembleBuilder(config)

        with pytest.raises(ValueError, match="infeasible.*max_weight"):
            builder.build(results, returns)

    def test_risk_parity_warns_on_min_weight_override(self, caplog: pytest.LogCaptureFixture) -> None:
        """risk_parity mode logs warning about min_weight being ignored."""
        names = ["StratA R1", "StratB R1"]
        mock_module, _ = _setup_qal_mock(names)

        results = [_make_screening_result(n) for n in names]
        returns = {n: _make_strategy_returns(n, seed=i) for i, n in enumerate(names)}

        config = EnsembleConfig(
            optimization_method=OptimizationMethod.RISK_PARITY,
            min_weight=0.05,
        )
        builder = EnsembleBuilder(config)

        with (
            patch.dict(
                "sys.modules",
                {"core": MagicMock(), "core.portfolio_optimizer": mock_module},
            ),
            caplog.at_level("WARNING"),
        ):
            builder.build(results, returns)

        assert any("risk_parity" in rec.message and "0.01" in rec.message for rec in caplog.records)
