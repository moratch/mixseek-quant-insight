"""Ensemble strategy builder using quant-alpha-lab PortfolioOptimizer.

Combines multiple screening-passed strategies into an optimized
portfolio using Sharpe maximization or risk parity.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd

from quant_insight.pipeline.result_models import ScreeningResult

logger = logging.getLogger(__name__)


class OptimizationMethod(StrEnum):
    """Portfolio optimization method."""

    SHARPE = "sharpe"
    RISK_PARITY = "risk_parity"


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for ensemble portfolio construction.

    Attributes:
        optimization_method: Portfolio optimization approach.
        max_weight: Maximum weight per strategy (prevents concentration).
        min_weight: Minimum weight per strategy (prevents trivial allocations).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        qal_path: Path to quant-alpha-lab root directory.
    """

    optimization_method: OptimizationMethod = OptimizationMethod.SHARPE
    max_weight: float = 0.5
    min_weight: float = 0.05
    risk_free_rate: float = 0.0
    qal_path: Path = Path("C:/Dev/quant-alpha-lab")


@dataclass
class EnsembleResult:
    """Result of ensemble portfolio optimization.

    Attributes:
        weights: Strategy name to portfolio weight mapping.
        expected_return: Annualized expected portfolio return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Portfolio Sharpe ratio.
        max_drawdown: Maximum drawdown of the combined portfolio.
        diversification_ratio: Ratio of weighted-sum volatility to portfolio
            volatility (>1 indicates diversification benefit).
        component_strategies: Ordered list of included strategy names.
        risk_contributions: Strategy name to risk contribution mapping.
        optimization_method: Method used for optimization.
        n_strategies_input: Number of passed strategies provided.
        correlation_matrix: Strategy return correlation matrix.
    """

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    component_strategies: list[str]
    risk_contributions: dict[str, float]
    optimization_method: str = ""
    n_strategies_input: int = 0
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)


class EnsembleBuilder:
    """Builds ensemble portfolios from screening-passed strategies.

    Usage:
        builder = EnsembleBuilder(EnsembleConfig())
        result = builder.build(screening_results, strategy_returns)
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self._ensure_qal_path()

    def _ensure_qal_path(self) -> None:
        """Add quant-alpha-lab to sys.path if not already present."""
        qal_str = str(self.config.qal_path)
        if qal_str not in sys.path:
            sys.path.insert(0, qal_str)

    def build(
        self,
        screening_results: list[ScreeningResult],
        strategy_returns: dict[str, pd.Series],
    ) -> EnsembleResult:
        """Build an optimized ensemble from screening-passed strategies.

        Only strategies with ``verdict.passed == True`` are included.
        Each passed strategy must have a corresponding entry in
        ``strategy_returns`` keyed by ``strategy_name``.

        Args:
            screening_results: List of ScreeningResult from P1 pipeline.
            strategy_returns: Map of strategy_name to daily return pd.Series.

        Returns:
            EnsembleResult with optimal weights and portfolio metrics.

        Raises:
            ValueError: If fewer than 2 passed strategies have return data.
        """
        passed = [r for r in screening_results if r.verdict.passed]
        if not passed:
            msg = "No screening-passed strategies to ensemble"
            raise ValueError(msg)

        # Match passed results to return data
        matched: list[tuple[ScreeningResult, pd.Series]] = []
        for result in passed:
            returns = strategy_returns.get(result.strategy_name)
            if returns is None:
                logger.warning(
                    "No return data for strategy '%s', skipping",
                    result.strategy_name,
                )
                continue
            if returns.empty:
                logger.warning(
                    "Empty return data for strategy '%s', skipping",
                    result.strategy_name,
                )
                continue
            matched.append((result, returns))

        if len(matched) < 2:
            msg = f"Need at least 2 strategies with return data for ensemble, got {len(matched)}"
            raise ValueError(msg)

        # Feasibility check: n * min_weight <= 1 <= n * max_weight
        n = len(matched)
        self._validate_weight_feasibility(n)

        # Warn if risk_parity silently overrides min_weight
        method = OptimizationMethod(self.config.optimization_method)
        if method == OptimizationMethod.RISK_PARITY and self.config.min_weight != 0.01:
            logger.warning(
                "risk_parity mode: qal uses fixed lower bound 0.01, ignoring configured min_weight=%.3f",
                self.config.min_weight,
            )

        logger.info(
            "Building ensemble: %d strategies (method=%s)",
            n,
            self.config.optimization_method,
        )

        return self._optimize(matched)

    def _validate_weight_feasibility(self, n: int) -> None:
        """Check that weight constraints are mathematically feasible.

        For n strategies with sum-to-1 constraint, we need:
            n * min_weight <= 1  (lower bounds don't exceed budget)
            n * max_weight >= 1  (upper bounds can reach budget)

        Raises:
            ValueError: If constraints are infeasible.
        """
        total_min = n * self.config.min_weight
        total_max = n * self.config.max_weight
        if total_min > 1.0:
            msg = (
                f"Weight constraints infeasible: {n} strategies * "
                f"min_weight={self.config.min_weight} = {total_min:.2f} > 1.0. "
                f"Reduce min_weight or use fewer strategies."
            )
            raise ValueError(msg)
        if total_max < 1.0:
            msg = (
                f"Weight constraints infeasible: {n} strategies * "
                f"max_weight={self.config.max_weight} = {total_max:.2f} < 1.0. "
                f"Increase max_weight or use more strategies."
            )
            raise ValueError(msg)

    def _optimize(
        self,
        matched: list[tuple[ScreeningResult, pd.Series]],
    ) -> EnsembleResult:
        """Run portfolio optimization via quant-alpha-lab.

        Args:
            matched: List of (ScreeningResult, daily_returns) tuples.

        Returns:
            EnsembleResult.
        """
        from core.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(
            risk_free_rate=self.config.risk_free_rate,
            max_weight=self.config.max_weight,
            min_weight=self.config.min_weight,
        )

        strategies = []
        for result, returns in matched:
            perf = optimizer.add_strategy(
                name=result.strategy_name,
                returns=returns,
            )
            strategies.append(perf)

        method = OptimizationMethod(self.config.optimization_method)
        if method == OptimizationMethod.SHARPE:
            portfolio = optimizer.optimize_sharpe(strategies)
        elif method == OptimizationMethod.RISK_PARITY:
            portfolio = optimizer.optimize_risk_parity(strategies)
        else:
            msg = f"Unknown optimization method: {method}"
            raise ValueError(msg)

        # Convert correlation matrix to serializable dict
        corr_dict: dict[str, dict[str, float]] = {}
        if portfolio.correlation_matrix is not None:
            corr_df: pd.DataFrame = portfolio.correlation_matrix
            for col in corr_df.columns:
                corr_dict[str(col)] = {str(idx): float(val) for idx, val in corr_df[col].items()}

        return EnsembleResult(
            weights=dict(portfolio.weights),
            expected_return=float(portfolio.expected_return),
            volatility=float(portfolio.volatility),
            sharpe_ratio=float(portfolio.sharpe_ratio),
            max_drawdown=float(portfolio.max_drawdown),
            diversification_ratio=float(portfolio.diversification_ratio),
            component_strategies=[s.name for s in strategies],
            risk_contributions=dict(portfolio.risk_contributions),
            optimization_method=str(self.config.optimization_method),
            n_strategies_input=len(matched),
            correlation_matrix=corr_dict,
        )
