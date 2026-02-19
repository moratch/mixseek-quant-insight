"""Screening CLI commands for quant-alpha-lab statistical verification."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from quant_insight.adapter.signal_to_strategy import ThresholdMethod
from quant_insight.pipeline.screening import ScreeningMode

screening_app = typer.Typer(
    name="screening",
    help="Statistical screening pipeline for MixSeek signal functions",
)


@screening_app.command("run")
def run_screening(
    workspace: Annotated[
        str,
        typer.Option(
            "--workspace",
            "-w",
            envvar="MIXSEEK_WORKSPACE",
            help="Workspace path (or set MIXSEEK_WORKSPACE env var)",
        ),
    ],
    execution_id: Annotated[
        str | None,
        typer.Option(
            "--execution-id",
            "-e",
            help="Specific execution_id to screen (default: latest)",
        ),
    ] = None,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Minimum MixSeek score filter"),
    ] = 0.0,
    mode: Annotated[
        ScreeningMode,
        typer.Option("--mode", help="Execution mode: fast (exploration) / strict (adoption)"),
    ] = ScreeningMode.FAST,
    threshold_method: Annotated[
        ThresholdMethod,
        typer.Option("--threshold-method", help="Signal discretization: quantile / fixed / zero"),
    ] = ThresholdMethod.QUANTILE,
    long_quantile: Annotated[
        float,
        typer.Option("--long-quantile", help="Quantile threshold for long signals (default: 0.9)"),
    ] = 0.9,
    short_quantile: Annotated[
        float,
        typer.Option("--short-quantile", help="Quantile threshold for short signals (default: 0.1)"),
    ] = 0.1,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output JSON path (default: stdout)"),
    ] = None,
    persist: Annotated[
        bool,
        typer.Option("--persist/--no-persist", help="Persist results to DuckDB for API access"),
    ] = False,
    # Screening criteria overrides
    min_oos_sharpe: Annotated[
        float,
        typer.Option("--min-oos-sharpe", help="Minimum OOS Sharpe ratio (default: 0.3)"),
    ] = 0.3,
    min_wfe: Annotated[
        float,
        typer.Option("--min-wfe", help="Minimum Walk-Forward Efficiency (default: 0.5)"),
    ] = 0.5,
    min_consistency: Annotated[
        float,
        typer.Option("--min-consistency", help="Minimum consistency score (default: 0.6)"),
    ] = 0.6,
    max_pbo: Annotated[
        float,
        typer.Option("--max-pbo", help="Maximum PBO (default: 0.4)"),
    ] = 0.4,
    min_dsr: Annotated[
        float,
        typer.Option("--min-dsr", help="Minimum Deflated Sharpe Ratio (default: 0.2)"),
    ] = 0.2,
    min_degradation_pvalue: Annotated[
        float,
        typer.Option("--min-degradation-pvalue", help="Minimum degradation p-value (default: 0.05)"),
    ] = 0.05,
    # CPCV / WFA settings
    cpcv_n_splits: Annotated[
        int,
        typer.Option("--cpcv-n-splits", help="CPCV number of splits (default: 6, try 10 for finer PBO)"),
    ] = 6,
    cpcv_n_test_groups: Annotated[
        int,
        typer.Option("--cpcv-n-test-groups", help="CPCV test groups (default: 2)"),
    ] = 2,
) -> None:
    """Extract candidates from DuckDB and verify via quant-alpha-lab.

    Runs WFA and CPCV analysis on each candidate's signal function,
    then produces pass/fail verdicts based on screening criteria.

    Example:
        quant-insight screening run -w workspace -e "abc-123" --mode fast -o results.json
    """
    from quant_insight.pipeline.screening import ScreeningConfig, ScreeningPipeline

    config = ScreeningConfig(
        workspace=Path(workspace),
        threshold_method=threshold_method,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        cpcv_n_splits=cpcv_n_splits,
        cpcv_n_test_groups=cpcv_n_test_groups,
        min_oos_sharpe=min_oos_sharpe,
        min_wfe=min_wfe,
        min_consistency=min_consistency,
        max_pbo=max_pbo,
        min_dsr=min_dsr,
        min_degradation_pvalue=min_degradation_pvalue,
    )
    pipeline = ScreeningPipeline(config)

    # Extract candidates
    typer.echo("Extracting candidates...")
    candidates = pipeline.extract_candidates(execution_id, min_score)
    if not candidates:
        typer.echo("No candidates found.")
        raise typer.Exit(0)

    typer.echo(f"Found {len(candidates)} candidate(s)")
    for c in candidates:
        typer.echo(f"  - {c['team_name']} R{c['round_number']} (score={c['score']:.2f})")

    # Run screening
    typer.echo(f"\nRunning {mode.value} screening...")
    batch_result = pipeline.screen_batch(candidates, mode=mode)

    # Serialize to JSON
    result_dict = asdict(batch_result)
    result_json = json.dumps(result_dict, default=str, ensure_ascii=False, indent=2)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")
        typer.echo(f"\nResults saved to: {output_path}")
    else:
        typer.echo(result_json)

    # Persist to DuckDB (fail-open: warning only on failure)
    if persist:
        try:
            from quant_insight.storage.screening_store import ScreeningResultStore

            store = ScreeningResultStore(workspace=Path(workspace))
            store.initialize_schema()
            store.save_batch(batch_result)
            typer.echo(f"Results persisted to DuckDB ({len(batch_result.results)} records)")
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to persist to DuckDB (non-fatal): {e}")

    # Summary
    typer.echo(f"\n{'=' * 40}")
    typer.echo(f"Candidates: {batch_result.n_candidates}")
    typer.echo(f"Passed:     {batch_result.n_passed}")
    typer.echo(f"Failed:     {batch_result.n_failed}")


@screening_app.command("show")
def show_result(
    result_file: Annotated[
        str,
        typer.Argument(help="Path to screening result JSON file"),
    ],
) -> None:
    """Display screening results in a human-readable format.

    Example:
        quant-insight screening show workspace/reports/screening_results.json
    """
    path = Path(result_file)
    if not path.exists():
        typer.echo(f"File not found: {path}", err=True)
        raise typer.Exit(1)

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    typer.echo(f"Screened at:  {data['screened_at']}")
    typer.echo(f"Candidates:   {data['n_candidates']} (Passed: {data['n_passed']}, Failed: {data['n_failed']})")
    typer.echo("")

    for result in data.get("results", []):
        _print_single_result(result)


def _print_single_result(result: dict[str, Any]) -> None:
    """Pretty-print a single screening result."""
    typer.echo(f"Strategy: {result['strategy_name']}")
    typer.echo(f"MixSeek Score: {result['mixseek_score']:.2f}")
    typer.echo(f"Mode: {result['mode']}")
    typer.echo("")

    wfa = result["wfa"]
    typer.echo(f"--- WFA Results ({wfa['n_cycles']} cycles) ---")
    typer.echo(f"Mean OOS Sharpe:  {wfa['mean_oos_sharpe']:.3f}")
    typer.echo(f"Mean WFE:         {wfa['mean_wfe'] * 100:.1f}%")
    typer.echo(f"Consistency:      {wfa['consistency_score'] * 100:.1f}%")
    typer.echo(f"Degradation:      {wfa['trend_direction']} (p={wfa['degradation_pvalue']:.3f})")
    typer.echo("")

    cpcv = result["cpcv"]
    typer.echo("--- CPCV Results ---")
    typer.echo(f"PBO:              {cpcv['pbo'] * 100:.1f}%")
    typer.echo(f"Deflated Sharpe:  {cpcv['deflated_sharpe']:.3f} (source: {cpcv['deflated_sharpe_source']})")
    typer.echo(f"Consistency:      {cpcv['consistency_ratio'] * 100:.1f}%")
    typer.echo("")

    verdict = result["verdict"]
    typer.echo("--- Verdict ---")
    for criterion, passed in verdict["criteria"].items():
        status = "PASS" if passed else "FAIL"
        typer.echo(f"  {criterion:30s} {status}")
    typer.echo(f"  {'─' * 36}")
    overall = "PASS" if verdict["passed"] else "FAIL"
    typer.echo(f"  {'Overall':30s} {overall}")
    typer.echo("")
