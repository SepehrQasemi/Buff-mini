"""Trading spec document generator for Stage-4 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any


FAMILY_RULES: dict[str, dict[str, str]] = {
    "DonchianBreakout": {
        "code": "src/buffmini/baselines/stage0.py :: _donchian_breakout",
        "human": "Enters long on Donchian high breakout; enters short on Donchian low breakdown.",
    },
    "RSIMeanReversion": {
        "code": "src/buffmini/baselines/stage0.py :: _rsi_mean_reversion",
        "human": "Buys oversold RSI reversals and sells overbought RSI reversals.",
    },
    "TrendPullback": {
        "code": "src/buffmini/baselines/stage0.py :: _trend_pullback",
        "human": "Trades pullbacks in EMA trend with RSI confirmation.",
    },
    "BollingerMeanReversion": {
        "code": "src/buffmini/baselines/stage0.py :: _bollinger_mean_reversion",
        "human": "Long below lower Bollinger with weak RSI, short above upper Bollinger with strong RSI.",
    },
    "RangeBreakoutTrendFilter": {
        "code": "src/buffmini/baselines/stage0.py :: _range_breakout_trend_filter",
        "human": "Breakout of long Donchian range only in EMA trend direction.",
    },
}


def generate_trading_spec(
    cfg: dict[str, Any],
    stage2_metadata: dict[str, Any],
    stage3_3_choice: dict[str, Any] | None,
    selected_candidates: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Path]:
    """Generate bot-ready Stage-4 trading spec and paper checklist markdown."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checklist_path = output.parent / "paper_trading_checklist.md"

    stage4_cfg = cfg["evaluation"]["stage4"]
    execution_cfg = cfg["execution"]
    risk_cfg = cfg["risk"]
    source = "Stage-3.3" if stage3_3_choice and stage3_3_choice.get("overall_choice", {}).get("status") == "OK" else "Stage-4 defaults"
    chosen_method = (
        str(stage3_3_choice["overall_choice"]["method"])
        if stage3_3_choice and stage3_3_choice.get("overall_choice", {}).get("status") == "OK"
        else str(stage4_cfg["default_method"])
    )
    chosen_leverage = (
        float(stage3_3_choice["overall_choice"]["chosen_leverage"])
        if stage3_3_choice and stage3_3_choice.get("overall_choice", {}).get("status") == "OK"
        else float(stage4_cfg["default_leverage"])
    )

    lines: list[str] = []
    lines.append("# Trading Spec")
    lines.append("")
    lines.append("## 1) Objective & Scope")
    lines.append("- This specification defines deterministic execution for the selected Buff-mini strategy portfolio.")
    lines.append("- Instruments: `" + ", ".join(str(symbol) for symbol in cfg["universe"]["symbols"]) + "`")
    lines.append(f"- Timeframe: `{cfg['universe']['timeframe']}`")
    lines.append("- This document is for execution consistency and risk control, not a guarantee of profitability.")
    lines.append("")

    lines.append("## 2) Inputs and Assumptions")
    lines.append(f"- Stage-2 source run: `{stage2_metadata.get('run_id', 'n/a')}`")
    lines.append(f"- Stage-1 source run: `{stage2_metadata.get('stage1_run_id', 'n/a')}`")
    lines.append(f"- Stage-3.3 source: `{source}`")
    lines.append(f"- Costs: round_trip_cost_pct={cfg['costs']['round_trip_cost_pct']}%, slippage_pct={cfg['costs']['slippage_pct']}")
    lines.append(f"- Funding assumption: `{cfg['costs']['funding_pct_per_day']}` per day")
    lines.append("- Data source: local parquet cache under `data/raw/` loaded via project data layer.")
    lines.append("")

    lines.append("## 3) Strategy Set (Portfolio)")
    lines.append(f"- Selected method: `{chosen_method}`")
    lines.append(f"- Selected leverage: `{chosen_leverage}x`")
    lines.append(f"- Execution mode: `{execution_cfg['mode']}` (`per_symbol_netting={execution_cfg['per_symbol_netting']}`)")
    lines.append("| candidate_id | family | gating | exit_mode | weight |")
    lines.append("| --- | --- | --- | --- | ---: |")
    for candidate in selected_candidates:
        lines.append(
            f"| {candidate['candidate_id']} | {candidate['strategy_family']} | {candidate['gating']} | "
            f"{candidate['exit_mode']} | {float(candidate['weight']):.6f} |"
        )
    lines.append("")

    lines.append("## 4) Entry Rules")
    for candidate in selected_candidates:
        family = str(candidate["strategy_family"])
        reference = FAMILY_RULES.get(
            family,
            {
                "code": "src/buffmini/baselines/stage0.py",
                "human": "Entry rule defined in baseline module.",
            },
        )
        lines.append(f"- `{candidate['candidate_id']}` ({family}):")
        lines.append(f"  code: `{reference['code']}`")
        lines.append(f"  params: `{candidate['parameters']}`")
        lines.append(f"  description: {reference['human']}")
    lines.append("")

    lines.append("## 5) Exit Rules")
    lines.append("- Exit implementation: `src/buffmini/backtest/engine.py :: run_backtest`")
    lines.append("- Uses ATR stop, ATR target, and time-stop with configured exit mode per candidate.")
    lines.append("- Precedence on same candle follows stop-first conservative execution.")
    lines.append("- Gating filters (volatility/regime) are applied at signal generation stage.")
    lines.append("")

    lines.append("## 6) Position Sizing & Leverage")
    lines.append(f"- Leverage: `{chosen_leverage}x` ({source})")
    lines.append(f"- sizing.mode: `{risk_cfg['sizing']['mode']}`")
    lines.append(f"- sizing.risk_per_trade_pct: `{risk_cfg['sizing']['risk_per_trade_pct']}%`")
    lines.append(f"- sizing.fixed_fraction_pct: `{risk_cfg['sizing']['fixed_fraction_pct']}%`")
    lines.append(f"- max_gross_exposure: `{risk_cfg['max_gross_exposure']}x`")
    lines.append(f"- max_net_exposure_per_symbol: `{risk_cfg['max_net_exposure_per_symbol']}x`")
    lines.append("- On cap breach, all desired exposures are scaled by one multiplier to remain inside limits.")
    lines.append("")

    lines.append("## 7) Kill-Switch")
    lines.append(f"- enabled: `{risk_cfg['killswitch']['enabled']}`")
    lines.append(f"- max_daily_loss_pct: `{risk_cfg['killswitch']['max_daily_loss_pct']}%`")
    lines.append(f"- max_peak_to_valley_dd_pct: `{risk_cfg['killswitch']['max_peak_to_valley_dd_pct']}%`")
    lines.append(f"- max_consecutive_losses: `{risk_cfg['killswitch']['max_consecutive_losses']}`")
    lines.append(f"- cool_down_bars: `{risk_cfg['killswitch']['cool_down_bars']}`")
    lines.append("- Trigger behavior: stop opening new positions until cooldown expires; existing positions are not force-closed.")
    lines.append("")

    lines.append("## 8) Re-Evaluation Plan")
    lines.append(f"- cadence: `{risk_cfg['reeval']['cadence']}`")
    lines.append(f"- min_new_bars: `{risk_cfg['reeval']['min_new_bars']}`")
    lines.append("- Re-run Stage-1 discovery, Stage-2 portfolio build, and Stage-3.3 leverage selection on schedule.")
    lines.append("")

    lines.append("## 9) Monitoring Checklist (Live)")
    lines.append("- Log per bar: equity, gross exposure, per-symbol net exposure, open positions, cooldown state.")
    lines.append("- Log per day: return, drawdown, loss streak, cap-bind events, kill-switch events.")
    lines.append("- Alert when: cap scaling > 0, kill-switch triggers, exposure exceeds configured limits.")
    lines.append("")
    lines.append("No guarantee of profitability. Designed to reduce overfitting and execution drift.")

    output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    checklist_lines = [
        "# Paper Trading Checklist",
        "",
        "- Confirm config hash and run IDs are recorded before each session.",
        "- Confirm execution mode and leverage source (Stage-3.3 or fallback defaults).",
        "- Verify risk caps and kill-switch thresholds match approved values.",
        "- Run Stage-4 simulator on latest local signals before enabling paper execution.",
        "- Review cap scaling frequency and kill-switch trigger reasons.",
        "- Re-evaluate according to cadence and minimum new-bar requirement.",
    ]
    checklist_path.write_text("\n".join(checklist_lines).strip() + "\n", encoding="utf-8")
    return {"trading_spec_path": output, "paper_checklist_path": checklist_path}

