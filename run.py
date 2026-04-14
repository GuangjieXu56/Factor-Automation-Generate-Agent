from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from agents.JudgeAgent import JA


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "factor_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JudgeAgent and export confirmed factors for downstream stock selection and portfolio construction.",
    )
    parser.add_argument("--pdf-path", default="data/sample1.pdf", help="Path to the source PDF.")
    parser.add_argument("--query", default="construct a factor", help="Task query for JudgeAgent.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DeepSeek-V3.2", "Qwen3.5-27B"],
        help="Judge models to compare.",
    )
    parser.add_argument(
        "--parquet-path",
        default="/data/stock_data.parquet",
        help="Stock panel parquet path passed to JudgeAgent.",
    )
    parser.add_argument("--numeric-tolerance", type=float, default=1e-5, help="Numeric tolerance for factor matching.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries for one LLM JSON parse round.")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max FCA retry rounds inside one model branch.")
    parser.add_argument(
        "--max-judge-iterations",
        type=int,
        default=3,
        help="Max judge-level refinement iterations when models disagree.",
    )
    parser.add_argument(
        "--export-root",
        default=str(DEFAULT_EXPORT_ROOT),
        help="Directory for run artifacts and exported confirmed factors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_root = Path(args.export_root)
    run_dir = build_run_dir(export_root=export_root, pdf_path=args.pdf_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = JA(
        parquet_path=args.parquet_path,
        model_list=args.models,
        numeric_tolerance=args.numeric_tolerance,
        max_judge_iterations=args.max_judge_iterations,
    )

    result = agent.run_ja(
        model_list=args.models,
        pdf_path=args.pdf_path,
        query=args.query,
        max_retires=args.max_retries,
        max_rounds=args.max_rounds,
        max_judge_iterations=args.max_judge_iterations,
        save_backtest=True,
    )

    exported_files = export_run_outputs(result=result, run_dir=run_dir)

    print_run_summary(run_dir=run_dir, result=result, exported_files=exported_files)


def build_run_dir(export_root: Path, pdf_path: str) -> Path:
    pdf_stem = Path(pdf_path).stem or "factor_task"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return export_root / f"{timestamp}__{safe_name(pdf_stem)}"


def export_run_outputs(result: Dict[str, Any], run_dir: Path) -> List[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    exported_files: List[Path] = []

    decision_path = run_dir / "decision.json"
    write_json(decision_path, result)
    exported_files.append(decision_path)

    manifest, factor_files = export_confirmed_factors(result=result, run_dir=run_dir)
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    exported_files.append(manifest_path)
    exported_files.extend(factor_files)

    if result.get("iteration_history"):
        iteration_path = run_dir / "iteration_history.json"
        write_json(iteration_path, result["iteration_history"])
        exported_files.append(iteration_path)

    if result.get("issue_report") is not None:
        issue_path = run_dir / "issue_report.json"
        write_json(issue_path, result["issue_report"])
        exported_files.append(issue_path)

    return exported_files


def export_confirmed_factors(result: Dict[str, Any], run_dir: Path) -> tuple[Dict[str, Any], List[Path]]:
    factors_dir = run_dir / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)

    factor_records: List[Dict[str, Any]] = []
    exported_files: List[Path] = []

    for factor in result.get("final_factors", []):
        factor_dir = factors_dir / f"{int(factor['factor_index']):02d}__{safe_name(str(factor.get('factor_name') or 'factor'))}"
        factor_dir.mkdir(parents=True, exist_ok=True)

        factor_df = load_factor_dataframe(factor["factor_value_path"])
        long_df = factor_df.reset_index()
        wide_df = factor_df.reset_index().pivot(index="Trddt", columns="Stkcd", values=factor_df.columns[-1]).sort_index()

        factor_long_path = factor_dir / "factor_values_long.parquet"
        factor_wide_path = factor_dir / "factor_values_wide.parquet"
        long_df.to_parquet(factor_long_path, index=False)
        wide_df.to_parquet(factor_wide_path)

        exported_files.extend([factor_long_path, factor_wide_path])

        backtest_export_path = None
        if factor.get("backtest_path"):
            backtest_df = load_backtest_series(factor["backtest_path"])
            backtest_export_path = factor_dir / "backtest.parquet"
            backtest_df.to_parquet(backtest_export_path, index=False)
            exported_files.append(backtest_export_path)

        metadata = {
            "factor_index": factor.get("factor_index"),
            "factor_name": factor.get("factor_name"),
            "expression": factor.get("expression"),
            "core_logic": factor.get("core_logic"),
            "data_source": factor.get("data_source"),
            "source_models": factor.get("source_models"),
            "expression_consensus": factor.get("expression_consensus"),
            "name_consensus": factor.get("name_consensus"),
            "storage": {
                "factor_values_long_parquet": str(factor_long_path),
                "factor_values_wide_parquet": str(factor_wide_path),
                "backtest_parquet": str(backtest_export_path) if backtest_export_path else None,
            },
        }
        metadata_path = factor_dir / "metadata.json"
        write_json(metadata_path, metadata)
        exported_files.append(metadata_path)

        factor_records.append(
            {
                "factor_index": factor.get("factor_index"),
                "factor_name": factor.get("factor_name"),
                "expression": factor.get("expression"),
                "core_logic": factor.get("core_logic"),
                "data_source": factor.get("data_source"),
                "source_models": factor.get("source_models"),
                "expression_consensus": factor.get("expression_consensus"),
                "name_consensus": factor.get("name_consensus"),
                "factor_values_long_parquet": str(factor_long_path),
                "factor_values_wide_parquet": str(factor_wide_path),
                "backtest_parquet": str(backtest_export_path) if backtest_export_path else None,
            }
        )

    factor_table_path = run_dir / "confirmed_factors.parquet"
    pd.DataFrame(
        factor_records,
        columns=[
            "factor_index",
            "factor_name",
            "expression",
            "core_logic",
            "data_source",
            "source_models",
            "expression_consensus",
            "name_consensus",
            "factor_values_long_parquet",
            "factor_values_wide_parquet",
            "backtest_parquet",
        ],
    ).to_parquet(factor_table_path, index=False)
    exported_files.append(factor_table_path)

    manifest = {
        "ts": result.get("ts"),
        "pdf_path": result.get("pdf_path"),
        "query": result.get("query"),
        "decision": result.get("decision"),
        "consistent": result.get("consistent"),
        "models": result.get("models"),
        "judge_iteration": result.get("judge_iteration"),
        "factor_count": len(factor_records),
        "model_reports": result.get("model_reports"),
        "confirmed_factors_table": str(factor_table_path),
        "factors": factor_records,
    }
    return manifest, exported_files


def load_factor_dataframe(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Factor dataframe file does not exist: {path}")
    return pd.read_pickle(path)


def load_backtest_series(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Backtest file does not exist: {path}")

    backtest = pd.read_csv(path, index_col=0)
    backtest.index.name = "Trddt"
    return backtest.reset_index()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": list(value.shape),
            "columns": [str(col) for col in value.columns],
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "length": int(len(value)),
            "name": str(value.name),
        }
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return repr(value)


def safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return cleaned.strip("_")[:80] or "factor"


def print_run_summary(run_dir: Path, result: Dict[str, Any], exported_files: List[Path]) -> None:
    print("Run complete.")
    print(f"Run directory: {run_dir}")
    print(f"Decision: {result.get('decision')}")
    print(f"Consistent: {result.get('consistent')}")
    print(f"Confirmed factor count: {len(result.get('final_factors', []))}")
    print("Exported files:")
    for path in exported_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
