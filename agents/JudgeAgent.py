from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from agents.FactorConstructAgent import FCA
from agents.KnowledgeExtractAgent import KEA
from utils.error_utils import record_error_event
from utils.tools import call_llm_api, rag_search, read_pdf


DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class JA:
    def __init__(
        self,
        parquet_path: str = "/data/stock_data.parquet",
        model_list: Optional[Sequence[str]] = None,
        numeric_tolerance: float = 1e-5,
        top_k: int = 5,
        max_judge_iterations: int = 3,
    ) -> None:
        self.parquet_path = parquet_path
        self.model_list = list(model_list or ["DeepSeek-V3.2", "Qwen3.5-27B", "GLM-5", "MiniMax-2.5"])
        self.numeric_tolerance = float(numeric_tolerance)
        self.top_k = int(top_k)
        self.max_judge_iterations = int(max_judge_iterations)

        project_root = Path(__file__).resolve().parents[1]
        self.project_root = project_root
        self.embedding_model_path = project_root / "hub" / "models--BAAI--bge-m3"
        self.kea = KEA()

        self.output_dir = project_root / "judgement_output"
        self.factor_store_dir = self.output_dir / "confirmed_factors"
        self.backtest_store_dir = self.output_dir / "backtests"
        self.factor_store_dir.mkdir(parents=True, exist_ok=True)
        self.backtest_store_dir.mkdir(parents=True, exist_ok=True)

        self.confirmed_history_path = self.output_dir / "confirmed_history.jsonl"
        self.mistakes_history_path = self.output_dir / "mistakes_history.jsonl"

    def run_ja(
        self,
        model_list: Sequence[str],
        pdf_path: str,
        query: str,
        max_retires: int = 5,
        max_rounds: int = 2,
        max_judge_iterations: Optional[int] = None,
        save_backtest: bool = True,
    ) -> Dict[str, Any]:
        active_models = list(model_list or self.model_list)
        if len(active_models) < 2:
            raise ValueError("JudgementAgent requires at least 2 models for comparison.")

        shared_context = self.context_sharing(pdf_path=pdf_path, query=query)
        total_iterations = max_judge_iterations or self.max_judge_iterations

        judge_feedback: Optional[str] = None
        iteration_history: List[Dict[str, Any]] = []
        final_decision: Optional[Dict[str, Any]] = None

        for judge_iteration in range(1, total_iterations + 1):
            branches = [
                self.run_single_model(
                    model=model,
                    shared_context=shared_context,
                    max_retries=max_retires,
                    max_rounds=max_rounds,
                    judge_feedback=judge_feedback,
                    save_backtest=save_backtest,
                )
                for model in active_models
            ]

            decision = self.compare_models(
                branches=branches,
                rag_context=shared_context["rag_context"],
                pdf_path=pdf_path,
                query=query,
                save_backtest=save_backtest,
            )
            decision["judge_iteration"] = judge_iteration
            decision["judge_feedback_used"] = judge_feedback
            final_decision = decision

            iteration_history.append(self._build_iteration_snapshot(decision))

            if decision.get("consistent"):
                print(f"Factors constructed from {pdf_path} are consistent after judge iteration {judge_iteration}.")
                break

            judge_feedback = self._build_judge_feedback(decision)
            if judge_iteration < total_iterations:
                print(
                    f"Judge iteration {judge_iteration} found inconsistencies for {pdf_path}; "
                    "feedback has been sent into the next FCA-assisted refinement round."
                )
            else:
                print(
                    f"Factors constructed from {pdf_path} are still inconsistent after "
                    f"{judge_iteration} judge iterations."
                )

        if final_decision is None:
            raise RuntimeError("JudgeAgent failed to produce a final decision.")

        final_decision["iteration_history"] = iteration_history
        final_decision["latest_feedback"] = judge_feedback

        if final_decision.get("consistent"):
            self._persist_confirmed_result(final_decision)
        else:
            self._persist_mistake(final_decision)

        return final_decision

    def context_sharing(self, pdf_path: str, query: str) -> Dict[str, Any]:
        """
        Prepare one shared RAG context and reuse it across all judge iterations.
        """
        try:
            file_text = read_pdf(pdf_path)
            embedding_model = SentenceTransformer(str(self.embedding_model_path), device=DEVICE)
            rag_context = rag_search(file_text, query, embedding_model, top_k=self.top_k)
        except Exception as e:
            record_error_event(
                stage="judgement.prepare_shared_context",
                error=e,
                current_output=None,
                extra={"pdf_path": pdf_path, "query": query},
            )
            raise
        finally:
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return {
            "pdf_path": pdf_path,
            "query": query,
            "file_text": file_text,
            "rag_context": rag_context,
        }

    def _context_to_instruction(
        self,
        model: str,
        feedback: Optional[str],
        shared_context: Dict[str, Any],
        max_retries: int = 5,
    ) -> Any:
        user_content = self._build_user_content(shared_context["rag_context"], feedback)
        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                raw = call_llm_api(
                    model=model,
                    system_content=self.kea.prompt["system_content"],
                    assistant_content=self.kea.prompt["assistant_content"],
                    user_content=user_content,
                )
                return self._parse_llm_json(raw)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"[Attempt {attempt + 1}] output handling failed for model {model}: {last_error}")
                user_content += f"\n\nAvoid the mistakes: Error in attempt {attempt + 1}: {last_error}"

        record_error_event(
            stage="judgement.call_llm_api",
            error=last_error or "Unknown LLM failure",
            current_output=None,
            extra={"model": model, "pdf_path": shared_context["pdf_path"], "query": shared_context["query"]},
        )
        return None

    def run_single_model(
        self,
        model: str,
        shared_context: Dict[str, Any],
        max_retries: int = 5,
        max_rounds: int = 2,
        judge_feedback: Optional[str] = None,
        save_backtest: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate -> execute -> backtest one branch. When FCA or judge feedback exists,
        feed it back into the next generation round to tighten convergence.
        """
        result: Dict[str, Any] = {
            "ok": False,
            "model": model,
            "status": "unknown",
            "shared_context": shared_context,
            "instruction": None,
            "factor_results": [],
            "error": None,
            "judge_feedback": judge_feedback,
        }

        feedback_parts: List[str] = [judge_feedback] if judge_feedback else []
        last_instruction: Any = None
        last_fca_result: Any = None

        for round_idx in range(max_rounds):
            merged_feedback = "\n\n".join(part for part in feedback_parts if part)
            instruction = self._context_to_instruction(
                model=model,
                shared_context=shared_context,
                max_retries=max_retries,
                feedback=merged_feedback or None,
            )
            last_instruction = instruction
            result["instruction"] = instruction

            if instruction is None:
                result["status"] = "kea_failed"
                result["error"] = "KEA failed to produce valid JSON output."
                continue

            if isinstance(instruction, dict) and instruction.get("no_factor") is True:
                result["status"] = "no_factor"
                result["no_factor"] = True
                result["error"] = "KEA returned no_factor"
                return result

            try:
                fca = FCA(parquet_path=self.parquet_path)
                fca_result = fca.handle_instruction(instruction)
                last_fca_result = fca_result
            except Exception as e:
                error_text = f"FCA failed: {type(e).__name__}: {e}"
                feedback_parts.append(error_text)
                result["status"] = "fca_failed"
                result["error"] = error_text
                if round_idx == max_rounds - 1:
                    result["last_instruction"] = last_instruction
                    result["last_fca_result"] = last_fca_result
                    return result
                continue

            if not isinstance(fca_result, dict):
                result["status"] = "fca_failed"
                result["error"] = f"Unexpected FCA result type: {type(fca_result).__name__}"
                result["last_instruction"] = last_instruction
                result["last_fca_result"] = last_fca_result
                return result

            if fca_result.get("ok") is not True:
                if fca_result.get("no_factor") is True:
                    result["status"] = "no_factor"
                    result["no_factor"] = True
                    result["error"] = "KEA returned no_factor"
                    return result

                fca_feedback = fca_result.get("feedback") or fca_result.get("error") or "FCA failed"
                feedback_parts.append(str(fca_feedback))
                result["status"] = "fca_failed"
                result["error"] = str(fca_feedback)
                if round_idx == max_rounds - 1:
                    result["last_instruction"] = last_instruction
                    result["last_fca_result"] = last_fca_result
                    return result
                continue

            factor_results = self._materialize_factor_results(
                instruction=instruction,
                df_factors=fca_result.get("df_factors", []),
                model=model,
                save_backtest=save_backtest,
            )
            if not factor_results:
                result["status"] = "no_factor"
                result["no_factor"] = True
                result["error"] = "No executable factor remained after FCA."
                return result

            result["ok"] = True
            result["status"] = "ok"
            result["factor_results"] = factor_results
            result["last_instruction"] = last_instruction
            result["last_fca_result"] = last_fca_result
            return result

        result["status"] = "fca_failed"
        result["error"] = result.get("error") or f"Failed after {max_rounds} rounds."
        result["last_instruction"] = last_instruction
        result["last_fca_result"] = last_fca_result
        return result

    def _materialize_factor_results(
        self,
        instruction: Any,
        df_factors: List[pd.DataFrame],
        model: str,
        save_backtest: bool,
    ) -> List[Dict[str, Any]]:
        factors = self._normalize_instruction_to_list(instruction)
        results: List[Dict[str, Any]] = []

        for i, item in enumerate(factors):
            factor_name = item.get("factor_name", f"factor_{i}")
            expression = item.get("expression", "")
            df_factor = df_factors[i] if i < len(df_factors) else None
            backtest_result = None

            if save_backtest and df_factor is not None:
                try:
                    backtest_result = FCA(parquet_path=self.parquet_path).backtest(factor_name, df_factor)
                except Exception as e:
                    backtest_result = None
                    record_error_event(
                        stage="judgement.branch.backtest_failed",
                        error=e,
                        current_output=expression,
                        extra={"model": model, "factor_name": factor_name},
                    )

            results.append(
                {
                    "model": model,
                    "factor_index": i,
                    "factor_name": factor_name,
                    "instruction": item,
                    "expression": expression,
                    "core_logic": item.get("core_logic"),
                    "data_source": item.get("data_source"),
                    "df_factor": df_factor,
                    "backtest": backtest_result,
                }
            )

        return results

    def compare_models(
        self,
        branches: List[Dict[str, Any]],
        rag_context: List[str],
        pdf_path: str,
        query: str,
        save_backtest: bool = True,
    ) -> Dict[str, Any]:
        timestamp = self._now_ts()
        ok_branches = [b for b in branches if b.get("status") == "ok"]
        no_factor_branches = [b for b in branches if b.get("status") == "no_factor"]
        failed_branches = [b for b in branches if b.get("status") not in {"ok", "no_factor"}]
        model_reports = self._build_model_reports(branches)

        decision: Dict[str, Any] = {
            "ts": timestamp,
            "agent": "JudgeAgent",
            "pdf_path": pdf_path,
            "query": query,
            "models": [b["model"] for b in branches],
            "model_reports": model_reports,
            "consistent": False,
            "final_factors": [],
            "issue_report": None,
        }

        if len(no_factor_branches) == len(branches):
            decision.update(
                {
                    "consistent": True,
                    "decision": "all_models_returned_no_factor",
                    "message": "All judge models agreed that no valid factor could be constructed.",
                }
            )
            return decision

        if failed_branches:
            decision.update(
                {
                    "decision": "models_failure",
                    "issue_report": self._build_failure_issue(branches=branches),
                }
            )
            return decision

        if no_factor_branches and ok_branches:
            decision.update(
                {
                    "decision": "no_factor_conflict",
                    "issue_report": self._build_no_factor_conflict_issue(branches=branches),
                }
            )
            return decision

        factor_counts = [len(b["factor_results"]) for b in ok_branches]
        if len(set(factor_counts)) != 1:
            decision.update(
                {
                    "decision": "factor_count_mismatch",
                    "issue_report": self._build_factor_count_mismatch_issue(ok_branches),
                }
            )
            return decision

        if not factor_counts or factor_counts[0] == 0:
            decision.update(
                {
                    "decision": "empty_factor_list",
                    "issue_report": self._build_empty_factor_issue(branches=branches),
                }
            )
            return decision

        confirmed_factors: List[Dict[str, Any]] = []
        all_consistent = True
        mismatch_by_factor: Dict[str, Dict[str, Any]] = {}

        for factor_index in range(factor_counts[0]):
            model_factor = [branch["factor_results"][factor_index] for branch in ok_branches]
            factor_comparison = self._compare_model_factor(model_factor=model_factor, ok_branches=ok_branches)

            if factor_comparison["value_consist"]:
                confirmed_factors.append(
                    self._build_confirmed_factor_record(
                        factor_index=factor_index,
                        model_factor=model_factor,
                        factor_comparison=factor_comparison,
                        save_backtest=save_backtest,
                    )
                )
            else:
                all_consistent = False
                mismatch_by_factor[str(factor_index)] = self._build_factor_mismatch_entry(factor_comparison)

        decision["final_factors"] = confirmed_factors

        if all_consistent:
            decision.update(
                {
                    "consistent": True,
                    "decision": "consistent_factor_values",
                }
            )
            return decision

        decision.update(
            {
                "decision": "factor_value_mismatch",
                "issue_report": self._build_value_mismatch_issue(mismatch_by_factor),
            }
        )
        return decision

    def _compare_model_factor(
        self,
        model_factor: List[Dict[str, Any]],
        ok_branches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base = model_factor[0]
        base_df_factor = base["df_factor"]
        count_names = Counter(item.get("factor_name") for item in model_factor)
        count_expressions = Counter(item.get("expression") for item in model_factor)

        pair_result: List[Dict[str, Any]] = []
        value_consist = True
        all_same_expression = len(count_expressions) == 1
        all_same_name = len(count_names) == 1
        final_name = count_names.most_common(1)[0][0] if count_names else None
        final_expression = count_expressions.most_common(1)[0][0] if count_expressions else None

        for item, branch in zip(model_factor[1:], ok_branches[1:]):
            same, result = self._compare_two_df_factors(left=base_df_factor, right=item["df_factor"])
            if not same:
                value_consist = False
            pair_result.append(
                {
                    "base_model": ok_branches[0]["model"],
                    "compare_model": branch["model"],
                    "value_consist": same,
                    **result,
                }
            )

        return {
            "factor_index": base.get("factor_index"),
            "base_model": base.get("model"),
            "base_df_factor": base_df_factor,
            "value_consist": value_consist,
            "all_same_expression": all_same_expression,
            "all_same_name": all_same_name,
            "factor_name": final_name,
            "expression": final_expression,
            "factor_names": {branch["model"]: item.get("factor_name") for branch, item in zip(ok_branches, model_factor)},
            "expressions": {branch["model"]: item.get("expression") for branch, item in zip(ok_branches, model_factor)},
            "pair_result": pair_result,
        }

    def _build_confirmed_factor_record(
        self,
        factor_index: int,
        model_factor: List[Dict[str, Any]],
        factor_comparison: Dict[str, Any],
        save_backtest: bool,
    ) -> Dict[str, Any]:
        canonical = model_factor[0]
        factor_name = self._majority_vote(
            [item.get("factor_name") for item in model_factor],
            default=canonical.get("factor_name"),
        )
        expression = self._majority_vote(
            [item.get("expression") for item in model_factor],
            default=canonical.get("expression"),
        )
        core_logic = self._majority_vote(
            [item.get("core_logic") for item in model_factor],
            default=canonical.get("core_logic"),
        )

        factor_file = self.factor_store_dir / (
            f"{self._now_compact()}__factor_{factor_index}__{self._safe_name(str(factor_name))}.pkl"
        )
        canonical["df_factor"].to_pickle(factor_file)

        backtest_path = None
        if save_backtest and canonical.get("backtest") is not None:
            backtest_path = self.backtest_store_dir / (
                f"{self._now_compact()}__factor_{factor_index}__{self._safe_name(str(factor_name))}.csv"
            )
            canonical["backtest"].to_csv(backtest_path)

        return {
            "factor_index": factor_index,
            "factor_name": factor_name,
            "expression": expression,
            "core_logic": core_logic,
            "data_source": canonical.get("data_source"),
            "source_models": [item.get("model") for item in model_factor],
            "expression_consensus": factor_comparison.get("all_same_expression"),
            "name_consensus": factor_comparison.get("all_same_name"),
            "factor_value_path": str(factor_file),
            "backtest_path": str(backtest_path) if backtest_path else None,
        }

    def _build_model_reports(self, branches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        reports: Dict[str, Dict[str, Any]] = {}
        for branch in branches:
            reports[str(branch.get("model"))] = {
                "status": branch.get("status"),
                "error": branch.get("error"),
                "factors": self._extract_factor_briefs(branch),
            }
        return reports

    def _extract_factor_briefs(self, branch: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
        factor_results = branch.get("factor_results", [])
        if factor_results:
            return [
                {
                    "factor_name": item.get("factor_name"),
                    "expression": item.get("expression"),
                }
                for item in factor_results
            ]

        return [
            {
                "factor_name": item.get("factor_name"),
                "expression": item.get("expression"),
            }
            for item in self._normalize_instruction_to_list(branch.get("instruction"))
        ]

    def _build_failure_issue(self, branches: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "reason": "models_failure",
            "summary": "At least one model failed before producing executable factors.",
        }

    def _build_no_factor_conflict_issue(self, branches: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "reason": "no_factor_conflict",
            "summary": "Some models returned no_factor while others produced executable factors.",
        }

    def _build_factor_count_mismatch_issue(self, ok_branches: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "reason": "factor_count_mismatch",
            "summary": "Models returned different numbers of final factors.",
        }

    def _build_empty_factor_issue(self, branches: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "reason": "empty_factor_list",
            "summary": "The judge process ended with zero executable factors across successful branches.",
        }

    def _build_value_mismatch_issue(self, mismatch_by_factor: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "reason": "factor_value_mismatch",
            "summary": "Models produced aligned factor counts, but at least one factor value series was inconsistent.",
            "mismatch_factors": mismatch_by_factor,
        }

    def _build_factor_mismatch_entry(self, factor_comparison: Dict[str, Any]) -> Dict[str, Any]:
        comparison_by_model = {
            str(item.get("compare_model")): {
                "value_consist": item.get("value_consist"),
                "same_index": item.get("same_index"),
                "nan_pattern_same": item.get("nan_pattern_same"),
                "left_only_rows": item.get("left_only_rows"),
                "right_only_rows": item.get("right_only_rows"),
                "overlap_rows": item.get("overlap_rows"),
                "max_abs_diff": item.get("max_abs_diff"),
                "tolerance": item.get("tolerance"),
            }
            for item in factor_comparison.get("pair_result", [])
        }
        return {
            "factor_name_by_model": factor_comparison.get("factor_names"),
            "expression_by_model": factor_comparison.get("expressions"),
            "comparison_by_model": comparison_by_model,
        }

    def _build_judge_feedback(self, decision: Dict[str, Any]) -> str:
        issue_report = decision.get("issue_report") or {}
        lines = [
            "Previous judge round was inconsistent. Please regenerate the final factor set with stricter consistency.",
            f"Decision type: {decision.get('decision')}.",
            "Requirements:",
            "1. Return the same number of final tradable factors across models.",
            "2. Do not output intermediate decomposition factors unless they are intended final outputs.",
            "3. Keep factor names and expressions concise, executable, and aligned with the shared context.",
            "4. Prefer one canonical final factor over multiple helper factors when the paper describes a combined signal.",
        ]

        if decision.get("decision") == "factor_count_mismatch":
            for model, report in decision.get("model_reports", {}).items():
                lines.append(
                    f"Model {model} returned {len(report.get('factors', []))} factors: "
                    f"{[item.get('factor_name') for item in report.get('factors', [])]}."
                )
        elif decision.get("decision") == "factor_value_mismatch":
            for factor_index, item in issue_report.get("mismatch_factors", {}).items():
                lines.append(
                    f"Factor index {factor_index} names by model: {item.get('factor_name_by_model')}."
                )
                lines.append(
                    f"Factor index {factor_index} expressions by model: {item.get('expression_by_model')}."
                )
                for compare_model, pair in item.get("comparison_by_model", {}).items():
                    lines.append(
                        f"Compare model {compare_model}: "
                        f"value_consist={pair.get('value_consist')}, max_abs_diff={pair.get('max_abs_diff')}."
                    )
        else:
            for model, report in decision.get("model_reports", {}).items():
                lines.append(
                    f"Model {model}: status={report.get('status')}, error={report.get('error')}."
                )

        return "\n".join(lines)

    def _build_iteration_snapshot(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "judge_iteration": decision.get("judge_iteration"),
            "decision": decision.get("decision"),
            "consistent": decision.get("consistent"),
            "model_statuses": {
                model: report.get("status")
                for model, report in decision.get("model_reports", {}).items()
            },
            "final_factor_count": len(decision.get("final_factors", [])),
            "issue_reason": (decision.get("issue_report") or {}).get("reason"),
            "judge_feedback_used": decision.get("judge_feedback_used"),
        }

    def _compare_two_df_factors(self, left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, Dict[str, Any]]:
        if left is None or right is None:
            return False, {"reason": "missing_factor_dataframe"}

        a = self._normalize_factor_df(left)
        b = self._normalize_factor_df(right)

        left_keys = set(zip(a["Trddt"], a["Stkcd"]))
        right_keys = set(zip(b["Trddt"], b["Stkcd"]))
        same_index = left_keys == right_keys

        merged = a.merge(b, on=["Trddt", "Stkcd"], how="outer", suffixes=("_left", "_right"), indicator=True)
        overlap = merged[merged["_merge"] == "both"].copy()
        left_only = int((merged["_merge"] == "left_only").sum())
        right_only = int((merged["_merge"] == "right_only").sum())

        if overlap.empty:
            return False, {
                "reason": "no_overlap",
                "same_index": same_index,
                "left_only_rows": left_only,
                "right_only_rows": right_only,
            }

        left_nan = overlap["value_left"].isna()
        right_nan = overlap["value_right"].isna()
        nan_pattern_same = bool((left_nan == right_nan).all())

        valid = overlap[~left_nan & ~right_nan].copy()
        if valid.empty:
            max_abs_diff = 0.0
            close_enough = nan_pattern_same and same_index and left_only == 0 and right_only == 0
        else:
            valid["abs_diff"] = (valid["value_left"] - valid["value_right"]).abs()
            max_abs_diff = float(valid["abs_diff"].max())
            close_enough = (
                max_abs_diff <= self.numeric_tolerance
                and nan_pattern_same
                and same_index
                and left_only == 0
                and right_only == 0
            )

        stats = {
            "same_index": same_index,
            "nan_pattern_same": nan_pattern_same,
            "left_only_rows": left_only,
            "right_only_rows": right_only,
            "overlap_rows": int(len(overlap)),
            "max_abs_diff": float(max_abs_diff),
            "tolerance": self.numeric_tolerance,
        }
        return bool(close_enough), stats

    def _normalize_factor_df(self, df_factor: pd.DataFrame) -> pd.DataFrame:
        df = df_factor.copy().reset_index()
        if len(df.columns) < 3:
            raise ValueError("Factor dataframe must contain Trddt, Stkcd and one value column.")

        value_col = df.columns[-1]
        df = df.rename(columns={value_col: "value"})

        required = {"Trddt", "Stkcd", "value"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Factor dataframe missing required columns: {sorted(missing)}")

        return df[["Trddt", "Stkcd", "value"]].sort_values(["Trddt", "Stkcd"]).reset_index(drop=True)

    def _persist_confirmed_result(self, decision: Dict[str, Any]) -> None:
        self._append_jsonl(self.confirmed_history_path, decision)

    def _persist_mistake(self, decision: Dict[str, Any]) -> None:
        self._append_jsonl(self.mistakes_history_path, decision)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        safe_payload = self._safe_jsonable(payload)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")

    def _build_user_content(self, rag_context: List[str], feedback: Optional[str]) -> str:
        user_content = self.kea.prompt["user_content"]
        if feedback:
            user_content += f"\nAvoid the mistakes: {feedback}\n"
        user_content += str(rag_context)
        return user_content

    def _parse_llm_json(self, raw_text: str) -> Any:
        cleaned = raw_text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE)
        return json.loads(cleaned)

    def _normalize_instruction_to_list(self, instruction: Any) -> List[Dict[str, Any]]:
        if instruction is None:
            return []
        if isinstance(instruction, dict):
            if instruction.get("no_factor") is True:
                return []
            return [instruction]
        if isinstance(instruction, list):
            return [item for item in instruction if isinstance(item, dict)]
        return []

    def _safe_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": list(value.shape),
                "columns": [str(x) for x in value.columns],
            }
        if isinstance(value, pd.Series):
            return {
                "type": "Series",
                "length": int(len(value)),
                "name": str(value.name),
            }
        if isinstance(value, dict):
            return {str(k): self._safe_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._safe_jsonable(v) for v in value]
        return repr(value)

    def _majority_vote(self, values: List[Any], default: Any = None) -> Any:
        cleaned = [value for value in values if value not in (None, "")]
        if not cleaned:
            return default
        return Counter(cleaned).most_common(1)[0][0]

    def _safe_name(self, text: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text or "factor")
        return safe.strip("_")[:80] or "factor"

    def _now_ts(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _now_compact(self) -> str:
        return datetime.utcnow().strftime("%Y%m%dT%H%M%S")
