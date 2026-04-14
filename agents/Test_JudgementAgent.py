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
from utils.interpreter import parse_expression_to_node
from utils.tools import call_llm_api, rag_search, read_pdf


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class JudgementAgent:
    """
    Compare factor outputs produced by multiple LLMs under the same data context.

    Core flow:
        shared PDF/RAG context
            -> multi-model KEA generation
            -> per-model FCA execution
            -> value-level consistency judgement
            -> save confirmed factors or mistakes history

    Notes:
    1. This implementation reuses the existing KEA prompt and FCA executor.
    2. It judges consistency primarily by factor values, not expression strings,
       because different expressions may still be semantically equivalent.
    3. Multi-factor instructions are compared by factor index.
       This fits the current repo's default usage well. If later you need
       unordered matching, you can add a matching layer on top.
    """

    def __init__(
        self,
        parquet_path: str = "/data/stock_data.parquet",
        judge_models: Optional[Sequence[str]] = None,
        top_k: int = 5,
        numeric_tolerance: float = 1e-8,
        output_dir: str = "judgement_outputs",
        persist_confirmed: bool = True,
        persist_mistakes: bool = True,
    ) -> None:
        self.kea = KEA()
        self.fca = FCA(parquet_path=parquet_path)
        self.judge_models = list(judge_models or ["DeepSeek-V3.2", "Qwen3.5-27B", "GLM-5"])
        self.top_k = top_k
        self.numeric_tolerance = float(numeric_tolerance)
        self.persist_confirmed = persist_confirmed
        self.persist_mistakes = persist_mistakes

        self.project_root = Path(__file__).resolve().parents[1]
        self.embedding_model_path = Path(self.kea.embedding_model_path)
        self.output_dir = self._resolve_project_path(output_dir)
        self.factor_store_dir = self.output_dir / "confirmed_factors"
        self.backtest_store_dir = self.output_dir / "backtests"
        self.factor_store_dir.mkdir(parents=True, exist_ok=True)
        self.backtest_store_dir.mkdir(parents=True, exist_ok=True)

        self.confirmed_history_path = self.output_dir / "confirmed_history.jsonl"
        self.mistakes_history_path = self.output_dir / "mistakes_history.jsonl"


    def run(
        self,
        pdf_path: str,
        query: str,
        models: Optional[Sequence[str]] = None,
        max_retries: int = 5,
        save_backtest: bool = True,
    ) -> Dict[str, Any]:
        active_models = list(models or self.judge_models)
        if len(active_models) < 2:
            raise ValueError("JudgementAgent requires at least 2 models for comparison.")

        shared_context = self.prepare_shared_context(pdf_path=pdf_path, query=query)
        branches = [
            self._run_single_branch(
                model=model,
                shared_context=shared_context,
                max_retries=max_retries,
                save_backtest=save_backtest,
            )
            for model in active_models
        ]

        decision = self._judge_branches(
            branches=branches,
            pdf_path=pdf_path,
            query=query,
            rag_context=shared_context["rag_context"],
            save_backtest=save_backtest,
        )

        if decision.get("consistent") and self.persist_confirmed:
            self._persist_confirmed_result(decision)
        elif not decision.get("consistent") and self.persist_mistakes:
            self._persist_mistake(decision)

        return decision

    def prepare_shared_context(self, pdf_path: str, query: str) -> Dict[str, Any]:
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

    def _run_single_branch(
        self,
        model: str,
        shared_context: Dict[str, Any],
        max_retries: int,
        save_backtest: bool,
    ) -> Dict[str, Any]:
        user_content = self._build_user_content(shared_context["rag_context"], feedback=None)
        branch: Dict[str, Any] = {
            "model": model,
            "status": "unknown",
            "instruction": None,
            "fca_result": None,
            "factor_results": [],
            "error": None,
        }

        instruction = None
        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                raw = call_llm_api(
                    model=model,
                    system_content=self.kea.prompt["system_content"],
                    assistant_content=self.kea.prompt["assistant_content"],
                    user_content=user_content,
                )
                instruction = self._parse_llm_json(raw)
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                user_content += f"\n\nAvoid the mistakes: Error in attempt {attempt + 1}: {last_error}"

        if instruction is None:
            branch["status"] = "kea_failed"
            branch["error"] = last_error or "Unknown KEA failure"
            record_error_event(
                stage="judgement.branch.kea_failed",
                error=branch["error"],
                current_output=None,
                extra={"model": model, "query": shared_context["query"], "pdf_path": shared_context["pdf_path"]},
            )
            return branch

        branch["instruction"] = instruction

        if isinstance(instruction, dict) and instruction.get("no_factor") is True:
            branch["status"] = "no_factor"
            return branch

        try:
            fca_result = self.fca.handle_instruction(instruction)
            branch["fca_result"] = fca_result
        except Exception as e:
            branch["status"] = "fca_failed"
            branch["error"] = f"{type(e).__name__}: {e}"
            record_error_event(
                stage="judgement.branch.fca_exception",
                error=e,
                current_output=instruction,
                extra={"model": model},
            )
            return branch

        if not isinstance(fca_result, dict):
            branch["status"] = "fca_failed"
            branch["error"] = f"Unexpected FCA result type: {type(fca_result).__name__}"
            return branch

        if fca_result.get("ok") is not True:
            if fca_result.get("no_factor") is True:
                branch["status"] = "no_factor"
            else:
                branch["status"] = "fca_failed"
                branch["error"] = fca_result.get("error", "Unknown FCA failure")
            return branch

        factor_results = self._materialize_factor_results(
            instruction=instruction,
            df_factors=fca_result.get("df_factors", []),
            model=model,
            save_backtest=save_backtest,
        )

        branch["status"] = "ok"
        branch["factor_results"] = factor_results
        return branch

    def _materialize_factor_results(
        self,
        instruction: Any,
        df_factors: List[pd.DataFrame],
        model: str,
        save_backtest: bool,
    ) -> List[Dict[str, Any]]:
        factors = self._normalize_instruction_to_list(instruction)
        results: List[Dict[str, Any]] = []

        for i, factor in enumerate(factors):
            factor_name = factor.get("factor_name", f"factor_{i}")
            expression = factor.get("expression", "")
            df_factor = df_factors[i] if i < len(df_factors) else None
            diagnostics = self._diagnose_expression(expression)

            backtest = None
            if save_backtest and df_factor is not None:
                try:
                    backtest = self.fca.backtest(factor_name, df_factor)
                except Exception as e:
                    backtest = None
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
                    "expression": expression,
                    "core_logic": factor.get("core_logic"),
                    "data_source": factor.get("data_source"),
                    "df_factor": df_factor,
                    "backtest": backtest,
                    "diagnostics": diagnostics,
                }
            )
        return results

    def _judge_branches(
        self,
        branches: List[Dict[str, Any]],
        pdf_path: str,
        query: str,
        rag_context: List[str],
        save_backtest: bool,
    ) -> Dict[str, Any]:
        timestamp = self._now_ts()
        branch_statuses = {branch["model"]: branch["status"] for branch in branches}
        ok_branches = [b for b in branches if b.get("status") == "ok"]
        no_factor_branches = [b for b in branches if b.get("status") == "no_factor"]
        failed_branches = [b for b in branches if b.get("status") not in {"ok", "no_factor"}]

        decision: Dict[str, Any] = {
            "ts": timestamp,
            "agent": "JudgementAgent",
            "pdf_path": pdf_path,
            "query": query,
            "models": [b["model"] for b in branches],
            "branch_statuses": branch_statuses,
            "branches": self._branch_summary(branches),
            "rag_context": rag_context,
            "consistent": False,
            "confirmed_factors": [],
            "mistakes_history": [],
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
                    "decision": "branch_failure",
                    "mistakes_history": self._build_failure_history(branches),
                }
            )
            return decision

        if no_factor_branches and ok_branches:
            decision.update(
                {
                    "decision": "no_factor_conflict",
                    "mistakes_history": self._build_failure_history(branches),
                }
            )
            return decision

        factor_counts = [len(b["factor_results"]) for b in ok_branches]
        if len(set(factor_counts)) != 1:
            decision.update(
                {
                    "decision": "factor_count_mismatch",
                    "mistakes_history": self._build_factor_count_mismatch_history(ok_branches),
                }
            )
            return decision

        if not factor_counts or factor_counts[0] == 0:
            decision.update(
                {
                    "decision": "empty_factor_list",
                    "mistakes_history": self._build_failure_history(branches),
                }
            )
            return decision

        pairwise_reports: List[Dict[str, Any]] = []
        confirmed_factors: List[Dict[str, Any]] = []
        all_consistent = True

        factor_count = factor_counts[0]
        for factor_index in range(factor_count):
            per_model = [branch["factor_results"][factor_index] for branch in ok_branches]
            compare_report = self._compare_factor_group(per_model, ok_branches)
            pairwise_reports.append(compare_report)

            if compare_report["value_consistent"]:
                confirmed_factors.append(
                    self._build_confirmed_factor_record(
                        factor_index=factor_index,
                        per_model=per_model,
                        compare_report=compare_report,
                        save_backtest=save_backtest,
                    )
                )
            else:
                all_consistent = False

        decision["pairwise_reports"] = pairwise_reports

        if all_consistent:
            decision.update(
                {
                    "consistent": True,
                    "decision": "consistent_factor_values",
                    "confirmed_factors": confirmed_factors,
                }
            )
            return decision

        decision.update(
            {
                "decision": "factor_value_mismatch",
                "mistakes_history": self._build_value_mismatch_history(ok_branches, pairwise_reports),
            }
        )
        return decision

    def _compare_factor_group(
        self,
        per_model: List[Dict[str, Any]],
        ok_branches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base = per_model[0]
        base_df = base["df_factor"]

        expression_counter = Counter(item.get("expression", "") for item in per_model)
        name_counter = Counter(item.get("factor_name", "") for item in per_model)

        pair_details: List[Dict[str, Any]] = []
        value_consistent = True
        all_same_expression = len(expression_counter) == 1
        all_same_name = len(name_counter) == 1

        for branch, item in zip(ok_branches[1:], per_model[1:]):
            same, stats = self._factor_values_equal(base_df, item["df_factor"])
            if not same:
                value_consistent = False
            pair_details.append(
                {
                    "base_model": ok_branches[0]["model"],
                    "compare_model": branch["model"],
                    "same_values": same,
                    **stats,
                }
            )

        return {
            "factor_index": base["factor_index"],
            "value_consistent": value_consistent,
            "all_same_expression": all_same_expression,
            "all_same_name": all_same_name,
            "expressions": {branch["model"]: item.get("expression") for branch, item in zip(ok_branches, per_model)},
            "factor_names": {branch["model"]: item.get("factor_name") for branch, item in zip(ok_branches, per_model)},
            "pair_details": pair_details,
        }

    def _build_confirmed_factor_record(
        self,
        factor_index: int,
        per_model: List[Dict[str, Any]],
        compare_report: Dict[str, Any],
        save_backtest: bool,
    ) -> Dict[str, Any]:
        canonical = per_model[0]
        factor_name = self._majority_vote([item.get("factor_name") for item in per_model], default=canonical["factor_name"])
        expression = self._majority_vote([item.get("expression") for item in per_model], default=canonical["expression"])
        core_logic = self._majority_vote([item.get("core_logic") for item in per_model], default=canonical.get("core_logic"))
        data_source = canonical.get("data_source")

        factor_file = self.factor_store_dir / f"{self._now_compact()}__factor_{factor_index}__{self._safe_name(factor_name)}.pkl"
        canonical["df_factor"].to_pickle(factor_file)

        backtest_path = None
        if save_backtest and canonical.get("backtest") is not None:
            backtest_path = self.backtest_store_dir / f"{self._now_compact()}__factor_{factor_index}__{self._safe_name(factor_name)}.csv"
            canonical["backtest"].to_csv(backtest_path)

        return {
            "factor_index": factor_index,
            "factor_name": factor_name,
            "expression": expression,
            "core_logic": core_logic,
            "data_source": data_source,
            "source_models": [item.get("model") for item in per_model],
            "model_expressions": [item.get("expression") for item in per_model],
            "model_factor_names": [item.get("factor_name") for item in per_model],
            "expression_consensus": compare_report["all_same_expression"],
            "name_consensus": compare_report["all_same_name"],
            "factor_value_path": str(factor_file),
            "backtest_path": str(backtest_path) if backtest_path else None,
            "comparison": compare_report,
        }

    def _build_failure_history(self, branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for branch in branches:
            history.append(
                {
                    "model": branch["model"],
                    "status": branch["status"],
                    "error": branch.get("error"),
                    "instruction": self._safe_jsonable(branch.get("instruction")),
                    "expression_diagnostics": self._branch_expression_diagnostics(branch),
                }
            )
        return history

    def _build_factor_count_mismatch_history(self, ok_branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for branch in ok_branches:
            history.append(
                {
                    "reason": "factor_count_mismatch",
                    "model": branch["model"],
                    "factor_count": len(branch["factor_results"]),
                    "factor_names": [x["factor_name"] for x in branch["factor_results"]],
                    "expressions": [x["expression"] for x in branch["factor_results"]],
                    "expression_diagnostics": self._branch_expression_diagnostics(branch),
                }
            )
        return history

    def _build_value_mismatch_history(
        self,
        ok_branches: List[Dict[str, Any]],
        pairwise_reports: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        branch_level = self._build_failure_history(ok_branches)
        return branch_level + [
            {
                "reason": "factor_value_mismatch",
                "pairwise_reports": pairwise_reports,
            }
        ]

    def _branch_expression_diagnostics(self, branch: Dict[str, Any]) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for item in branch.get("factor_results", []):
            diag = dict(item.get("diagnostics", {}))
            diag["factor_name"] = item.get("factor_name")
            diag["expression"] = item.get("expression")
            diagnostics.append(diag)

        if not diagnostics:
            normalized = self._normalize_instruction_to_list(branch.get("instruction"))
            for i, item in enumerate(normalized):
                diagnostics.append(
                    {
                        "factor_index": i,
                        **self._diagnose_expression(item.get("expression", "")),
                    }
                )
        return diagnostics

    def _diagnose_expression(self, expression: str) -> Dict[str, Any]:
        node_or_err = parse_expression_to_node(expression)
        if isinstance(node_or_err, dict):
            fields, operators = self._collect_node_usage(node_or_err)
            return {
                "parse_ok": True,
                "errors": [],
                "fields": sorted(fields),
                "operators": sorted(operators),
            }

        return {
            "parse_ok": False,
            "errors": list(getattr(node_or_err, "errors", [])),
            "fields": [],
            "operators": [],
        }

    def _collect_node_usage(self, node: Dict[str, Any]) -> tuple[set[str], set[str]]:
        fields: set[str] = set()
        operators: set[str] = set()

        def walk(n: Any) -> None:
            if not isinstance(n, dict):
                return
            op = n.get("op")
            if op == "field":
                field = n.get("field")
                if isinstance(field, str):
                    fields.add(field)
                return
            if isinstance(op, str):
                operators.add(op)
            for value in n.values():
                if isinstance(value, dict):
                    walk(value)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)

        walk(node)
        return fields, operators

    def _factor_values_equal(self, left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, Dict[str, Any]]:
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
        payload = {
            "ts": decision["ts"],
            "decision": decision.get("decision"),
            "pdf_path": decision.get("pdf_path"),
            "query": decision.get("query"),
            "models": decision.get("models"),
            "branch_statuses": decision.get("branch_statuses"),
            "confirmed_factors": decision.get("confirmed_factors"),
        }
        self._append_jsonl(self.confirmed_history_path, payload)

    def _persist_mistake(self, decision: Dict[str, Any]) -> None:
        payload = {
            "ts": decision["ts"],
            "decision": decision.get("decision"),
            "pdf_path": decision.get("pdf_path"),
            "query": decision.get("query"),
            "models": decision.get("models"),
            "branch_statuses": decision.get("branch_statuses"),
            "mistakes_history": decision.get("mistakes_history"),
            "branches": decision.get("branches"),
        }
        self._append_jsonl(self.mistakes_history_path, payload)

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
            return [x for x in instruction if isinstance(x, dict)]
        return []

    def _branch_summary(self, branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for branch in branches:
            summary.append(
                {
                    "model": branch["model"],
                    "status": branch["status"],
                    "error": branch.get("error"),
                    "factor_count": len(branch.get("factor_results", [])),
                    "factor_names": [x.get("factor_name") for x in branch.get("factor_results", [])],
                    "expressions": [x.get("expression") for x in branch.get("factor_results", [])],
                }
            )
        return summary

    def _resolve_project_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.project_root / path

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
        cleaned = [v for v in values if v not in (None, "")]
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


if __name__ == "__main__":
    agent = JudgementAgent(
        parquet_path="/data/stock_data.parquet",
        judge_models=["DeepSeek-V3.2", "Qwen3.5-27B", "GLM-5"],
        top_k=5,
        numeric_tolerance=1e-8,
    )

    result = agent.run(
        pdf_path="data/sample1.pdf",
        query="construct a factor",
        max_retries=3,
        save_backtest=True,
    )
    print(json.dumps(agent._safe_jsonable(result), ensure_ascii=False, indent=2))
