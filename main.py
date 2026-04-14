from agents.FactorConstructAgent import FCA
from agents.KnowledgeExtractAgent import KEA


def run_pipeline(pdf_path: str, query: str, model: str = "DeepSeek-V3.2", max_rounds: int = 5,
):
    """
    KEA -> FCA pipeline with retry feedback loop.

    Returns on success:
      - factor_name
      - instruction (raw KEA JSON output)
      - df_factor (factor panel from FCA)
      - backtest (IC series)
    """
    kea = KEA()
    fca = FCA(parquet_path="/data/stock_data.parquet")

    feedback = None
    last_instruction = None
    last_fca_result = None

    for _ in range(max_rounds):
        instruction = kea.extract_knowledge(
            pdf_path=pdf_path,
            query=query,
            model=model,
            feedback=feedback,
            max_retries=5,
        )
        last_instruction = instruction

        fca_result = fca.handle_instruction(instruction)
        last_fca_result = fca_result

        if isinstance(fca_result, dict) and fca_result.get("ok") is True:
            df_factors = fca_result["df_factors"]

            # Determine factor names for downstream backtest.
            if isinstance(instruction, list):
                factors = instruction
            else:
                factors = []

            if not factors:
                return {
                    "ok": False,
                    "error": "No factor object found in instruction.",
                    "instruction": instruction,
                }

            # If multiple factors -> return list of per-factor results.
            outputs = []
            for i, item in enumerate(factors):
                factor_name = item.get("factor_name", f"factor_{i}")
                print(df_factors[i])
                backtest_result = fca.backtest(factor_name, df_factors[i])
                outputs.append(
                    {
                        "factor_name": factor_name,
                        "instruction": item,
                        "df_factor": df_factors[i],
                        "backtest": backtest_result,
                    }
                )

            return {"ok": True, "results": outputs}

        if isinstance(fca_result, dict) and fca_result.get("no_factor") is True:
            return {"ok": True, "message": "KEA returned no_factor.", "instruction": instruction}

        # FCA failed: pass feedback to the next KEA call.
        if isinstance(fca_result, dict) and fca_result.get("ok") is False:
            feedback = fca_result.get("feedback")
        else:
            feedback = str(fca_result)

    return {
        "ok": False,
        "error": f"Failed after {max_rounds} rounds.",
        "last_instruction": last_instruction,
        "last_fca_result": last_fca_result,
    }


def main():
    result = run_pipeline(
        pdf_path="data/sample1.pdf",
        query="construct a factor",
        model="DeepSeek-V3.2",
        max_rounds=2,
    )
    print(result)


if __name__ == "__main__":
    main()

