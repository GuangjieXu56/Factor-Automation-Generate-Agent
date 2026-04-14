import json
import torch
from utils.tools import read_pdf, rag_search, call_llm_api
from sentence_transformers import SentenceTransformer
from pathlib import Path
from utils.error_utils import record_error_event


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class KEA:
    def __init__(self):
        # Load allowed fields/operators from JSON files so they can be updated without code changes.
        project_root = Path(__file__).resolve().parents[1]
        fields_path = project_root / "configs" / "fields.json"
        operators_path = project_root / "configs" / "operators.json"
        self.embedding_model_path = project_root / "hub" / "models--BAAI--bge-m3"

        with open(fields_path, "r", encoding="utf-8") as f:
            fields_dict = json.load(f)
        with open(operators_path, "r", encoding="utf-8") as f:
            operators_dict = json.load(f)

        fields_str = json.dumps(fields_dict, ensure_ascii=False)
        operators_str = json.dumps(operators_dict, ensure_ascii=False)

        self.prompt = {
            "system_content": (
                "You are a quantitative research agent that generates stock factors as symbolic expressions."
                "Your task is to convert financial text into VALID factor expressions. \n"
                "STRICT RULES: \n"
                "1. Each factor MUST be expressed as a symbolic expression, NOT natural language. \n"
                "2. You MUST ONLY use: - Allowed data fields (X) and - Allowed operators (O). \n"
                "3. You MUST NOT invent new fields or operators. \n"
                "4. Expressions must be fully composable and syntactically valid. \n"
                "5. Do NOT explain anything. \n"
                f"ALLOWED DATA FIELDS (X): {fields_str}, ALLOWED OPERATORS (O):{operators_str}. \n"
                "EXPRESSION FORMAT RULES: \n"
                "- Use function-style expressions: e.g., add(x, y), ts_mean(x, n), rank(x). \n"
                "- Nested expressions are allowed. \n"
                "- All arguments must be valid. \n"
                "OUTPUT FORMAT (STRICT JSON): \n"
                "[\{\{\"factor_name\": \"\", \"expression\": \"\", \"core_logic\": \"\", \"data_source\": [] \}\}] \n"
                "If no valid factor can be constructed: \{\{\"no_factor\": true\}\}. "
            ), 

            "assistant_content": (
                "I will strictly follow the rules and only output pure JSON for stock factor extraction."
            ), 

            "user_content": (
                "Convert the following financial text into valid factor expressions using the allowed fields and operators."
            )
        }
        self.project_root = project_root

    def set_prompt(self, system_content=None, assistant_content=None):
        if system_content is not None:
            self.prompt['system_content'] = system_content
        if assistant_content is not None:
            self.prompt['assistant_content'] = assistant_content
        return None

    def extract_knowledge(self, pdf_path, query, model, feedback=None, max_retries=5):
        try:
            file = read_pdf(pdf_path)
            embedding_model = SentenceTransformer(str(self.embedding_model_path), device=DEVICE)
            rag_context = rag_search(file, query, embedding_model)
            user_content = (
                self.prompt["user_content"] 
                + f"\nAvoid the mistakes: {feedback}\n" + str(rag_context)
                )
        except Exception as e:
            record_error_event(
                stage="kea.prepare_context",
                error=e,
                current_output=None,
                extra={"pdf_path": pdf_path, "query": query},
            )
            raise

        for attempt in range(max_retries):
            instruction = None
            try:
                instruction = call_llm_api(
                    model=model,
                    system_content=self.prompt["system_content"],
                    assistant_content=self.prompt["assistant_content"],
                    user_content=user_content,
                )
                cleaned = (instruction.replace("```json", "").replace("```", "").strip())
                solution = json.loads(cleaned)
                return solution

            except Exception as e:
                error_text = f"{type(e).__name__}: {e}"
                print(f"[Attempt {attempt+1}] output handling failed: {error_text}")
                user_content += "\n\n" + f"Avoid the mistakes: Error in attempt {attempt+1}: {error_text}"
        
        record_error_event(
                stage="kea.call_llm_api",
                error="Failed to call LLM API after multiple attempts.",
                current_output=None,
                extra={"pdf_path": pdf_path, "query": query, "model": model, "max_retries": max_retries},
            )
        return None
    