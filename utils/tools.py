import pdfplumber
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from utils.error_utils import record_error_event


def read_pdf(file_path):
    """
    read a pdf file
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    except Exception as e:
        record_error_event(
            stage="tools.read_pdf",
            error=e,
            current_output=None,
            extra={"file_path": str(file_path)},
        )
        raise


def rag_search(text, query, model, top_k=5):
    """
    Split a full text into chunks with limited length, using LLM embedding
    for RAG to find the top_k most relevant chunks to the query.
    """
    try:
        # split the full text into chunks
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o-mini",
            chunk_size=300,
            chunk_overlap=60
        )
        chunks = splitter.split_text(text)

        # make embedding with given model
        chunk_embeddings = model.encode(
            ["Represent this sentence for retrieval: " + c for c in chunks],
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=model.device).cpu().numpy()

        query_embedding = model.encode(
            ["Represent this sentence for retrieval: " + query],
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=model.device).cpu().numpy()

        # construct a faiss, perform vector search using inner product
        dim = chunk_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(chunk_embeddings))

        _, I = index.search(query_embedding, k=top_k)

        top_chunks = [chunks[i] for i in I[0]]
        return top_chunks
    except Exception as e:
        record_error_event(
            stage="tools.rag_search",
            error=e,
            current_output=None,
            extra={"query": query, "top_k": top_k},
        )
        raise


def call_llm_api(model, system_content, assistant_content, user_content):
    """
    model: the LLM called, limited in DeepSeek, Qwen, GLM, MiniMax
    """
    if model not in ['DeepSeek-V3.2', 'Qwen3.5-27B', 'GLM-5', 'MiniMax-2.5']:
        raise ValueError(f"{model} can not be called! Please input the right model name!")
    try:
        client = OpenAI(
            api_key="",
            base_url="https://llmapi.paratera.com"
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_content}"},
                {"role": "assistant", "content": f"{assistant_content}"},
                {"role": "user", "content": f"{user_content}"}
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content
    except Exception as e:
        record_error_event(
            stage="tools.call_llm_api",
            error=e,
            current_output=None,
            extra={"model": model},
        )
        raise
