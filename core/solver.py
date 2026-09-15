"""
Shared solver logic for prompt construction and RAG context retrieval.
Used by web_app/main.py, test/batch_evaluate.py, and test/evaluate_exercise.py.
"""

import os
from utils.tracing import observe, update_current_observation

# ---------------------------------------------------------------------------
# Tool instruction constants
# ---------------------------------------------------------------------------

SEARCH_TOOL_INSTRUCTION = (
    "You have access to the tool `search_textbook(query)` to search the textbook "
    "for relevant definitions, theorems, and examples if needed."
)

CALC_TOOL_INSTRUCTION = (
    "You have access to the tool `calculate_linear_algebra(code)` to run Python code "
    "to perform matrix operations, row reductions, or algebra. The tool returns stdout, "
    "so print your results. IMPORTANT: Do not write import statements in your code. "
    "SymPy public functions/classes (such as Matrix, symbols, solve, etc.) and NumPy "
    "(as np) are already pre-imported in the execution environment. REMINDER: You must "
    "always use this tool to verify and perform any mathematical or linear algebra "
    "calculations. Do not rely on calculations provided in the prompt or exercise text "
    "as they may be inaccurate or misleading."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_tool_instructions(use_search: bool, use_calc: bool) -> str:
    """Build the tool instruction block for the LLM prompt."""
    instructions = []
    if use_search:
        instructions.append(SEARCH_TOOL_INSTRUCTION)
    if use_calc:
        instructions.append(CALC_TOOL_INSTRUCTION)
    return "\n".join(instructions)


def build_solver_prompt(question: str, context: str, tool_instructions: str, use_rag: bool) -> str:
    """Build the complete LLM prompt for solving a linear algebra exercise."""
    if use_rag and context:
        return f"""You are a mathematics professor. Solve the following linear algebra exercise step-by-step.
Use the relevant textbook context provided below to guide your solution, referring to definitions, theorems, and row reduction notations as described in the context.
Whenever you rely on a definition, theorem, or method from the retrieved context, explicitly cite it (e.g., 'According to Lemma 1.2...').
{tool_instructions}

--- CONTEXT ---
{context}

--- EXERCISE ---
{question}

Provide your complete mathematical solution. Keep your explanation concise but mathematically rigorous. Cite relevant theorems or definitions when you apply them.
"""
    else:
        return f"""You are a mathematics professor. Solve the following linear algebra exercise step-by-step.
Do not use any external textbook context unless you search for it.

{tool_instructions}

--- EXERCISE ---
{question}

Provide your complete mathematical solution. Keep your explanation concise but mathematically rigorous.
"""


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def get_embedding_suffix(api_key: str) -> str:
    """Determine whether to use Gemini or local embeddings based on API key."""
    if api_key and api_key != "YOUR_GEMINI_API_KEY":
        return "gemini"
    return "local"


def get_rag_collections(db_path: str, api_key: str):
    """
    Initialize ChromaDB client and return theory/chunks collections.

    Args:
        db_path: Path to the ChromaDB persistent storage directory.
        api_key: Gemini API key (used to select embedding function).

    Returns:
        tuple: (theory_col, chunks_col, warnings)
            theory_col and chunks_col may be None if retrieval fails.
            warnings is a list of warning message strings.
    """
    import chromadb
    from utils.embeddings import CustomGeminiEmbeddingFunction

    warnings = []
    suffix = get_embedding_suffix(api_key)
    embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "models/text-embedding-004").strip()

    embedding_function = None
    if suffix == "gemini":
        embedding_function = CustomGeminiEmbeddingFunction(api_key=api_key, model_name=embedding_model_name)

    client = chromadb.PersistentClient(path=db_path)

    theory_col_name = f"textbook_theory_{suffix}"
    chunks_col_name = f"chapter_chunks_{suffix}"

    theory_col = None
    chunks_col = None

    try:
        if embedding_function:
            theory_col = client.get_collection(theory_col_name, embedding_function=embedding_function)
        else:
            theory_col = client.get_collection(theory_col_name)
    except Exception as e:
        warnings.append(f"Could not retrieve collection '{theory_col_name}': {e}")

    try:
        if embedding_function:
            chunks_col = client.get_collection(chunks_col_name, embedding_function=embedding_function)
        else:
            chunks_col = client.get_collection(chunks_col_name)
    except Exception as e:
        warnings.append(f"Could not retrieve collection '{chunks_col_name}': {e}")

    return theory_col, chunks_col, warnings


@observe(as_type="retriever", name="rag-retrieval")
def query_rag_context(clean_question: str, theory_col=None, chunks_col=None,
                      n_theory: int = 2, n_chunks: int = 2):
    """
    Query RAG collections for relevant context to a question.

    Args:
        clean_question: The cleaned exercise question text to query with.
        theory_col: ChromaDB collection for textbook theory (optional).
        chunks_col: ChromaDB collection for chapter paragraph chunks (optional).
        n_theory: Number of theory results to retrieve.
        n_chunks: Number of chunk results to retrieve.

    Returns:
        tuple: (context_text, rag_items, warnings)
            context_text: Combined context string for the LLM prompt.
            rag_items: List of dicts with structured retrieval data.
                Each item has keys: source, type, title, label, content,
                distance, subsection.
            warnings: List of warning message strings.
    """
    theory_context_pieces = []
    chunk_context_pieces = []
    rag_items = []
    warnings = []

    # 1. Query theory collection
    if theory_col:
        try:
            res = theory_col.query(query_texts=[clean_question], n_results=n_theory)
            if res and res['documents'] and len(res['documents'][0]) > 0:
                for doc, dist, meta in zip(res['documents'][0], res['distances'][0], res['metadatas'][0]):
                    raw_latex = meta.get('raw_content', doc)
                    title_info = f" ({meta.get('title')})" if meta.get('title') else ""
                    label_info = f" (Label: {meta.get('label')})" if meta.get('label') else ""

                    rag_items.append({
                        "source": "theory",
                        "type": meta.get('type', 'theory'),
                        "title": meta.get('title', 'Theory'),
                        "label": meta.get('label', ''),
                        "content": raw_latex,
                        "distance": float(dist),
                        "subsection": meta.get('subsection', ''),
                    })
                    theory_context_pieces.append(
                        f"[{meta.get('type', 'theory').upper()}]{title_info}{label_info}:\n{raw_latex}"
                    )
        except Exception as e:
            print(f"Theory retrieval error: {e}")
            warnings.append(f"Failed to query theory collection: {str(e)}")

    # 2. Query chunks collection
    if chunks_col:
        try:
            res = chunks_col.query(query_texts=[clean_question], n_results=n_chunks)
            if res and res['documents'] and len(res['documents'][0]) > 0:
                for doc, dist, meta in zip(res['documents'][0], res['distances'][0], res['metadatas'][0]):
                    raw_latex = meta.get('raw_content', doc)

                    rag_items.append({
                        "source": "chunk",
                        "type": "paragraph",
                        "title": f"Subsection: {meta.get('subsection', 'Unknown')}",
                        "label": "",
                        "content": raw_latex,
                        "distance": float(dist),
                        "subsection": meta.get('subsection', 'Unknown'),
                    })
                    chunk_context_pieces.append(
                        f"Paragraph Excerpt (Subsection '{meta.get('subsection', 'Unknown')}'):\n{raw_latex}"
                    )
        except Exception as e:
            print(f"Chunks retrieval error: {e}")
            warnings.append(f"Failed to query textbook paragraphs: {str(e)}")

    # Combine context
    context_parts = []
    if theory_context_pieces:
        context_parts.append("--- RELEVANT TEXTBOOK THEORY ---\n" + "\n\n".join(theory_context_pieces))
    if chunk_context_pieces:
        context_parts.append("--- RELEVANT CHAPTER CONTEXT ---\n" + "\n\n".join(chunk_context_pieces))
    context_text = "\n\n".join(context_parts) if context_parts else ""

    update_current_observation(
        input={"query": clean_question, "n_theory": n_theory, "n_chunks": n_chunks},
        output={
            "retrieved_count": len(rag_items),
            "items": [
                {
                    "source": item["source"],
                    "type": item["type"],
                    "title": item["title"],
                    "label": item["label"],
                    "distance": item["distance"],
                    "subsection": item["subsection"]
                }
                for item in rag_items
            ],
            "context_length": len(context_text)
        }
    )

    return context_text, rag_items, warnings
