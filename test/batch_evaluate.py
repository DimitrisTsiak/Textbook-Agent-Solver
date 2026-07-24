import os
import sys
import json
import re
import time
import argparse
from datetime import datetime
import google.generativeai as genai
from google.api_core import exceptions

# Resolve paths relative to the script location to allow importing from tools and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.search import search_textbook
from tools.calculator import calculate_linear_algebra
from utils.env import load_env
from utils.embeddings import CustomGeminiEmbeddingFunction

# load_env is imported from utils.env

def get_model_slug(model_name):
    """
    Cleans model name to be safe for filenames.
    """
    name = model_name.replace("models/", "")
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    return name

def main():
    # Force line buffering for standard output to ensure real-time logging in tasks
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Batch evaluate LLM solver on Chapter 1 exercises.")
    parser.add_argument("--model", type=str, default="models/gemini-3.5-flash-lite", 
                        help="Gemini API model name to use for reasoning.")
    parser.add_argument("--no-rag", action="store_true", 
                        help="Disable retrieval-augmented generation (RAG).")
    parser.add_argument("--n-theory", type=int, default=2, 
                        help="Number of textbook theory items to retrieve.")
    parser.add_argument("--n-chunks", type=int, default=2, 
                        help="Number of textbook paragraph chunks to retrieve.")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit the number of exercises to evaluate (for testing).")
    parser.add_argument("--use-search", action="store_true", 
                        help="Enable the BM25 textbook keyword search tool for the solver.")
    parser.add_argument("--use-calc", action="store_true", 
                        help="Enable the linear algebra Python code calculator tool for the solver.")
    parser.add_argument("--temperature", type=float, default=None, 
                        help="Sampling temperature for model generation.")
    parser.add_argument("--top-p", type=float, default=None, 
                        help="Top-p sampling parameter.")
    parser.add_argument("--top-k", type=int, default=None, 
                        help="Top-k sampling parameter.")
    parser.add_argument("--max-output-tokens", type=int, default=None, 
                        help="Maximum output tokens constraint.")
    args = parser.parse_args()

    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    env_path = os.path.join(project_root, ".env")
    json_path = os.path.join(project_root, "parsed_gr2.json")
    db_path = os.path.join(project_root, "chroma_db")
    results_dir = os.path.join(script_dir, "results")

    # Ensure output results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Load environment variables
    load_env(env_path)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("[ERROR] Gemini API Key is not set or placeholder remains in .env.")
        print("Please edit the .env file in the project root folder and insert a valid GEMINI_API_KEY.")
        sys.exit(1)

    # Configure Gemini API
    genai.configure(api_key=api_key)

    # Determine embedding configuration
    suffix = "gemini" if api_key and api_key != "YOUR_GEMINI_API_KEY" else "local"
    embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "models/text-embedding-004").strip()
    if suffix == "local":
        embedding_model_name = "local SentenceTransformers (all-MiniLM-L6-v2)"

    # Load exercises from JSON
    if not os.path.exists(json_path):
        print(f"[ERROR] Parsed JSON data not found at {json_path}. Please run parse_latex.py first.")
        sys.exit(1)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    exercises = [item for item in data if item['category'] == 'exercise']
    if args.limit is not None:
        exercises = exercises[:args.limit]

    total_exercises = len(exercises)
    print(f"Loaded {total_exercises} exercises for evaluation.")

    # Initialize ChromaDB client if RAG is enabled
    use_rag = not args.no_rag
    chroma_client = None
    theory_col = None
    chunks_col = None
    embedding_function = None

    if use_rag:
        if not os.path.exists(db_path):
            print(f"[WARNING] ChromaDB folder not found at '{db_path}'. Disabling RAG...")
            use_rag = False
        else:
            try:
                import chromadb
                print(f"Connecting to ChromaDB at '{db_path}'...")
                chroma_client = chromadb.PersistentClient(path=db_path)
                
                # Setup custom embedding function for gemini if suffix is gemini
                if suffix == "gemini":
                    embedding_function = CustomGeminiEmbeddingFunction(api_key=api_key, model_name=embedding_model_name)

                # Fetch collections
                theory_col_name = f"textbook_theory_{suffix}"
                chunks_col_name = f"chapter_chunks_{suffix}"

                try:
                    if embedding_function:
                        theory_col = chroma_client.get_collection(theory_col_name, embedding_function=embedding_function)
                    else:
                        theory_col = chroma_client.get_collection(theory_col_name)
                except Exception as e:
                    print(f"[WARNING] Could not retrieve collection '{theory_col_name}': {e}")

                try:
                    if embedding_function:
                        chunks_col = chroma_client.get_collection(chunks_col_name, embedding_function=embedding_function)
                    else:
                        chunks_col = chroma_client.get_collection(chunks_col_name)
                except Exception as e:
                    print(f"[WARNING] Could not retrieve collection '{chunks_col_name}': {e}")

            except ImportError:
                print("[WARNING] chromadb package is not installed. Disabling RAG...")
                use_rag = False

    # Initialize results structures
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_status = "rag" if use_rag else "norag"
    search_status = "search" if args.use_search else "nosearch"
    calc_status = "calc" if args.use_calc else "nocalc"
    model_slug = get_model_slug(args.model)
    output_filename = f"eval_run_{model_slug}_{rag_status}_{search_status}_{calc_status}_{timestamp_str}.json"
    output_filepath = os.path.join(results_dir, output_filename)

    generation_config = {}
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
    if args.top_p is not None:
        generation_config["top_p"] = args.top_p
    if args.top_k is not None:
        generation_config["top_k"] = args.top_k
    if args.max_output_tokens is not None:
        generation_config["max_output_tokens"] = args.max_output_tokens

    run_metadata = {
        "timestamp_start": datetime.now().isoformat(),
        "timestamp_end": None,
        "model_name": args.model,
        "use_rag": use_rag,
        "use_search": args.use_search,
        "use_calc": args.use_calc,
        "generation_config": generation_config if generation_config else None,
        "embedding_model_name": embedding_model_name if use_rag else None,
        "retrieval_suffix": suffix if use_rag else None,
        "n_theory_results": args.n_theory if use_rag else 0,
        "n_chunk_results": args.n_chunks if use_rag else 0,
        "total_exercises_processed": 0,
        "success_count": 0,
        "failure_count": 0
    }

    results = []

    # Prepare model
    active_tools = []
    if args.use_search:
        active_tools.append(search_textbook)
    if args.use_calc:
        active_tools.append(calculate_linear_algebra)
        
    if active_tools:
        tool_names = ", ".join([t.__name__ for t in active_tools])
        print(f"Configuring Generative Model '{args.model}' with tools: [{tool_names}]...")
        model = genai.GenerativeModel(
            args.model,
            tools=active_tools,
            generation_config=generation_config if generation_config else None
        )
    else:
        print(f"Configuring Generative Model '{args.model}' (all tools disabled)...")
        model = genai.GenerativeModel(
            args.model,
            generation_config=generation_config if generation_config else None
        )

    for idx, ex in enumerate(exercises):
        print(f"\n[{idx+1}/{total_exercises}] Processing Exercise {ex['index']} (ID: {ex['id']})...")
        
        question_text = ex['question']
        official_answer = ex['answer']
        clean_question = ex.get('clean_question', question_text)

        # Context retrieval
        retrieved_context_info = {"theory": [], "chunks": []}
        context_text = ""
        theory_context_pieces = []
        chunk_context_pieces = []

        if use_rag:
            # 1. Retrieve theory
            if theory_col:
                try:
                    res = theory_col.query(query_texts=[clean_question], n_results=args.n_theory)
                    if res and res['documents'] and len(res['documents'][0]) > 0:
                        for doc, dist, meta in zip(res['documents'][0], res['distances'][0], res['metadatas'][0]):
                            raw_latex = meta.get('raw_content', doc)
                            retrieved_context_info["theory"].append({
                                "id": meta.get('label', ''),
                                "type": meta.get('type', 'theory'),
                                "title": meta.get('title', ''),
                                "content": raw_latex,
                                "distance": float(dist)
                            })
                            title_info = f" ({meta.get('title')})" if meta.get('title') else ""
                            label_info = f" (Label: {meta.get('label')})" if meta.get('label') else ""
                            theory_context_pieces.append(
                                f"[{meta.get('type', 'theory').upper()}]{title_info}{label_info}:\n{raw_latex}"
                            )
                except Exception as e:
                    print(f"  [WARNING] Theory query failed: {e}")

            # 2. Retrieve chunks
            if chunks_col:
                try:
                    res = chunks_col.query(query_texts=[clean_question], n_results=args.n_chunks)
                    if res and res['documents'] and len(res['documents'][0]) > 0:
                        for doc, dist, meta in zip(res['documents'][0], res['distances'][0], res['metadatas'][0]):
                            raw_latex = meta.get('raw_content', doc)
                            retrieved_context_info["chunks"].append({
                                "subsection": meta.get('subsection', 'Unknown'),
                                "content": raw_latex,
                                "distance": float(dist)
                            })
                            chunk_context_pieces.append(
                                f"Paragraph Excerpt (Subsection '{meta.get('subsection', 'Unknown')}'):\n{raw_latex}"
                            )
                except Exception as e:
                    print(f"  [WARNING] Chunk query failed: {e}")

            # Combine contexts
            context_parts = []
            if theory_context_pieces:
                context_parts.append("--- RELEVANT TEXTBOOK THEORY ---\n" + "\n\n".join(theory_context_pieces))
            if chunk_context_pieces:
                context_parts.append("--- RELEVANT CHAPTER CONTEXT ---\n" + "\n\n".join(chunk_context_pieces))
            if context_parts:
                context_text = "\n\n".join(context_parts)

        # Formulate Prompt
        instructions = []
        if args.use_search:
            instructions.append("You have access to the tool `search_textbook(query)` to search the textbook for relevant definitions, theorems, and examples if needed.")
        if args.use_calc:
            instructions.append("You have access to the tool `calculate_linear_algebra(code)` to run Python code to perform matrix operations, row reductions, or algebra. The tool returns stdout, so print your results. IMPORTANT: Do not write import statements in your code. SymPy public functions/classes (such as Matrix, symbols, solve, etc.) and NumPy (as np) are already pre-imported in the execution environment. REMINDER: You must always use this tool to verify and perform any mathematical or linear algebra calculations. Do not rely on calculations provided in the prompt or exercise text as they may be inaccurate or misleading.")
            
        tool_instructions = "\n".join(instructions)

        if use_rag and context_text:
            prompt = f"""You are a mathematics professor. Solve the following linear algebra exercise step-by-step.
Use the relevant textbook context provided below to guide your solution, referring to definitions, theorems, and row reduction notations as described in the context.

{tool_instructions}

--- CONTEXT ---
{context_text}

--- EXERCISE ---
{question_text}

Provide your complete mathematical solution. Keep your explanation concise but mathematically rigorous. Cite relevant theorems or definitions when you apply them.
"""
        else:
            prompt = f"""You are a mathematics professor. Solve the following linear algebra exercise step-by-step.
Do not use any external textbook context unless you search for it.

{tool_instructions}

--- EXERCISE ---
{question_text}

Provide your complete mathematical solution. Keep your explanation concise but mathematically rigorous.
"""

        # Generate answer with retries for rate limits
        llm_answer = ""
        status = "success"
        error_msg = None
        duration = 0.0

        start_time = time.time()
        max_attempts = 5
        base_sleep = 65

        for attempt in range(max_attempts):
            try:
                if active_tools:
                    chat = model.start_chat(enable_automatic_function_calling=True)
                    response = chat.send_message(prompt, request_options={"timeout": 60.0})
                else:
                    response = model.generate_content(prompt, request_options={"timeout": 60.0})
                llm_answer = response.text
                break
            except exceptions.ResourceExhausted:
                # Quota issue, wait and retry
                sleep_time = base_sleep + (attempt * 10)
                print(f"  [Rate Limit] 429 ResourceExhausted. Attempt {attempt+1}/{max_attempts}. Sleeping {sleep_time} seconds...")
                time.sleep(sleep_time)
            except Exception as e:
                print(f"  [Error] Failed to generate content: {e}")
                status = "error"
                error_msg = str(e)
                break
        else:
            # Loop ended without break
            print("  [Error] Failed to generate content after max retries.")
            status = "error"
            error_msg = "Max retries exceeded due to rate limit."

        duration = time.time() - start_time
        print(f"  Result: {status.upper()} | Duration: {duration:.2f}s")

        # Count tool calls
        tool_call_count = 0
        if active_tools and status == "success":
            try:
                for content in chat.history:
                    for part in content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            tool_call_count += 1
            except Exception as e:
                print(f"  [WARNING] Failed to extract tool call count: {e}")

        # Save result
        results.append({
            "exercise_index": ex['index'],
            "exercise_id": ex['id'],
            "location": {
                "chapter": ex['location']['chapter'],
                "section": ex['location']['section'],
                "subsection": ex['location']['subsection']
            },
            "question": question_text,
            "official_answer": official_answer,
            "retrieved_context": retrieved_context_info if use_rag else None,
            "generated_answer": llm_answer,
            "use_search": args.use_search,
            "use_calc": args.use_calc,
            "tool_call_count": tool_call_count,
            "duration_seconds": round(duration, 2),
            "status": status,
            "error_message": error_msg
        })

        # Update metadata counters
        run_metadata["total_exercises_processed"] += 1
        if status == "success":
            run_metadata["success_count"] += 1
        else:
            run_metadata["failure_count"] += 1

        # Write current state incrementally to prevent loss
        report = {
            "metadata": run_metadata,
            "results": results
        }
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    # End run metadata
    run_metadata["timestamp_end"] = datetime.now().isoformat()
    report = {
        "metadata": run_metadata,
        "results": results
    }
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n==============================================")
    print("Batch Evaluation Completed!")
    print(f"Report File: {output_filepath}")
    print(f"Processed: {run_metadata['total_exercises_processed']}")
    print(f"Success: {run_metadata['success_count']}")
    print(f"Failure: {run_metadata['failure_count']}")
    print("==============================================")

if __name__ == '__main__':
    main()
