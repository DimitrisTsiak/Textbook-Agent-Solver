import os
import sys
import json
import re
import google.generativeai as genai

# Resolve paths to allow importing from tools and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.search import search_textbook
from tools.calculator import calculate_linear_algebra
from utils.env import load_env
from core.solver import (
    build_tool_instructions,
    build_solver_prompt,
    get_rag_collections,
    query_rag_context,
)

# =====================================================================
# CONFIGURATION
# =====================================================================
EXERCISE_INDEX = 10
MODEL_NAME = "models/gemini-3.5-flash-lite"  

# RAG CONFIGURATION
USE_RAG = True               # Set to True to retrieve context from ChromaDB
N_THEORY_RESULTS = 2         # Number of theory blocks to retrieve
N_CHUNK_RESULTS = 2          # Number of textbook paragraphs to retrieve
USE_SEARCH_TOOL = True       # Set to True to give the agent access to the BM25 search tool
USE_CALCULATOR_TOOL = True   # Set to True to give the agent access to the linear algebra calculator tool
# =====================================================================

# load_env is imported from utils.env

def find_exercise_by_index(json_path, target_index):
    """
    Loads parsed JSON and returns the exercise matching target_index.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Parsed data not found at {json_path}. Please run parse_latex.py first.")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    exercises = [item for item in data if item['category'] == 'exercise']
    
    for ex in exercises:
        if ex['index'] == target_index:
            return ex
            
    return None

def evaluate():
    print(f"Loading environment variables...")
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("\n[ERROR] Gemini API Key is not set or placeholder remains in .env.")
        print("Please edit the .env file in the project root folder and insert a valid GEMINI_API_KEY.")
        return
        
    # Find the exercise
    json_path = os.path.join("..", "parsed_gr1.json")
    if not os.path.exists(json_path):
        json_path = "parsed_gr1.json"
        
    print(f"Retrieving exercise {EXERCISE_INDEX} from '{json_path}'...")
    exercise = find_exercise_by_index(json_path, EXERCISE_INDEX)
    
    if not exercise:
        print(f"[ERROR] Could not find exercise with index {EXERCISE_INDEX} in the JSON file.")
        return
        
    question_text = exercise['question']
    official_answer = exercise['answer']
    clean_question = exercise.get('clean_question', question_text)
    
    # -------------------------------------------------------------
    # ChromaDB Context Retrieval (RAG)
    # -------------------------------------------------------------
    use_rag = USE_RAG
    context = ""
    retrieved_summary = []
    suffix = "local"
    
    if use_rag:
        db_path = "chroma_db"
        if not os.path.exists(db_path):
            db_path = os.path.join("..", "chroma_db")
            
        if not os.path.exists(db_path):
            print(f"[WARNING] ChromaDB folder not found at '{db_path}'. Skipping RAG retrieval...")
        else:
            try:
                print(f"Connecting to ChromaDB at '{db_path}'...")
                theory_col, chunks_col, setup_warnings = get_rag_collections(db_path, api_key)
                for w in setup_warnings:
                    print(f"[WARNING] {w}")
                
                context, rag_items, query_warnings = query_rag_context(
                    clean_question, theory_col=theory_col, chunks_col=chunks_col,
                    n_theory=N_THEORY_RESULTS, n_chunks=N_CHUNK_RESULTS
                )
                for w in query_warnings:
                    print(f"[WARNING] {w}")
                
                # Build markdown summary from structured rag_items
                for item in rag_items:
                    if item["source"] == "theory":
                        retrieved_summary.append(
                            f"* **{item['type'].capitalize()}** (Dist: {item['distance']:.4f}) - Label: `{item['label'] or 'None'}`"
                        )
                    else:
                        retrieved_summary.append(
                            f"* **Chapter Chunk** (Dist: {item['distance']:.4f}) - Subsection: `{item['subsection']}`"
                        )
                
                if context:
                    print("Retrieved context successfully.")
                    suffix = "gemini" if api_key and api_key != "YOUR_GEMINI_API_KEY" else "local"
                else:
                    print("No matching context was retrieved.")
            except ImportError:
                print("[WARNING] chromadb package is not installed. Skipping RAG retrieval...")
                use_rag = False
            
    # -------------------------------------------------------------
    # Prompt Construction & LLM Call
    # -------------------------------------------------------------
    tool_instructions = build_tool_instructions(USE_SEARCH_TOOL, USE_CALCULATOR_TOOL)
    prompt = build_solver_prompt(question_text, context, tool_instructions, use_rag)
        
    # Configure Gemini LLM
    genai.configure(api_key=api_key)
    
    active_tools = []
    if USE_SEARCH_TOOL:
        active_tools.append(search_textbook)
    if USE_CALCULATOR_TOOL:
        active_tools.append(calculate_linear_algebra)
        
    if active_tools:
        tool_names = ", ".join([t.__name__ for t in active_tools])
        print(f"Configuring {MODEL_NAME} API client with tools: [{tool_names}]...")
        model = genai.GenerativeModel(
            MODEL_NAME,
            tools=active_tools
        )
        print("Calling Gemini LLM with automatic function calling enabled...")
        try:
            chat = model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "top_k": 30,
                    "max_output_tokens": 8192,
                }
            )
            llm_answer = response.text
            print("LLM generated answer successfully.")
        except Exception as e:
            print(f"[ERROR] Gemini API invocation failed: {e}")
            return
    else:
        print(f"Configuring {MODEL_NAME} API client (all tools disabled)...")
        model = genai.GenerativeModel(MODEL_NAME)
        print("Calling Gemini LLM...")
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "top_k": 30,
                    "max_output_tokens": 8192,
                }
            )
            llm_answer = response.text
            print("LLM generated answer successfully.")
        except Exception as e:
            print(f"[ERROR] Gemini API invocation failed: {e}")
            return
        
    # Output to markdown file
    output_dir = "test"
    if not os.path.exists(output_dir):
        if os.path.basename(os.getcwd()) == "test":
            output_dir = "."
        else:
            os.makedirs(output_dir)
            
    out_filename = f"comparison_exercise_{EXERCISE_INDEX}.md"
    out_path = os.path.join(output_dir, out_filename)
    
    retrieved_items_list = "\n".join(retrieved_summary) if retrieved_summary else "* No items retrieved."
    
    md_content = f"""# Comparison for Exercise {EXERCISE_INDEX}

## Location
* **Chapter**: {exercise['location']['chapter']}
* **Section**: {exercise['location']['section']}
* **Subsection**: {exercise['location']['subsection']}
* **ID**: `{exercise['id']}`
* **Recommended**: {exercise['recommended']}
* **Puzzle**: {exercise['puzzle']}

## RAG Configuration
* **RAG Enabled**: {use_rag}
* **Search Tool Enabled**: {USE_SEARCH_TOOL}
* **Calculator Tool Enabled**: {USE_CALCULATOR_TOOL}
* **Retrieval Suffix**: `{suffix}`
* **Retrieved Theory Count**: {N_THEORY_RESULTS}
* **Retrieved Paragraph Count**: {N_CHUNK_RESULTS}

### Retrieved Index Items:
{retrieved_items_list}

---

## Question (LaTeX)
```latex
{question_text}
```

---

## Gemini LLM Answer ({MODEL_NAME})
{llm_answer}

---

## Official Textbook Answer (LaTeX)
```latex
{official_answer}
```
"""
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"\nSuccess! Solution and comparison written to: {os.path.abspath(out_path)}")

if __name__ == '__main__':
    evaluate()
