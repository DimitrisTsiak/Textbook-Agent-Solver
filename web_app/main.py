import os
import sys
import json
import re
import time
import traceback
from typing import Generator
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Resolve project paths to import tools and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env import load_env
from utils.embeddings import CustomGeminiEmbeddingFunction
from tools.calculator import calculate_linear_algebra
from tools.search import search_textbook
import google.generativeai as genai

app = FastAPI(title="Linear Algebra Solver Blackboard")

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load env in project root
load_env(os.path.join(project_root, ".env"))
api_key = os.environ.get("GEMINI_API_KEY", "").strip()

# Initialize Gemini
if api_key and api_key != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=api_key)

def load_exercises(dataset: str = "gr2"):
    json_filename = "parsed_gr2.json" if dataset == "gr2" else "parsed_gr1.json"
    json_path = os.path.join(project_root, json_filename)
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Filter for exercises
        exercises = []
        for item in data:
            if item.get('category') == 'exercise':
                exercises.append({
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "clean_question": item.get("clean_question", item.get("question")),
                    "answer": item.get("answer"),
                    "location": item.get("location", {})
                })
        return exercises
    except Exception as e:
        print(f"Error loading exercises: {e}")
        return []

@app.get("/api/exercises")
def get_exercises(dataset: str = "gr2"):
    exercises = load_exercises(dataset)
    return JSONResponse(content={"exercises": exercises})

def sse_event(event_type: str, data: dict) -> str:
    """Helper to format SSE data packet."""
    packet = {
        "type": event_type,
        "data": data
    }
    return f"data: {json.dumps(packet)}\n\n"

def solve_and_stream_generator(
    exercise_index: int,
    dataset: str,
    use_rag: bool,
    use_search: bool,
    use_calc: bool,
    model_name: str
) -> Generator[str, None, None]:
    try:
        # Load environment variables in case they were updated
        load_env(os.path.join(project_root, ".env"))
        local_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        
        if not local_api_key or local_api_key == "YOUR_GEMINI_API_KEY":
            yield sse_event("error", {"message": "Gemini API Key is not set or placeholder remains in .env. Please configure a valid GEMINI_API_KEY."})
            return
            
        genai.configure(api_key=local_api_key)
        
        # Load selected exercise
        exercises = load_exercises(dataset)
        exercise = next((ex for ex in exercises if ex["index"] == exercise_index), None)
        if not exercise:
            yield sse_event("error", {"message": f"Could not find exercise {exercise_index} in dataset {dataset}."})
            return
            
        yield sse_event("status", {"message": f"Loaded exercise {exercise_index}."})
        
        # RAG context retrieval
        context = ""
        rag_items = []
        suffix = "gemini" if local_api_key and local_api_key != "YOUR_GEMINI_API_KEY" else "local"
        
        if use_rag:
            yield sse_event("status", {"message": "Connecting to ChromaDB and querying relevant context..."})
            db_path = os.path.join(project_root, "chroma_db")
            if not os.path.exists(db_path):
                yield sse_event("warning", {"message": f"ChromaDB folder not found at '{db_path}'. Skipping RAG retrieval..."})
            else:
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=db_path)
                    
                    embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "models/text-embedding-004").strip()
                    embedding_function = None
                    if suffix == "gemini":
                        embedding_function = CustomGeminiEmbeddingFunction(api_key=local_api_key, model_name=embedding_model_name)
                        
                    theory_col_name = f"textbook_theory_{suffix}"
                    chunks_col_name = f"chapter_chunks_{suffix}"
                    
                    clean_q = exercise.get('clean_question', exercise['question'])
                    theory_context = []
                    chunk_context = []
                    
                    # 1. Retrieve theory
                    try:
                        if embedding_function:
                            theory_col = client.get_collection(theory_col_name, embedding_function=embedding_function)
                        else:
                            theory_col = client.get_collection(theory_col_name)
                        
                        res = theory_col.query(query_texts=[clean_q], n_results=2)
                        if res and res['documents'] and len(res['documents'][0]) > 0:
                            for doc, dist, meta in zip(res['documents'][0], res['distances'][0], res['metadatas'][0]):
                                raw_latex = meta.get('raw_content', doc)
                                title_info = f" ({meta.get('title')})" if meta.get('title') else ""
                                label_info = f" (Label: {meta.get('label')})" if meta.get('label') else ""
                                
                                item = {
                                    "type": meta.get('type', 'theory'),
                                    "title": meta.get('title', 'Theory'),
                                    "label": meta.get('label', ''),
                                    "content": raw_latex,
                                    "distance": float(dist)
                                }
                                rag_items.append(item)
                                theory_context.append(f"[{meta.get('type', 'theory').upper()}]{title_info}{label_info}:\n{raw_latex}")
                    except Exception as e:
                        print(f"Theory retrieval error: {e}")
                        yield sse_event("warning", {"message": f"Failed to query theory collection: {str(e)}"})
                        
                    # 2. Retrieve chapter chunks
                    try:
                        if embedding_function:
                            chunks_col = client.get_collection(chunks_col_name, embedding_function=embedding_function)
                        else:
                            chunks_col = client.get_collection(chunks_col_name)
                        
                        res = chunks_col.query(query_texts=[clean_q], n_results=2)
                        if res and res['documents'] and len(res['documents'][0]) > 0:
                            for doc, dist, meta in zip(res['documents'][0], res['distances'][0], res['metadatas'][0]):
                                raw_latex = meta.get('raw_content', doc)
                                item = {
                                    "type": "paragraph",
                                    "title": f"Subsection: {meta.get('subsection', 'Unknown')}",
                                    "label": "",
                                    "content": raw_latex,
                                    "distance": float(dist)
                                }
                                rag_items.append(item)
                                chunk_context.append(f"Paragraph Excerpt (Subsection '{meta.get('subsection', 'Unknown')}'):\n{raw_latex}")
                    except Exception as e:
                        print(f"Chunks retrieval error: {e}")
                        yield sse_event("warning", {"message": f"Failed to query textbook paragraphs: {str(e)}"})
                        
                    # Combine context
                    context_parts = []
                    if theory_context:
                        context_parts.append("--- RELEVANT TEXTBOOK THEORY ---\n" + "\n\n".join(theory_context))
                    if chunk_context:
                        context_parts.append("--- RELEVANT CHAPTER CONTEXT ---\n" + "\n\n".join(chunk_context))
                    if context_parts:
                        context = "\n\n".join(context_parts)
                        
                except Exception as e:
                    yield sse_event("warning", {"message": f"RAG connection failed: {str(e)}"})
                    
        # Send details about retrieved context
        if use_rag and rag_items:
            yield sse_event("rag_data", {"items": rag_items})
            yield sse_event("status", {"message": f"Retrieved {len(rag_items)} references from database."})
        elif use_rag:
            yield sse_event("status", {"message": "No RAG context could be retrieved."})
            
        # Formulate instruction prompt
        instructions = []
        if use_search:
            instructions.append("You have access to the tool `search_textbook(query)` to search the textbook for relevant definitions, theorems, and examples if needed.")
        if use_calc:
            instructions.append("You have access to the tool `calculate_linear_algebra(code)` to run Python code to perform matrix operations, row reductions, or algebra. The tool returns stdout, so print your results. IMPORTANT: Do not write import statements in your code. SymPy public functions/classes (such as Matrix, symbols, solve, etc.) and NumPy (as np) are already pre-imported in the execution environment. REMINDER: You must always use this tool to verify and perform any mathematical or linear algebra calculations. Do not rely on calculations provided in the prompt or exercise text as they may be inaccurate or misleading.")
            
        tool_instructions = "\n".join(instructions)
        
        if use_rag and context:
            prompt = f"""You are a mathematics professor teaching one student. Solve the following linear algebra exercise step-by-step.
Use the relevant textbook context provided below to guide your solution, referring to definitions, theorems, and row reduction notations as described in the context.

{tool_instructions}

--- CONTEXT ---
{context}

--- EXERCISE ---
{exercise['question']}

Provide your complete mathematical solution. Keep your explanation concise but mathematically rigorous. Cite relevant theorems or definitions when you apply them.
"""
        else:
            prompt = f"""You are a mathematics professor. Solve the following linear algebra exercise step-by-step.
Do not use any external textbook context unless you search for it.

{tool_instructions}

--- EXERCISE ---
{exercise['question']}

Provide your complete mathematical solution. Keep your explanation concise but mathematically rigorous.
"""
            
        yield sse_event("status", {"message": "Configuring Gemini LLM agent..."})
        
        active_tools = []
        if use_search:
            active_tools.append(search_textbook)
        if use_calc:
            active_tools.append(calculate_linear_algebra)
            
        model = genai.GenerativeModel(model_name, tools=active_tools if active_tools else None)
        chat = model.start_chat()
        
        yield sse_event("status", {"message": "Generating solution on blackboard..."})
        
        # Invoke chat and handle tool loop manually to stream execution
        response_stream = chat.send_message(prompt, stream=True)
        
        def process_stream(stream):
            f_calls = []
            text_acc = []
            for chunk in stream:
                # Check for parts
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                for part in chunk.candidates[0].content.parts:
                    if part.function_call:
                        f_calls.append(part.function_call)
                    if part.text:
                        text_acc.append(part.text)
                        # Yield text chunk directly to client
                        yield {"type": "text", "content": part.text}
            return f_calls, "".join(text_acc)
            
        # First turn
        f_calls, text = [], ""
        for ev in process_stream(response_stream):
            if ev["type"] == "text":
                yield sse_event("text", {"content": ev["content"]})
            elif ev["type"] == "function_call":
                # We handle function calls inside the loop below
                pass
                
        # Retrieve function calls from the completed response_stream
        if chat.history:
            last_message = chat.history[-1]
            if last_message.role == "model":
                f_calls = [part.function_call for part in last_message.parts if part.function_call]
                
        while f_calls:
            response_parts = []
            for fc in f_calls:
                func_name = fc.name
                func_args = dict(fc.args)
                
                # Format arguments nicely
                args_str = json.dumps(func_args, indent=2)
                yield sse_event("tool_call", {"name": func_name, "args": args_str})
                yield sse_event("status", {"message": f"Running tool: {func_name}..."})
                
                # Execute
                result = ""
                try:
                    if func_name == "calculate_linear_algebra":
                        code = func_args.get("code", "")
                        result = calculate_linear_algebra(code)
                    elif func_name == "search_textbook":
                        query = func_args.get("query", "")
                        result = search_textbook(query)
                    else:
                        result = f"Error: Unknown tool '{func_name}'"
                except Exception as ex_tool:
                    result = f"Tool execution failed: {str(ex_tool)}"
                    
                yield sse_event("tool_result", {"name": func_name, "result": result})
                
                response_parts.append({
                    "function_response": {
                        "name": func_name,
                        "response": {"result": result}
                    }
                })
                
            yield sse_event("status", {"message": "Sending tool outcomes to agent..."})
            
            # Send results back and stream next responses
            next_stream = chat.send_message(response_parts, stream=True)
            
            for ev in process_stream(next_stream):
                if ev["type"] == "text":
                    yield sse_event("text", {"content": ev["content"]})
                    
            # Check for subsequent tool calls
            f_calls = []
            if chat.history:
                last_message = chat.history[-1]
                if last_message.role == "model":
                    f_calls = [part.function_call for part in last_message.parts if part.function_call]
                    
        yield sse_event("done", {})
        
    except Exception as e:
        traceback.print_exc()
        yield sse_event("error", {"message": f"An error occurred: {str(e)}", "traceback": traceback.format_exc()})

@app.get("/api/solve")
def solve_exercise(
    index: int = Query(..., description="Exercise index to solve"),
    dataset: str = Query("gr2", description="gr1 or gr2 dataset"),
    rag: bool = Query(True, description="Enable RAG"),
    search: bool = Query(True, description="Enable search tool"),
    calc: bool = Query(True, description="Enable calc tool"),
    model: str = Query("models/gemini-3.5-flash-lite", description="Model name")
):
    return StreamingResponse(
        solve_and_stream_generator(index, dataset, rag, search, calc, model),
        media_type="text/event-stream"
    )

# Serve the HTML page directly on root GET
@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = os.path.join(script_dir, "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback if templates/index.html is not created yet
        return HTMLResponse(content="<h1>Blackboard App is starting... Please wait.</h1>")

if __name__ == "__main__":
    import uvicorn
    # Start on localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
