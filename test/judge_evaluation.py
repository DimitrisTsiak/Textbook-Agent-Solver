import os
import sys
import json
import re
import time
import argparse
import google.generativeai as genai
from google.api_core import exceptions

# Resolve paths to allow importing from tools and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env import load_env
from utils.tracing import observe, flush_traces

# load_env is imported from utils.env

@observe(name="judge-evaluation")
def main():
    # Force line buffering for standard output to ensure real-time logging
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="LLM-as-a-judge grading script.")
    parser.add_argument("--file", type=str, default=None, 
                        help="Path to the JSON evaluation results file. Defaults to the latest file in test/results/.")
    parser.add_argument("--model", type=str, default="models/gemini-3.5-flash-lite", 
                        help="Gemini API model name to use as judge.")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-evaluation of already graded exercises.")
    args = parser.parse_args()

    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    env_path = os.path.join(project_root, ".env")
    results_dir = os.path.join(script_dir, "results")

    # Load environment variables
    load_env(env_path)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("[ERROR] Gemini API Key is not set or placeholder remains in .env.")
        sys.exit(1)

    # Configure Gemini API
    genai.configure(api_key=api_key)

    # Select results file
    file_path = args.file
    if not file_path:
        # Find the latest JSON file in test/results/
        if not os.path.exists(results_dir):
            print(f"[ERROR] Results directory '{results_dir}' does not exist.")
            sys.exit(1)
        files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")]
        if not files:
            print("[ERROR] No JSON results files found in test/results/.")
            sys.exit(1)
        # Filter files by modification time
        file_path = max(files, key=os.path.getmtime)

    if not os.path.exists(file_path):
        print(f"[ERROR] Specified file not found: {file_path}")
        sys.exit(1)

    print(f"Loading results file for grading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "results" not in data:
        print("[ERROR] Invalid results file structure. Could not find 'results' key.")
        sys.exit(1)

    # Instantiate the judge model
    print(f"Configuring Judge Model '{args.model}'...")
    judge_model = genai.GenerativeModel(
        args.model,
        generation_config={"temperature": 0}
    )

    results_list = data["results"]
    total_items = len(results_list)
    print(f"Loaded {total_items} results. Commencing grading...")

    for idx, res in enumerate(results_list):
        exercise_id = res.get("exercise_id", f"Index {res.get('exercise_index')}")
        
        # Check if already graded and if force is not set
        if "score" in res and res["score"] is not None and not args.force:
            print(f"[{idx+1}/{total_items}] Skipping {exercise_id} (already graded with score {res['score']})")
            continue

        if res.get("status") != "success":
            print(f"[{idx+1}/{total_items}] Skipping {exercise_id} (status: {res.get('status')})")
            res["score"] = 0
            continue

        question = res.get("question", "")
        official_ans = res.get("official_answer", "")
        generated_ans = res.get("generated_answer", "")

        # Formulate prompt for judge
        prompt = f"""You are an expert mathematics professor grading a student's response against the official textbook answer key.

Task:
Compare the Student's Generated Answer against the Official Textbook Answer to determine if they are mathematically aligned.
Alignment means they:
- Prove or solve the same mathematical statement.
- Arrive at the same final solutions/conclusions.
- Exhibit the correct mathematical logic.
Minor differences in phrasing, LaTeX formatting, or choice of notation are completely acceptable.

--- EXERCISE QUESTION ---
{question}

--- OFFICIAL TEXTBOOK ANSWER ---
{official_ans}

--- STUDENT'S GENERATED ANSWER ---
{generated_ans}

Evaluate alignment.
If they are mathematically aligned, output exactly 1.
If they are not aligned (e.g., student got the wrong answer, incorrect steps, or different solution), output exactly 0.

Output ONLY '1' or '0' (no markdown, no other text).
"""

        print(f"[{idx+1}/{total_items}] Grading {exercise_id}...", end="", flush=True)

        score = 0
        max_attempts = 5
        base_sleep = 65
        success = False

        for attempt in range(max_attempts):
            try:
                response = judge_model.generate_content(prompt, request_options={"timeout": 60.0})
                text = response.text.strip()
                
                # Extract 0 or 1 using regex
                match = re.search(r'\b(0|1)\b', text)
                if match:
                    score = int(match.group(1))
                    success = True
                    break
                else:
                    print(f"\n  [Warning] Judge output '{text}' was not 0 or 1. Retrying...")
                    time.sleep(2)
            except exceptions.ResourceExhausted:
                sleep_time = base_sleep + (attempt * 10)
                print(f"\n  [Rate Limit] 429 ResourceExhausted in Judge. Sleeping {sleep_time} seconds...")
                time.sleep(sleep_time)
                print(f"[{idx+1}/{total_items}] Retrying grading for {exercise_id}...", end="", flush=True)
            except Exception as e:
                print(f"\n  [Error] Judge API error: {e}")
                time.sleep(5)
                break

        if success:
            res["score"] = score
            print(f" Score: {score}")
        else:
            res["score"] = 0
            print(f" Grading Failed (defaulted to 0)")

        # Save progress incrementally
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Compute and print metrics
    graded_scores = [r.get("score") for r in results_list if r.get("score") is not None]
    total_graded = len(graded_scores)
    correct = sum(1 for s in graded_scores if s == 1)

    # Update metadata if not present
    if "metadata" not in data:
        data["metadata"] = {}
    
    # Store grading info in metadata
    data["metadata"]["judge_model"] = args.model
    data["metadata"]["total_graded"] = total_graded
    data["metadata"]["correct_graded"] = correct
    data["metadata"]["accuracy"] = correct / total_graded if total_graded > 0 else 0.0

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n==============================================")
    print("Grading Completed!")
    print(f"Total graded questions: {total_graded}")
    print(f"Correct answers: {correct}")
    if total_graded > 0:
        accuracy = (correct / total_graded) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total_graded})")
    else:
        print("Accuracy: N/A")
    print("==============================================")

if __name__ == '__main__':
    try:
        main()
    finally:
        flush_traces()
