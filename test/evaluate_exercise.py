import os
import json
import google.generativeai as genai

# =====================================================================
# CONFIGURATION
# Change this index to evaluate different exercises (1 to 62)
# =====================================================================
EXERCISE_INDEX = 2
#MODEL_NAME = "gemini-2.5-flash-lite"
MODEL_NAME = "gemma-4-26b-a4b-it"  
# =====================================================================

def load_env(env_path="../.env"):
    """
    Loads env variables from .env file (looking up one level since this script is in test/).
    """
    env_vars = {}
    # First check in current directory, then parent directory
    paths_to_check = [".env", env_path]
    for p in paths_to_check:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, val = line.split('=', 1)
                            val = val.strip().strip('"').strip("'")
                            env_vars[key.strip()] = val
            break
    return env_vars

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
    env = load_env()
    api_key = env.get("GEMINI_API_KEY", "").strip()
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("\n[ERROR] Gemini API Key is not set or placeholder remains in .env.")
        print("Please edit the .env file in the project root folder and insert a valid GEMINI_API_KEY.")
        return
        
    # Find the exercise
    json_path = os.path.join("..", "parsed_gr1.json")
    # Fallback to local if running from project root
    if not os.path.exists(json_path):
        json_path = "parsed_gr1.json"
        
    print(f"Retrieving exercise {EXERCISE_INDEX} from '{json_path}'...")
    exercise = find_exercise_by_index(json_path, EXERCISE_INDEX)
    
    if not exercise:
        print(f"[ERROR] Could not find exercise with index {EXERCISE_INDEX} in the JSON file.")
        return
        
    question_text = exercise['question']
    official_answer = exercise['answer']
    
    # Configure Gemini LLM
    print(f"Configuring {MODEL_NAME} API client...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""You are a mathematics professor. Solve the following linear algebra exercise step-by-step.
Do not use any external textbook context, solve it from first principles.

Exercise:
{question_text}

Provide your complete mathematical solution. Keep your explanation concise but mathematically rigorous.
"""
    
    print("Calling Gemini LLM to generate answer...")
    try:
        response = model.generate_content(prompt)
        llm_answer = response.text
        print("LLM generated answer successfully.")
    except Exception as e:
        print(f"[ERROR] Gemini API invocation failed: {e}")
        return
        
    # Output to markdown file
    output_dir = "test"
    if not os.path.exists(output_dir):
        # Handle if already inside test/
        if os.path.basename(os.getcwd()) == "test":
            output_dir = "."
        else:
            os.makedirs(output_dir)
            
    out_filename = f"comparison_exercise_{EXERCISE_INDEX}.md"
    out_path = os.path.join(output_dir, out_filename)
    
    md_content = f"""# Comparison for Exercise {EXERCISE_INDEX}

## Location
* **Chapter**: {exercise['location']['chapter']}
* **Section**: {exercise['location']['section']}
* **Subsection**: {exercise['location']['subsection']}
* **ID**: `{exercise['id']}`
* **Recommended**: {exercise['recommended']}
* **Puzzle**: {exercise['puzzle']}

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
