import os
import google.generativeai as genai

def load_env(env_path="../.env"):
    """
    Loads env variables from .env file (looking up one level since this script is in test/).
    """
    env_vars = {}
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

def main():
    print("Loading environment variables...")
    env = load_env()
    api_key = env.get("GEMINI_API_KEY", "").strip()
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("\n[ERROR] Gemini API Key is not set or placeholder remains in .env.")
        print("Please edit the .env file in the project root folder and insert a valid GEMINI_API_KEY.")
        return
        
    print("Configuring Google Generative AI client...")
    genai.configure(api_key=api_key)
    
    print("Fetching available models...\n")
    try:
        models = list(genai.list_models())
        print(f"Successfully retrieved {len(models)} models:")
        
        # Group models by type for readability
        for m in models:
            # Print model identifier and supported methods
            methods = ", ".join(m.supported_generation_methods)
            print(f"  - Name: {m.name}")
            print(f"    Description: {m.description}")
            print(f"    Methods: {methods}\n")
            
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")

if __name__ == '__main__':
    main()
