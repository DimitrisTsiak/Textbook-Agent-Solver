import os
import sys
import google.generativeai as genai

# Resolve paths to allow importing from tools and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env import load_env

def main():
    print("Loading environment variables...")
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
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
