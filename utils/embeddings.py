import time
from chromadb import EmbeddingFunction
from google.api_core import exceptions
import google.generativeai as genai

class CustomGeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        
    def __call__(self, input: list) -> list:
        # If input is a single string instead of a list
        if isinstance(input, str):
            input = [input]
            
        batch_size = 45
        embeddings = []
        for i in range(0, len(input), batch_size):
            batch = input[i:i+batch_size]
            print(f"  [Embedding Service] Embedding batch of {len(batch)} items (index {i} to {i+len(batch)})...")
            
            # Attempt query with automatic retry on rate limit
            while True:
                try:
                    response = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    embeddings.extend(response['embedding'])
                    break  # Success
                except exceptions.ResourceExhausted:
                    print("  [Embedding Service] Rate limit reached (429 ResourceExhausted). Sleeping 65 seconds to refill quota...")
                    time.sleep(65)
                except Exception as e:
                    print(f"  [Embedding Service] Unexpected error: {e}")
                    raise e
            
            if i + batch_size < len(input):
                print("  [Embedding Service] Rate limit safeguard: Sleeping 65 seconds before next batch...")
                time.sleep(65)
                
        return embeddings
