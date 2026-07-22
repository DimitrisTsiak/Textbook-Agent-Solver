import os
import json
import re
import chromadb
from parse_latex import remove_comments, extract_braced_content, clean_tex_formatting
from utils.latex_cleaner import clean_latex_for_embeddings

def load_env(env_path=".env"):
    """
    Manually loads key-value pairs from a .env file into os.environ.
    Ensures zero external dependency for environment loading.
    """
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.split('=', 1)
                        val = val.strip().strip('"').strip("'")
                        os.environ[key.strip()] = val

def chunk_chapter_file(filepath):
    """
    Reads the LaTeX chapter file, segments it by chapter/section/subsection,
    and splits the body of each segment into semantic paragraph chunks.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    text = remove_comments(raw_text)
    
    # Segment by structure
    pattern = re.compile(r'\\(chapter|section|subsection|subsectionoptional)\b')
    matches = list(pattern.finditer(text))
    
    chunks = []
    current_chapter = ""
    current_section = ""
    current_subsection = ""
    
    for i in range(len(matches)):
        m = matches[i]
        start_pos = m.start()
        cmd = m.group(1)
        
        # Get content of the header command
        header_text, next_idx = extract_braced_content(text, start_pos)
        if header_text is not None:
            cleaned_header = clean_tex_formatting(header_text)
            
            # Segment body goes until the next header (or end of file)
            segment_end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            segment_body = text[next_idx : segment_end].strip()
            
            # Update current hierarchy context
            if cmd == 'chapter':
                current_chapter = cleaned_header
                current_section = ""
                current_subsection = ""
            elif cmd == 'section':
                current_section = cleaned_header
                current_subsection = ""
            else:  # subsection / subsectionoptional
                current_subsection = cleaned_header
                
            # Split segment body into paragraphs by double newlines
            paragraphs = segment_body.split('\n\n')
            for p_idx, p in enumerate(paragraphs):
                p_clean = p.strip()
                if not p_clean:
                    continue
                # Skip exercise blocks as they are handled separately
                if '\\begin{exercises}' in p_clean or '\\end{exercises}' in p_clean:
                    continue
                # Skip tiny sentences or references
                if len(p_clean) < 20:
                    continue
                
                chunks.append({
                    'content': p_clean,
                    'metadata': {
                        'chapter': current_chapter,
                        'section': current_section,
                        'subsection': current_subsection,
                        'paragraph_index': p_idx,
                        'source': os.path.basename(filepath)
                    }
                })
    return chunks

def setup_chroma_db():
    # Load environment variables
    load_env()
    
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model_name = os.environ.get("EMBEDDING_MODEL_NAME", "models/text-embedding-004").strip()
    
    embedding_function = None
    suffix = "local"
    
    if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY":
        from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
        print(f"Gemini API key detected! Configuring database to use Gemini embeddings ('{model_name}').")
        embedding_function = GoogleGenerativeAiEmbeddingFunction(api_key=gemini_key, model_name=model_name)
        suffix = "gemini"
    else:
        print("Gemini API key not configured or placeholder remains in .env.")
        print("Falling back to local SentenceTransformers (all-MiniLM-L6-v2) for embeddings...")
        suffix = "local"

    # Define model-suffix collection names
    chunks_col_name = f"chapter_chunks_{suffix}"
    theory_col_name = f"textbook_theory_{suffix}"
    solutions_col_name = f"agent_solutions_{suffix}"

    db_path = "./chroma_db"
    print(f"Initializing persistent ChromaDB client at '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)
    
    # -------------------------------------------------------------
    # 1. Collection for Chapter Chunks
    # -------------------------------------------------------------
    print(f"\n--- Collection 1: {chunks_col_name} ---")
    try:
        client.delete_collection(chunks_col_name)
        print(f"Deleted existing '{chunks_col_name}' collection.")
    except Exception:
        pass
    
    if embedding_function:
        chapter_chunks = client.create_collection(chunks_col_name, embedding_function=embedding_function)
    else:
        chapter_chunks = client.create_collection(chunks_col_name)
        
    src_file = os.path.join("linear-algebra-master", "src", "gr", "gr1.tex")
    
    print(f"Chunking chapter file: {src_file}...")
    chunks = chunk_chapter_file(src_file)
    print(f"Generated {len(chunks)} paragraphs chunks.")
    
    # Store cleaned content in the document field for embeddings
    # Store raw LaTeX in metadata['raw_content'] for the LLM solver
    documents = [clean_latex_for_embeddings(c['content']) for c in chunks]
    metadatas = [{'raw_content': c['content'], **c['metadata']} for c in chunks]
    ids = [f"chunk_gr1_{idx}" for idx in range(len(chunks))]
    
    print(f"Adding chunks to '{chunks_col_name}'...")
    chapter_chunks.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Stored {len(ids)} chunks.")
    
    # -------------------------------------------------------------
    # 2. Collection for Textbook Theory
    # -------------------------------------------------------------
    print(f"\n--- Collection 2: {theory_col_name} ---")
    try:
        client.delete_collection(theory_col_name)
        print(f"Deleted existing '{theory_col_name}' collection.")
    except Exception:
        pass
    
    if embedding_function:
        textbook_theory = client.create_collection(theory_col_name, embedding_function=embedding_function)
    else:
        textbook_theory = client.create_collection(theory_col_name)
        
    json_path = "parsed_gr1.json"
    
    print(f"Loading parsed data from '{json_path}'...")
    with open(json_path, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
        
    theory_items = [item for item in parsed_data if item['category'] == 'theory']
    print(f"Found {len(theory_items)} theory items.")
    
    # Store clean_content in the document field for embeddings
    # Store raw LaTeX in metadata['raw_content'] for the LLM solver
    theory_docs = [item['clean_content'] for item in theory_items]
    theory_metadatas = []
    
    for idx, item in enumerate(theory_items):
        meta = {
            'raw_content': item['content'],
            'type': item['type'],
            'label': item['label'] if item['label'] else "",
            'title': item['title'] if item['title'] else "",
            'chapter': item['location']['chapter'],
            'section': item['location']['section'],
            'subsection': item['location']['subsection']
        }
        theory_metadatas.append(meta)
        
    theory_ids = []
    for idx, item in enumerate(theory_items):
        if item['label']:
            theory_ids.append(item['label'])
        else:
            theory_ids.append(f"theory_gr1_{idx}")
            
    print(f"Adding theory elements to '{theory_col_name}'...")
    textbook_theory.add(documents=theory_docs, metadatas=theory_metadatas, ids=theory_ids)
    print(f"Stored {len(theory_ids)} theory elements.")
    
    # -------------------------------------------------------------
    # 3. Collection for Agent Solutions
    # -------------------------------------------------------------
    print(f"\n--- Collection 3: {solutions_col_name} ---")
    try:
        client.delete_collection(solutions_col_name)
        print(f"Deleted existing '{solutions_col_name}' collection.")
    except Exception:
        pass
    
    if embedding_function:
        agent_solutions = client.create_collection(solutions_col_name, embedding_function=embedding_function)
    else:
        agent_solutions = client.create_collection(solutions_col_name)
        
    print(f"Pre-created empty '{solutions_col_name}' collection.")
    
    # -------------------------------------------------------------
    # Verification Query
    # -------------------------------------------------------------
    print(f"\n--- Verifying Retrieval on Active Suffix: '{suffix}' ---")
    query = "Gauss's Method and row operations"
    print(f"Querying active collections for: '{query}'...")
    
    # Query chapter chunks
    chunk_results = chapter_chunks.query(query_texts=[query], n_results=1)
    print(f"\nTop match from '{chunks_col_name}':")
    for doc, dist, meta in zip(chunk_results['documents'][0], chunk_results['distances'][0], chunk_results['metadatas'][0]):
        print(f"  - [Dist: {dist:.4f}] Section: {meta.get('subsection')}")
        print(f"    [CLEANED EMBEDDING DOCUMENT]:")
        print(f"      {doc}")
        print(f"    [RAW LATEX FOR LLM]:")
        raw_snippet = "\n".join(meta.get('raw_content', '').splitlines()[:3])
        print(f"      {raw_snippet}\n      ...")
        
    # Query theory elements
    theory_results = textbook_theory.query(query_texts=[query], n_results=1)
    print(f"\nTop match from '{theory_col_name}':")
    for doc, dist, meta in zip(theory_results['documents'][0], theory_results['distances'][0], theory_results['metadatas'][0]):
        print(f"  - [Dist: {dist:.4f}] Type: {meta.get('type')} | Title: {meta.get('title')} | Label: {meta.get('label')}")
        print(f"    [CLEANED EMBEDDING DOCUMENT]:")
        print(f"      {doc}")
        print(f"    [RAW LATEX FOR LLM]:")
        raw_snippet = "\n".join(meta.get('raw_content', '').splitlines()[:3])
        print(f"      {raw_snippet}\n      ...")

if __name__ == '__main__':
    setup_chroma_db()
