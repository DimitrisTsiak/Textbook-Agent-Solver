import os
import re
import math
import json
import sys

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: List of dicts, e.g. [{"content": "clean text", "raw_content": "raw text", "metadata": {...}}]
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = []
        self.doc_terms = []
        self.df = {}
        self.idf = {}
        self.ndocs = len(corpus)
        self._initialize()

    def _tokenize(self, text):
        # simple word tokenization: lowercase and find alphanumeric tokens
        return re.findall(r'\w+', text.lower())

    def _initialize(self):
        total_len = 0
        for doc in self.corpus:
            tokens = self._tokenize(doc["content"])
            self.doc_lengths.append(len(tokens))
            self.doc_terms.append(tokens)
            total_len += len(tokens)
            
            # Record document frequencies for tokens
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
        
        self.avgdl = total_len / self.ndocs if self.ndocs > 0 else 0
        
        # Calculate IDF for each term
        for token, df in self.df.items():
            self.idf[token] = math.log(1.0 + (self.ndocs - df + 0.5) / (df + 0.5))

    def score(self, query, doc_idx):
        query_tokens = self._tokenize(query)
        doc_tokens = self.doc_terms[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        # Frequency of query tokens in the document
        tf = {}
        for token in doc_tokens:
            if token in query_tokens:
                tf[token] = tf.get(token, 0) + 1
                
        score = 0.0
        for token in query_tokens:
            if token not in self.idf:
                continue
            f = tf.get(token, 0)
            idf = self.idf[token]
            
            # BM25 term weight
            numerator = f * (self.k1 + 1.0)
            denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)
            
        return score

    def search(self, query, n_results=5):
        scores = []
        for i in range(self.ndocs):
            s = self.score(query, i)
            if s > 0:
                scores.append((s, self.corpus[i]))
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:n_results]


def load_textbook_corpus():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Temporarily append project root to sys.path to resolve root-level imports
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from utils.latex_cleaner import clean_latex_for_embeddings
    from setup_db import chunk_chapter_file
    
    corpus = []
    
    # Load parsed_gr1.json (theory items)
    json_path = os.path.join(project_root, "parsed_gr1.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        
        theory_items = [item for item in parsed_data if item['category'] == 'theory']
        for item in theory_items:
            corpus.append({
                "content": item['clean_content'],
                "raw_content": item['content'],
                "metadata": {
                    "type": item['type'],
                    "title": item['title'] if item['title'] else "",
                    "label": item['label'] if item['label'] else "",
                    "chapter": item['location']['chapter'],
                    "section": item['location']['section'],
                    "subsection": item['location']['subsection']
                }
            })
            
    # Load chapter chunks
    tex_path = os.path.join(project_root, "linear-algebra-master", "src", "gr", "gr1.tex")
    if os.path.exists(tex_path):
        chunks = chunk_chapter_file(tex_path)
        for c in chunks:
            corpus.append({
                "content": clean_latex_for_embeddings(c['content']),
                "raw_content": c['content'],
                "metadata": {
                    "type": "chapter_paragraph",
                    "chapter": c['metadata']['chapter'],
                    "section": c['metadata']['section'],
                    "subsection": c['metadata']['subsection']
                }
            })
            
    return corpus


class BM25Searcher:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("[BM25] Initializing corpus and searcher instance...")
            corpus = load_textbook_corpus()
            cls._instance = BM25(corpus)
            print(f"[BM25] Loaded searcher with {len(corpus)} documents.")
        return cls._instance


if __name__ == '__main__':
    # Simple self-test
    print("Testing BM25Searcher...")
    searcher = BM25Searcher.get_instance()
    results = searcher.search("Gauss's Method", n_results=3)
    
    print(f"\nFound {len(results)} results:")
    for score, doc in results:
        print(f"\n--- [Score: {score:.4f}] Type: {doc['metadata']['type']} ---")
        print(f"Content: {doc['content'][:150]}...")
