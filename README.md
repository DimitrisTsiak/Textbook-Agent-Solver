# Linear Algebra Textbook Exercise Solver

This project builds an autonomous textbook exercise solver for Jim Hefferon's undergraduate textbook *Linear Algebra*. The system parses the raw LaTeX source code, indexes the content (definitions, theorems, examples) in **ChromaDB**, and implements a tool-augmented agent architecture using the **Google Gemini API**. The agent leverages semantic **RAG**, keyword search via **BM25**, and a symbolic/numeric Python calculation tool using **NumPy** and **SymPy** to solve exercises sequentially. Correctness is evaluated via an automated **LLM-as-a-Judge** pipeline that compares solver outputs directly against official answers.

## Book Source
Source LaTeX files are located in `linear-algebra-master/`. The book and answer manuals are available at [hefferon.net/linearalgebra](https://hefferon.net/linearalgebra).

## Setup & Database Build

Install dependencies:
```bash
pip install chromadb google-generativeai
```

Configure your API key in `.env`:
```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
EMBEDDING_MODEL_NAME=embedding_model_name
```
*If no API key is provided, the indexer falls back to local SentenceTransformers 

### Parsing and Indexing

Extract textbook elements and exercises:
```bash
python parse_latex.py
```
This generates a structured JSON database `parsed_gr1.json`.

Build and populate the Chroma vector database:
```bash
python setup_db.py
```
This chunks the text, cleans the LaTeX formatting for high-quality embedding matching, and indexes everything into `./chroma_db` (storing the raw LaTeX in metadata to be supplied to the solver model).

### Evaluation

To test raw model performance on an exercise and compare it to the textbook's answer, run:
```bash
python test/evaluate_exercise.py
```
This outputs a side-by-side comparison in `test/comparison_exercise_{index}.md`. You can configure the targeted exercise index and model in the script.

## Evaluation of tool usage and RAG on gr2 chapter exercises using an LLM as Judge to compare the solution with the textbook solutions

- **LLM Model**: `gemini-3.5-flash-lite`
- **RAG**: Fetches textbook contents and theorems relevant to the exercise
- **Search Tool**: Fetches textbook contents based on keywords
- **Calc Tool**: Runs executable Python code using `numpy` and `sympy` to execute symbolic and numeric linear algebra operations
- **Judge Model**: `gemini-3.5-flash-lite`

| Configuration | CLI Arguments | LLM-as-a-Judge Accuracy (Linear Geometry chapter 42 exercises) | 
| :--- | :--- | :--- |
| **1. Base Model** | `python test/batch_evaluate.py --no-rag` | **90.48%** (38/42)   |
| **2. + RAG** | `python test/batch_evaluate.py` | **90.48%** (38/42) |
| **3. + Search Tool** | `python test/batch_evaluate.py --use-search` | **90.48%** (38/42) |
| **4. + Linear Algebra Tool** | `python test/batch_evaluate.py --use-search --use-calc` | **92.86%** (39/42) |


