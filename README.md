# Linear Algebra Textbook Exercise Solver

This project builds a textbook exercise solver for Jim Hefferon's undergraduate textbook *Linear Algebra*. The system parses the raw LaTeX source code, indexes the content (definitions, theorems, examples) in ChromaDB, and uses RAG to solve exercises sequentially, building on solutions to prior exercises.

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
