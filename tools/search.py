from utils.tracing import observe, update_current_observation

@observe(as_type="retriever", name="bm25-search-tool")
def search_textbook(query: str) -> str:
    """
    Search the linear algebra textbook for keywords and retrieve relevant text chunks or theory items.
    
    Args:
        query: The keyword search query.
        
    Returns:
        A formatted string containing the top matching search results.
    """
    print(f"\n[Tool Execution] search_textbook called with query: '{query}'")
    update_current_observation(input={"query": query})
    try:
        from utils.bm25 import BM25Searcher
        searcher = BM25Searcher.get_instance()
        results = searcher.search(query, n_results=3)
        if not results:
            res_str = "No matching textbook text found for that query."
            update_current_observation(output={"results_count": 0, "text": res_str})
            return res_str
        
        output_parts = []
        for idx, (score, doc) in enumerate(results):
            meta = doc['metadata']
            title = f" ({meta['title']})" if meta.get('title') else ""
            label = f" (Label: {meta['label']})" if meta.get('label') else ""
            header = f"Result #{idx+1} [Type: {meta['type'].upper()}]{title}{label}"
            output_parts.append(f"--- {header} ---\n{doc['raw_content']}")
        formatted_output = "\n\n".join(output_parts)
        update_current_observation(output={"results_count": len(results), "text": formatted_output})
        return formatted_output
    except Exception as e:
        err_msg = f"Error executing search: {str(e)}"
        update_current_observation(output={"error": err_msg})
        return err_msg
