"""Database query command implementation."""

import os
import pinecone
from shared import utils


def merge_hits(h1, h2):
    """Get the unique hits from two search results and return them as single list"""
    # Deduplicate by _id
    deduped_hits = {hit['_id']: hit for hit in h1 + h2}.values()

    # Sort by _score descending
    sorted_hits = sorted(deduped_hits, key=lambda x: x['_score'], reverse=True)

    return sorted_hits


def semantic_search(query_text: str, dense_index: pinecone.Pinecone.Index, top_k: int = 10):
    """
    Perform semantic search using only the dense index.
    
    Args:
        query_text: Search query string
        dense_index: Pinecone dense index
        top_k: Number of results to return
        
    Returns:
        Pinecone search results hits
    """
    results = dense_index.search(
        namespace="__default__",
        query={
            "top_k": top_k,
            "inputs": {
                "text": query_text
            }
        }
    )
    
    return results.result.hits if hasattr(results, 'result') and hasattr(results.result, 'hits') else []


def lexical_search(query_text: str, sparse_index: pinecone.Pinecone.Index, top_k: int = 10):
    """
    Perform lexical search using only the sparse index.
    
    Args:
        query_text: Search query string
        sparse_index: Pinecone sparse index
        top_k: Number of results to return
        
    Returns:
        Pinecone search results hits
    """
    results = sparse_index.search(
        namespace="__default__",
        query={
            "top_k": top_k,
            "inputs": {
                "text": query_text
            }
        }
    )
    
    return results.result.hits if hasattr(results, 'result') and hasattr(results.result, 'hits') else []


def hybrid_search(query_text: str, dense_index: pinecone.Pinecone.Index, sparse_index: pinecone.Pinecone.Index, top_k: int = 10):
    """
    Perform hybrid search by querying both dense and sparse indexes, then reranking.
    
    Args:
        query_text: Search query string
        dense_index: Pinecone dense index
        sparse_index: Pinecone sparse index
        top_k: Number of results to return
        
    Returns:
        List of reranked search results hits
    """
    # Query both indexes - get more results than needed for better reranking
    sparse_results = lexical_search(query_text, sparse_index, top_k * 2)
    dense_results = semantic_search(query_text, dense_index, top_k * 2)

    # Merge the results
    merged_results = merge_hits(sparse_results, dense_results)
    
    if not merged_results:
        return []
    
    # Limit to 100 results for reranking (Pinecone limit)
    max_rerank = min(100, len(merged_results))
    merged_results = merged_results[:max_rerank]
    
    # Get Pinecone client for reranking
    pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Prepare results for reranking - extract text from fields
    formatted_merged = [{'_id': hit['_id'], 'text': hit['fields']['text']} for hit in merged_results]
    
    # Rerank using cohere-rerank-3.5
    rerank_response = pc.inference.rerank(
        model="cohere-rerank-3.5",
        query=query_text,
        documents=formatted_merged,
        top_n=top_k,
        return_documents=True
    )
    
    # Create mapping of vector_id -> rerank_score
    rerank_scores = {rerank_result.document['_id']: rerank_result.score 
                     for rerank_result in rerank_response.data}
    
    # Add rerank scores to merged_results
    for hit in merged_results:
        if hit['_id'] in rerank_scores:
            hit['_score'] = rerank_scores[hit['_id']]
    
    # Sort merged_results by rerank score (descending) and return top_k
    reranked_results = sorted(
        merged_results,
        key=lambda hit: hit.get('_score', 0),
        reverse=True
    )[:top_k]
    
    return reranked_results


def format_results(results):
    """
    Format search results for display.
    
    Args:
        results: List of search result hits
        
    Returns:
        Formatted string representation
    """
    if not results:
        return "No results found."
    
    output = []
    for i, match in enumerate(results, 1):
        # Access as dict (consistent with merge_hits and hybrid_search)
        vector_id = match.get('_id', None)
        score = match.get('_score', None)
        fields = match.get('fields', {})
        
        # Extract metadata from fields
        appid = fields.get('appid', None)
        name = fields.get('name', '')
        recommendationid = fields.get('recommendationid', None)
        text = fields.get('text', '')
        
        # Convert appid and recommendationid to integers for display (remove .0)
        appid_display = int(appid) if appid is not None else None
        recommendationid_display = int(recommendationid) if recommendationid is not None else None
        
        # Determine result type
        has_recommendationid = recommendationid is not None
        result_type = "Review" if has_recommendationid else "Game"
        
        # Handle score formatting
        score_str = f"{score:.4f}" if score is not None else "N/A"
        
        # Clean up text for display
        if text:
            text_cleaned = ' '.join(text.split())
            text_preview = text_cleaned[:500] + "..." if len(text_cleaned) > 500 else text_cleaned
        else:
            text_preview = ''
        
        output.append(f"\n{i}. Score: {score_str} ({result_type})")
        output.append(f"   ID: {vector_id}")
        output.append(f"   App ID: {appid_display if appid_display is not None else 'N/A'}")
        output.append(f"   Name: {name}")
        if has_recommendationid:
            output.append(f"   Review ID: {recommendationid_display}")
        if text_preview:
            output.append(f"   Text: {text_preview}")
    
    return "\n".join(output)


def main(query: str, mode: str = 'hybrid', top_k: int = 10):
    """
    Execute the database-query command.
    
    Args:
        query: Search query string
        mode: Search mode - 'hybrid', 'semantic', or 'lexical'
        top_k: Number of results to return
    """
    # Get indexes
    dense_index, sparse_index = utils.get_indexes()
    
    # Perform search based on mode
    if mode == 'semantic':
        print(f"Performing semantic search for: '{query}'")
        matches = semantic_search(query, dense_index, top_k)
    elif mode == 'lexical':
        print(f"Performing lexical search for: '{query}'")
        matches = lexical_search(query, sparse_index, top_k)
    else:  # hybrid (default)
        print(f"Performing hybrid search for: '{query}'")
        matches = hybrid_search(query, dense_index, sparse_index, top_k)
    
    # Display results
    print(f"\nFound {len(matches)} results:")
    print(format_results(matches))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m database.query <query> [-m|--mode hybrid|semantic|lexical] [--top-k N]")
        sys.exit(1)
    
    # Simple argument parsing for direct execution
    query = sys.argv[1]
    mode = 'hybrid'
    top_k = 10
    
    if '-m' in sys.argv:
        idx = sys.argv.index('-m')
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
    elif '--mode' in sys.argv:
        idx = sys.argv.index('--mode')
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
    
    if '--top-k' in sys.argv:
        idx = sys.argv.index('--top-k')
        if idx + 1 < len(sys.argv):
            top_k = int(sys.argv[idx + 1])
    
    main(query, mode, top_k)
