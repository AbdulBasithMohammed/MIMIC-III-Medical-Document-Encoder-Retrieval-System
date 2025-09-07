import os
import json
from datetime import datetime
from bleurt import score as bleurt_score
from cosine_retrieval import setup_clients, search_mimic_documents
from keyword_retrieval import keyword_match_documents as keyword_search
from reranking_retrieval import initial_retrieval, rerank_documents, setup_clients as setup_reranking_clients
from sentence_transformers import SentenceTransformer

def calculate_cosine_similarity(query: str, text: str, model: SentenceTransformer) -> float:
    """
    Calculate cosine similarity between query and text.
    
    Args:
        query: The search query
        text: The document text
        model: SentenceTransformer model
    
    Returns:
        Cosine similarity score
    """
    query_embedding = model.encode(query)
    text_embedding = model.encode(text)
    return float(query_embedding @ text_embedding.T)

def evaluate_retrieval(query: str, collection_name: str, retrieval_method: str = "cosine"):
    """
    Evaluate retrieval results using both cosine similarity and BLEURT scores.
    
    Args:
        query: The search query string
        collection_name: Name of the Qdrant collection to search in
        retrieval_method: One of "cosine", "keyword", or "reranking"
    
    Returns:
        Dictionary containing the result and scores
    """
    # Initialize BLEURT scorer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bleurt_dir = os.path.join(current_dir, "bleurt")
    checkpoint = os.path.join(bleurt_dir, "BLEURT-20")
    scorer = bleurt_score.BleurtScorer(checkpoint)
    
    # Initialize model for cosine similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize result variables
    result = None
    cosine_score = None
    rerank_score = None
    
    # Initialize clients based on retrieval method
    if retrieval_method == "cosine":
        qdrant, _ = setup_clients()
        results = search_mimic_documents(
            query=query,
            qdrant=qdrant,
            model=model,
            collection_name=collection_name,
            limit=1,
            score_threshold=0.3
        )
        if results:
            result = results[0]
            cosine_score = result["score"]
    elif retrieval_method == "keyword":
        qdrant, _ = setup_clients()
        results = keyword_search(
            query=query,
            collection_name=collection_name,
            limit=1
        )
        if results:
            result = results[0]
            cosine_score = calculate_cosine_similarity(query, result["text"], model)
    elif retrieval_method == "reranking":
        qdrant, _, gemini_model = setup_reranking_clients()
        # Get initial results
        initial_results = initial_retrieval(
            query=query,
            qdrant=qdrant,
            model=model,
            collection_name=collection_name,
            initial_limit=20,
            score_threshold=0.3
        )
        # Rerank results
        reranked_results = rerank_documents(
            query=query,
            documents=initial_results,
            gemini_model=gemini_model,
            final_limit=1
        )
        if reranked_results:
            result = reranked_results[0]
            cosine_score = calculate_cosine_similarity(query, result["text"], model)
            rerank_score = result["rerank_score"]
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")
    
    if result is None:
        return {
            "query": query,
            "collection": collection_name,
            "retrieval_method": retrieval_method,
            "result": None,
            "cosine_score": None,
            "rerank_score": None,
            "bleurt_score": None
        }
    
    # Get BLEURT score
    bleurt_score_value = scorer.score(
        references=[query],
        candidates=[result["text"]]
    )[0]
    
    return {
        "query": query,
        "collection": collection_name,
        "retrieval_method": retrieval_method,
        "result": result,
        "cosine_score": cosine_score,
        "rerank_score": rerank_score,
        "bleurt_score": float(bleurt_score_value)
    }

def display_results(results: list, query: str):
    """Display search results with both cosine and BLEURT scores."""
    print("\nQuery:", query)
    print("\nResults with scores:")
    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(f"Retrieval Method: {result['retrieval_method']}")
        print(f"Collection: {result['collection']}")
        if result['rerank_score'] is not None:
            print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(f"Cosine Score: {result['cosine_score']:.4f}")
        print(f"BLEURT Score: {result['bleurt_score']:.4f}")
        print(f"File: {result['result']['file_path']}")
        print("Text:", result["result"]["text"][:200] + "...")  # Show first 200 chars
        print("-" * 80)

def save_results(results, filename=None):
    """Save evaluation results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    # Test queries
    test_queries = [
        # Original queries
        "When is the usual visit to the vascular surgery clinic after discharge from the hospital?",
        "What are the common complications after heart surgery?",
        "How long does it take to recover from a knee replacement surgery?",
        "What medications are typically prescribed after a stroke?",
        "What are the dietary restrictions after gallbladder surgery?",
        "What are the warning signs of a potential blood clot after surgery?",
        "How should patients manage pain after spinal fusion surgery?",
        "What are the typical rehabilitation exercises after rotator cuff repair?",
        "What are the signs of infection at a surgical incision site?",
        "How soon can patients resume normal activities after hip replacement surgery?"
    ]
    
    # Collections to evaluate
    collections = ["mimic_ex", "mimic_ex_v2", "mimic_ex_v3", "mimic_ex_v4"]
    
    # Retrieval methods
    retrieval_methods = ["cosine", "keyword", "reranking"]
    
    # Store all results
    all_results = []
    
    # Run evaluation
    for query in test_queries:
        for collection in collections:
            for method in retrieval_methods:
                print(f"\nEvaluating query: {query[:50]}...")
                print(f"Collection: {collection}")
                print(f"Method: {method}")
                
                result = evaluate_retrieval(query, collection, method)
                all_results.append(result)
                
                # Print current result
                if result['result'] is not None:
                    print(f"Cosine Score: {result['cosine_score']:.4f}")
                    if result['rerank_score'] is not None:
                        print(f"Rerank Score: {result['rerank_score']:.4f}")
                    print(f"BLEURT Score: {result['bleurt_score']:.4f}")
                else:
                    print("No results found")
                print("-" * 80)
    
    # Save results
    save_results(all_results)

if __name__ == "__main__":
    main()