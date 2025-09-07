from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def setup_clients():
    """Initialize Qdrant and SentenceTransformer clients."""
    qdrant = QdrantClient("localhost", port=6333)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return qdrant, model

def search_mimic_documents(
    query: str,
    qdrant: QdrantClient,
    model: SentenceTransformer,
    collection_name: str = "mimic_ex",  # Default to v1 collection
    limit: int = 5,
    score_threshold: float = 0.3
) -> List[Dict]:
    """
    Search for similar documents in the MIMIC-III database.
    
    Args:
        query: The search query string
        qdrant: Qdrant client instance
        model: SentenceTransformer model instance
        collection_name: Name of the Qdrant collection to search in
        limit: Number of results to return
        score_threshold: Minimum similarity score threshold
    
    Returns:
        List of similar documents with their scores and metadata
    """
    # Encode the query
    query_embedding = model.encode(query)
    
    # Search in Qdrant
    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=limit,
        score_threshold=score_threshold
    )
    
    # Format results
    results = []
    for hit in search_results:
        results.append({
            "text": hit.payload["text"],
            "file_path": hit.payload["file_path"],
            "score": hit.score
        })
    
    return results

def display_results(results: List[Dict], query: str):
    """Display search results in a formatted way."""
    print(f"\nQuery: '{query}'")
    if results:
        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx} (score: {result['score']:.4f}):")
            print(f"File: {result['file_path']}")
            print("Text:")
            print(result['text'])  # Show first 500 chars
            print("-" * 80)
    else:
        print("No sufficiently similar documents found.")

def main():
    # Initialize clients
    qdrant, model = setup_clients()

    # Single test query - natural medical question
    query = "When is the usual visit to the vascular surgery clinic after discharge from the hospital?"

    # Search parameters
    collection_name = "mimic_ex"  # Using v1 collection
    limit = 5    # Number of results to return
    threshold = 0.3    # Similarity threshold

    try:
        # Perform search
        print("Searching for relevant documents...")
        results = search_mimic_documents(
            query=query,
            qdrant=qdrant,
            model=model,
            collection_name=collection_name,
            limit=limit,
            score_threshold=threshold
        )
        
        # Display results
        display_results(results, query)
        
    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == "__main__":
    main() 