from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
import os

# Medical stop words to exclude from scoring
MEDICAL_STOP_WORDS = {
    'what', 'are', 'the', 'for', 'with', 'and', 'in', 'on', 'at', 'to', 'of',
    'a', 'an', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might',
    'must', 'can', 'could', 'that', 'this', 'these', 'those', 'which', 'who',
    'whom', 'whose', 'where', 'when', 'why', 'how'
}

def calculate_keyword_score(query_keywords: set, doc_keywords: set, matching_keywords: set) -> float:
    """
    Calculate a weighted score for keyword matching.
    
    Args:
        query_keywords: Set of keywords from the query
        doc_keywords: Set of keywords from the document
        matching_keywords: Set of matching keywords
    
    Returns:
        Weighted score between 0 and 1
    """
    # Remove stop words from consideration
    query_keywords = query_keywords - MEDICAL_STOP_WORDS
    matching_keywords = matching_keywords - MEDICAL_STOP_WORDS
    
    if not query_keywords:  # If all keywords were stop words
        return 0.0
    
    # Base score from keyword matches
    base_score = len(matching_keywords) / len(query_keywords)
    
    # Additional weight for medical terms
    medical_terms = {'medications', 'prescribed', 'patients', 'congestive', 'heart', 'failure'}
    medical_matches = matching_keywords.intersection(medical_terms)
    
    # Add bonus for medical term matches
    medical_bonus = len(medical_matches) * 0.1  # 0.1 bonus per medical term
    
    # Final score is base score plus medical bonus, capped at 1.0
    return min(base_score + medical_bonus, 1.0)

def keyword_match_documents(
    query: str,
    collection_name: str = "mimic_ex",
    limit: int = 5,
    threshold: float = 0.3
) -> List[Dict]:
    """
    Retrieve documents using keyword matching instead of cosine similarity.
    
    Args:
        query: The search query
        collection_name: Name of the Qdrant collection
        limit: Maximum number of results to return
        threshold: Minimum score threshold for results
    
    Returns:
        List of matching documents with their scores
    """
    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)
    
    # Get all documents from the collection
    all_documents = client.scroll(
        collection_name=collection_name,
        limit=1000,  # Adjust based on your collection size
        with_payload=True,
        with_vectors=False
    )[0]
    
    # Calculate keyword match scores
    results = []
    query_keywords = set(query.lower().split())
    
    for doc in all_documents:
        # Get document text from payload
        doc_text = doc.payload.get("text", "").lower()
        
        # Calculate keyword match score
        doc_keywords = set(doc_text.split())
        matching_keywords = query_keywords.intersection(doc_keywords)
        
        # Calculate weighted score
        score = calculate_keyword_score(query_keywords, doc_keywords, matching_keywords)
        
        if score >= threshold:
            results.append({
                "file_path": doc.payload.get("file_path", ""),
                "text": doc_text,
                "score": score,
                "matching_keywords": list(matching_keywords)
            })
    
    # Sort by score and limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

def display_results(results: List[Dict], query: str):
    """Display search results in a formatted way."""
    print(f"\nQuery: '{query}'")
    if results:
        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Matching Keywords: {', '.join(result['matching_keywords'])}")
            print(f"File: {result['file_path']}")
            print("Text:")
            print(result['text'])
    else:
        print("No sufficiently similar documents found.")

if __name__ == "__main__":
    # Test query
    test_query = "When is the usual visit to the vascular surgery clinic after discharge from the hospital?"
    
    try:
        # Perform keyword matching search
        results = keyword_match_documents(
            query=test_query,
            collection_name="mimic_ex_v3",
            limit=5,
            threshold=0.3
        )
        
        # Display results
        display_results(results, test_query)
        
    except Exception as e:
        print(f"Error during search: {str(e)}") 