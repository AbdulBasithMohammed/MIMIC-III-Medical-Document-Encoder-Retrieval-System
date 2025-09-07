from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import google.generativeai as genai
from tqdm import tqdm
import os
import time
import re

def setup_clients():
    """Initialize Qdrant, SentenceTransformer, and Gemini clients."""
    # Initialize Qdrant client
    qdrant = QdrantClient("localhost", port=6333)
    
    # Initialize SentenceTransformer for initial retrieval
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize Gemini for reranking
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    
    return qdrant, model, gemini_model

def initial_retrieval(
    query: str,
    qdrant: QdrantClient,
    model: SentenceTransformer,
    collection_name: str = "mimic_ex_v2",
    initial_limit: int = 20,
    score_threshold: float = 0.3
) -> List[Dict]:
    """
    Perform initial retrieval using vector similarity search.
    
    Args:
        query: The search query string
        qdrant: Qdrant client instance
        model: SentenceTransformer model instance
        collection_name: Name of the Qdrant collection
        initial_limit: Number of documents to retrieve initially
        score_threshold: Minimum similarity score threshold
    
    Returns:
        List of retrieved documents with their scores and metadata
    """
    # Encode the query
    query_embedding = model.encode(query)
    
    # Search in Qdrant
    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=initial_limit,
        score_threshold=score_threshold
    )
    
    # Format results and remove duplicates
    seen_files = set()
    results = []
    
    for hit in search_results:
        file_path = hit.payload["file_path"]
        if file_path not in seen_files:
            seen_files.add(file_path)
            results.append({
                "text": hit.payload["text"],
                "file_path": file_path,
                "initial_score": hit.score
            })
    
    return results[:5]  # Return top 5 unique documents

def create_batch_reranking_prompt(query: str, documents: List[str]) -> str:
    """
    Create a prompt for the LLM to assess multiple documents' relevance at once.
    
    Args:
        query: The search query
        documents: List of document texts to evaluate
    
    Returns:
        Formatted prompt for the LLM
    """
    prompt = f"""For each of the following documents, rate their relevance to the query on a scale of 1-10. 
Consider the following criteria:
1. How directly the document answers the specific question
2. Whether it provides clear, actionable information
3. The specificity and relevance of the timing information provided
4. The context and completeness of the answer

Return the scores as a comma-separated list of numbers, one for each document in order.
Use the full range of scores (1-10) to differentiate between documents based on their relevance.

Query: {query}

Documents:
"""
    for i, doc in enumerate(documents, 1):
        prompt += f"\nDocument {i}:\n{doc}\n"
    
    prompt += "\nRelevance Scores (comma-separated, use full range 1-10):"
    return prompt

def rerank_documents(
    query: str,
    documents: List[Dict],
    gemini_model,
    final_limit: int = 5
) -> List[Dict]:
    """
    Rerank documents using Gemini LLM in batches.
    
    Args:
        query: The search query
        documents: List of documents to rerank
        gemini_model: Initialized Gemini model
        final_limit: Number of documents to return after reranking
    
    Returns:
        List of reranked documents with both initial and reranking scores
    """
    reranked_docs = []
    batch_size = 5  # Process 5 documents at a time
    
    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_texts = [doc["text"] for doc in batch]
        
        try:
            # Create prompt for the batch
            prompt = create_batch_reranking_prompt(query, batch_texts)
            
            # Get response from Gemini with retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = gemini_model.generate_content(prompt)
                    response_text = response.text.strip()
                    
                    # Parse comma-separated scores
                    scores = [float(score.strip()) for score in response_text.split(',')]
                    
                    if len(scores) == len(batch):
                        # Add scores to documents
                        for doc, raw_score in zip(batch, scores):
                            # Normalize from 1-10 scale to 0-1 scale
                            normalized_score = (raw_score - 1) / 9
                            doc["raw_rerank_score"] = raw_score
                            doc["rerank_score"] = normalized_score
                            reranked_docs.append(doc)
                        break  # Success, exit retry loop
                    else:
                        print(f"Unexpected number of scores in response: {response_text}")
                        continue
                    
                except Exception as e:
                    if "rate_limit_exceeded" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"Rate limit exceeded. Waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    print(f"Error reranking batch: {e}")
                    break  # Exit retry loop on other errors
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    # Sort by reranking score and take top K
    reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_docs[:final_limit]

def display_results(results: List[Dict], query: str):
    """Display search results in a formatted way."""
    print(f"\nQuery: '{query}'")
    if results:
        # First show all initial documents with their raw scores
        print("\nInitial Retrieval Results (all documents):")
        for idx, result in enumerate(results, 1):
            print(f"\nDocument {idx}:")
            print(f"Initial Score: {result['initial_score']:.4f}")
            print(f"Raw Rerank Score: {result['raw_rerank_score']:.1f}/10")
            print(f"Normalized Rerank Score: {result['rerank_score']:.4f}")
            print(f"File: {result['file_path']}")
            print("Text:")
            print(result['text'])  # Show full text
            print("-" * 80)
        
        # Then show the final top results after reranking
        print("\nFinal Top Results After Reranking:")
        for idx, result in enumerate(sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:5], 1):
            print(f"\nResult {idx}:")
            print(f"Initial Score: {result['initial_score']:.4f}")
            print(f"Raw Rerank Score: {result['raw_rerank_score']:.1f}/10")
            print(f"Normalized Rerank Score: {result['rerank_score']:.4f}")
            print(f"File: {result['file_path']}")
            print("Text:")
            print(result['text'])  # Show full text
            print("-" * 80)
    else:
        print("No sufficiently similar documents found.")

def main():
    # Initialize clients
    qdrant, model, gemini_model = setup_clients()

    # Single test query - natural medical question
    query = "When is the usual visit to the vascular surgery clinic after discharge from the hospital?"

    # Search parameters
    collection_name = "mimic_ex_v2"  # Using v2 collection
    initial_limit = 20  # Retrieve more documents initially
    final_limit = 5    # Return top 5 after reranking
    threshold = 0.3    # Initial similarity threshold

    try:
        # Initial retrieval
        print("Performing initial retrieval...")
        initial_results = initial_retrieval(
            query=query,
            qdrant=qdrant,
            model=model,
            collection_name=collection_name,
            initial_limit=initial_limit,
            score_threshold=threshold
        )
        
        if not initial_results:
            print("No documents found in initial retrieval.")
            return
        
        # Rerank results
        print("Reranking results...")
        reranked_results = rerank_documents(
            query=query,
            documents=initial_results,
            gemini_model=gemini_model,
            final_limit=final_limit
        )
        
        # Display results
        display_results(reranked_results, query)
        
    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == "__main__":
    main() 