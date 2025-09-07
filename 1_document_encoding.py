# docker run -p 6333:6333 qdrant/qdrant

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import os
import glob
from tqdm import tqdm
import numpy as np

# Initialize Qdrant client
qdrant = QdrantClient("localhost", port=6333)

# Create a collection with the correct vector size (384 for all-MiniLM-L6-v2)
collection_name = "mimic_ex"
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name)
    
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_mimic_files(directory_path: str, batch_size: int = 1000):
    """
    Process MIMIC-III text files and store them in Qdrant.
    
    Args:
        directory_path: Path to the directory containing MIMIC-III text files
        batch_size: Number of files to process in each batch
    """
    # Get all text files
    file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
    total_files = len(file_paths)
    print(f"Found {total_files} MIMIC-III files to process")

    # Process files in batches
    for i in tqdm(range(0, total_files, batch_size), desc="Processing batches"):
        batch_files = file_paths[i:i + batch_size]
        batch_points = []
        
        # Process each file in the batch
        for file_path in tqdm(batch_files, desc="Processing files in batch", leave=False):
            try:
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:  # Skip empty files
                    continue
                
                # Create embedding
                embedding = model.encode(content)
                
                # Create point for Qdrant
                point = models.PointStruct(
                    id=i + len(batch_points),  # Unique ID
                    vector=embedding.tolist(),
                    payload={
                        "text": content,
                        "file_path": file_path
                    }
                )
                batch_points.append(point)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Upload batch to Qdrant
        if batch_points:
            qdrant.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            print(f"Uploaded batch of {len(batch_points)} documents")

def search_mimic_documents(query: str, limit: int = 5):
    """
    Search for similar documents in the MIMIC-III database.
    
    Args:
        query: The search query string
        limit: Number of results to return
    
    Returns:
        List of similar documents with their scores
    """
    # Encode the query
    query_embedding = model.encode(query)
    
    # Search in Qdrant
    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=limit,
        score_threshold=0.3
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

if __name__ == "__main__":
    # Process MIMIC-III files
    mimic_dir = "datasets/mimic_ex"
    process_mimic_files(mimic_dir)
    
    # Test some medical queries
    test_queries = [
        "Patient with heart failure",
        "Diabetes diagnosis",
        "Blood pressure measurement",
        "Chest pain symptoms",
        "Medication administration"
    ]
    
    print("\nTesting retrieval with medical queries:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_mimic_documents(query)
        if results:
            for result in results:
                print(f"\nSimilar document (score: {result['score']:.4f}):")
                print(f"File: {result['file_path']}")
                print(f"Text: {result['text'][:200]}...")  # Show first 200 chars
        else:
            print("No sufficiently similar documents found.")