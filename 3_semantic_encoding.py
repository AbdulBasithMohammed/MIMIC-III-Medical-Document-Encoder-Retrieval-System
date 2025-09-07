import os
import glob
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm

class SentenceTransformerEmbeddings:
    """Wrapper class to adapt SentenceTransformer for SemanticChunker"""
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert encode method to embed_documents for compatibility"""
        return self.model.encode(texts).tolist()

def encode_text_files(
    directory_path: str,
    collection_name: str = "mimic_ex_v3",
    chunk_size: int = 1000,
    batch_size: int = 100,
    file_pattern: str = "*.txt",
    recreate_collection: bool = False
) -> None:
    """
    Process MIMIC-III patient history text files with semantic chunking and store in Qdrant.
    
    Args:
        directory_path: Path to the directory containing text files
        collection_name: Name of the Qdrant collection
        chunk_size: Target size for chunks
        batch_size: Number of files to process in each batch
        file_pattern: Pattern to match text files
        recreate_collection: Whether to recreate the collection if it exists
    """
    # Initialize clients
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model)
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    qdrant = QdrantClient("localhost", port=6333)
    
    # Handle collection creation/recreation
    try:
        if recreate_collection:
            # Delete existing collection if it exists
            try:
                qdrant.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                print(f"Error deleting collection: {e}")
        
        # Create new collection
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Using existing collection: {collection_name}")
        else:
            print(f"Error creating collection: {e}")
            return
    
    # Get all text files
    full_path = os.path.abspath(directory_path)
    file_paths = glob.glob(os.path.join(full_path, file_pattern))
    print(f"Found {len(file_paths)} files to process")
    
    if not file_paths:
        print("No files found. Please check the directory path and file pattern.")
        return
    
    # Process files in batches
    total_chunks = 0
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Processing files"):
        batch_files = file_paths[i:i + batch_size]
        points = []
        
        for file_path in batch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Split text into semantic chunks
                chunks = chunker.split_text(text)
                
                # Create embeddings for each chunk
                chunk_embeddings = model.encode(chunks)
                
                # Create points for each chunk
                for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    points.append(models.PointStruct(
                        id=len(points),
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk,
                            "file_path": file_path,
                            "chunk_index": idx,
                            "total_chunks": len(chunks)
                        }
                    ))
                
                total_chunks += len(chunks)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        # Upload batch to Qdrant
        if points:
            qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
    
    print(f"\nProcessing complete!")
    print(f"Total files processed: {len(file_paths)}")
    print(f"Total chunks created: {total_chunks}")

if __name__ == "__main__":
    # Example usage
    encode_text_files(
        directory_path="datasets/mimic_ex",
        collection_name="mimic_ex_v3",
        chunk_size=1000,
        batch_size=100,
        recreate_collection=True
    ) 