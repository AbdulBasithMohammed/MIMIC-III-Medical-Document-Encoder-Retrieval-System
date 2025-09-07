import os
import glob
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import numpy as np

def recursive_chunk_text(
    text: str,
    model: SentenceTransformer,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    similarity_threshold: float = 0.85
) -> List[str]:
    """
    Split text into chunks using recursive character splitting and similarity grouping.
    
    Args:
        text: The text to split
        model: SentenceTransformer model for embeddings
        chunk_size: Target size for chunks
        chunk_overlap: Overlap between chunks
        similarity_threshold: Threshold for semantic similarity
    
    Returns:
        List of chunks
    """
    # First split into sentences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    sentences = text_splitter.split_text(text)
    
    # Create embeddings for sentences
    sentence_embeddings = model.encode(sentences)
    
    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_chunk_embedding = None
    
    for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
        if not current_chunk:
            # Start new chunk
            current_chunk.append(sentence)
            current_chunk_embedding = embedding
        else:
            # Calculate similarity with current chunk
            similarity = np.dot(current_chunk_embedding, embedding) / (
                np.linalg.norm(current_chunk_embedding) * np.linalg.norm(embedding)
            )
            
            if similarity >= similarity_threshold:
                # Add to current chunk
                current_chunk.append(sentence)
                # Update chunk embedding (weighted average)
                current_chunk_embedding = (current_chunk_embedding * len(current_chunk) + embedding) / (len(current_chunk) + 1)
            else:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_embedding = embedding
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def encode_text_files(
    directory_path: str,
    collection_name: str = "mimic_ex_v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 100,
    file_pattern: str = "*.txt",
    similarity_threshold: float = 0.85,
    recreate_collection: bool = False
) -> None:
    """
    Process MIMIC-III patient history text files with recursive chunking and store in Qdrant.
    
    Args:
        directory_path: Path to the directory containing text files
        collection_name: Name of the Qdrant collection
        chunk_size: Target size for chunks
        chunk_overlap: Overlap between chunks
        batch_size: Number of files to process in each batch
        file_pattern: Pattern to match text files
        similarity_threshold: Threshold for semantic similarity
        recreate_collection: Whether to recreate the collection if it exists
    """
    # Initialize clients
    model = SentenceTransformer("all-MiniLM-L6-v2")
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
                
                # Split text into chunks
                chunks = recursive_chunk_text(
                    text,
                    model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    similarity_threshold=similarity_threshold
                )
                
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
        collection_name="mimic_ex_v2",
        chunk_size=1000,
        chunk_overlap=200,
        batch_size=100,
        similarity_threshold=0.85,
        recreate_collection=True
    ) 