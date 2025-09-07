import os
import glob
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# Load sentence transformer model for embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Qdrant client initialization
qdrant = QdrantClient("localhost", port=6333)

def read_section_headers(file_path):
    headers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                headers.append(line)
    return headers

def create_chunks(text, section_headers):
    chunks = []
    
    # First, find all section headers and their positions
    header_positions = []
    for header in section_headers:
        pattern = r"(?<=\.\s){}(?::)".format(re.escape(header))
        for match in re.finditer(pattern, text):
            header_positions.append((match.start(), header))
    
    # Sort headers by their position in the text
    header_positions.sort(key=lambda x: x[0])
    
    # Add the start of the text as a chunk if there's text before the first header
    if header_positions and header_positions[0][0] > 0:
        first_chunk = text[:header_positions[0][0]].strip()
        if first_chunk:
            chunks.append(first_chunk)
    
    # Process text between headers
    for i in range(len(header_positions)):
        start_pos, header = header_positions[i]
        
        # Find the end position (either next header or end of text)
        end_pos = header_positions[i+1][0] if i+1 < len(header_positions) else len(text)
        
        # Extract the chunk including the header
        chunk = text[start_pos:end_pos].strip()
        if chunk and chunk not in chunks:  # Only add if not already in chunks
            chunks.append(chunk)
    
    # Add any remaining text after the last header
    if header_positions:
        last_pos = header_positions[-1][0]
        remaining_text = text[last_pos:].strip()
        if remaining_text and remaining_text not in chunks:  # Only add if not already in chunks
            chunks.append(remaining_text)
    else:
        # If no headers found, use the entire text as one chunk
        chunks.append(text.strip())
    
    return chunks

def process_files(
    directory_path: str,
    headers_file_path: str,
    collection_name: str = "mimic_ex_v4",
    batch_size: int = 100,
    file_pattern: str = "*.txt",
    recreate_collection: bool = False
) -> None:
    """
    Process multiple text files with section-based chunking and store in Qdrant.
    
    Args:
        directory_path: Path to the directory containing text files
        headers_file_path: Path to the file containing section headers
        collection_name: Name of the Qdrant collection
        batch_size: Number of files to process in each batch
        file_pattern: Pattern to match text files
        recreate_collection: Whether to recreate the collection if it exists
    """
    # Read section headers once
    section_headers = read_section_headers(headers_file_path)
    print(f"Loaded {len(section_headers)} section headers")

    # Handle collection creation/recreation
    try:
        if recreate_collection:
            try:
                qdrant.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                print(f"Error deleting collection: {e}")
        
        # Create new collection
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=sentence_model.get_sentence_embedding_dimension(),
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
                # Read and process the file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read().lower()
                
                # Create chunks based on section headers
                chunks = create_chunks(text, section_headers)
                
                # Create embeddings for each chunk
                for idx, chunk in enumerate(chunks):
                    try:
                        embedding = sentence_model.encode(chunk)
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
                    except Exception as e:
                        print(f"Error processing chunk {idx} in file {file_path}: {e}")
                        continue
                
                total_chunks += len(chunks)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        # Upload batch to Qdrant
        if points:
            try:
                qdrant.upsert(
                    collection_name=collection_name,
                    points=points
                )
            except Exception as e:
                print(f"Error uploading batch to Qdrant: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Total files processed: {len(file_paths)}")
    print(f"Total chunks created: {total_chunks}")

if __name__ == "__main__":
    process_files(
        directory_path="datasets/mimic_ex",
        headers_file_path="unique_section_headers.txt",
        collection_name="mimic_ex_v4",
        batch_size=100,
        file_pattern="*.txt",
        recreate_collection=True
    )
