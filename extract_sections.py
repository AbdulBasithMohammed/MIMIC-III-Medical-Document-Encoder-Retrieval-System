import os
import re
import google.generativeai as genai
from typing import List, Set
from dotenv import load_dotenv
import time
from google.api_core.exceptions import ResourceExhausted

# Load environment variables
load_dotenv()

# Set up Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

DATASET_DIR = "datasets/mimic_ex"
OUTPUT_FILE = "unique_section_headers.txt"
PROCESSED_FILES = "processed_files.txt"

def get_processed_files() -> Set[str]:
    """Get the list of already processed files"""
    if not os.path.exists(PROCESSED_FILES):
        return set()
    
    with open(PROCESSED_FILES, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def save_processed_file(filename: str):
    """Save a processed file to the tracking file"""
    with open(PROCESSED_FILES, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

def get_existing_headers() -> Set[str]:
    """Get the set of headers already collected"""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

# Regex to capture patterns like ". physical exam: "
section_pattern = re.compile(r'\.\s*([a-z \-/]+):\s', re.IGNORECASE)

# First, let's verify the directory exists and contains files
if not os.path.exists(DATASET_DIR):
    print(f"Error: Directory {DATASET_DIR} does not exist")
    exit(1)

files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.txt')]
if not files:
    print(f"Error: No .txt files found in {DATASET_DIR}")
    exit(1)

# Get already processed files and existing headers
processed_files = get_processed_files()
existing_headers = get_existing_headers()

# Filter out already processed files
files_to_process = [f for f in files if f not in processed_files]
print(f"Found {len(files)} total files, {len(files_to_process)} remaining to process")

# Extract section headers and store in list
all_sections = existing_headers.copy()
for filename in files_to_process[:1000]:  # Process next 1000 files
    file_path = os.path.join(DATASET_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            # Find all matches in the text
            matches = section_pattern.finditer(text)
            for match in matches:
                section = match.group(1).strip()
                if len(section) <= 20:
                    all_sections.add(section)
        # Mark file as processed
        save_processed_file(filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"Found {len(all_sections)} unique potential section headers (including existing)")

# Convert to sorted list of (index, section) tuples
sorted_sections = sorted(all_sections)
numbered_sections = [(i + 1, section) for i, section in enumerate(sorted_sections)]

# Create function to query Gemini for validation
def validate_with_gemini(headers: List[tuple]) -> Set[str]:
    # Dataset information to provide context to Gemini
    dataset_info = (
        "This dataset is the MIMIC-EX (Medical Information Mart for Intensive Care) "
        "which contains de-identified health-related data for patients admitted to critical care units. "
        "The data is stored in text files where each section contains medical details about the patient's condition, "
        "diagnosis, and treatment. These section headers typically describe aspects like physical exams, diagnosis, "
        "lab results, medications, and discharge plans. The goal is to identify and keep only the valid section headers, "
        "ignoring irrelevant text that may not correspond to actual sections."
    )
    
    # Format the context with dataset information and section headers
    context = "\n".join([f"Potential Section: '{header[1]}'" for header in headers])
    prompt = f"""
    Dataset Information:
    {dataset_info}
    
    Given the following potential section headers, please identify which ones are actual section headers and which are not.
    A valid section header is typically a medical documentation section like "Physical Exam", "History of Present Illness", etc.
    
    {context}
    
    Please respond with only the valid section headers, one per line, without any additional text or explanation.
    """
    
    try:
        # Generate response using the configured model
        response = gemini_model.generate_content(prompt)
        
        # Process response and extract valid headers
        valid_sections = set()
        for line in response.text.split('\n'):
            section = line.strip()
            if section:
                valid_sections.add(section)
        
        return valid_sections
    except ResourceExhausted as e:
        print(f"\nRate limit reached. Saving progress...")
        raise e

# Create a function to process headers in chunks
def process_headers_in_chunks() -> Set[str]:
    chunk_size = 100  # Process 100 headers at a time
    valid_sections = set()
    
    # Process in chunks
    for i in range(0, len(numbered_sections), chunk_size):
        chunk = numbered_sections[i:i+chunk_size]
        try:
            filtered_headers = validate_with_gemini(chunk)
            valid_sections.update(filtered_headers)
            print(f"Processed chunk {i//chunk_size + 1}, found {len(filtered_headers)} valid headers")
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(1)
            
        except ResourceExhausted:
            print(f"\nRate limit reached after processing {len(valid_sections)} headers")
            return valid_sections
    
    return valid_sections

def read_and_display_headers(file_path: str):
    """Read and display the count of section headers from the output file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        headers = [line.strip() for line in f if line.strip()]
        print(f"\nTotal valid section headers found: {len(headers)}")

try:
    # Call the chunk processing
    valid_headers = process_headers_in_chunks()
    
    # Get existing headers and combine with new ones
    existing_headers = get_existing_headers()
    all_valid_headers = sorted(existing_headers.union(valid_headers))
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        for section in all_valid_headers:
            output_file.write(f"{section}\n")
    
    print(f"\n--- Valid Section Headers Saved to {OUTPUT_FILE} ---")
    print(f"Previous headers: {len(existing_headers)}")
    print(f"New headers: {len(valid_headers - existing_headers)}")
    print(f"Total headers: {len(all_valid_headers)}")
    
except ResourceExhausted:
    print("\nProcessing stopped due to rate limit. Check the output file for progress.")
    if os.path.exists(OUTPUT_FILE):
        read_and_display_headers(OUTPUT_FILE)
