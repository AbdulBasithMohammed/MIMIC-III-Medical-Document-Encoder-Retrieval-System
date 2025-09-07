# MIMIC-III Medical Document Encoder Retrieval System

A comprehensive medical document retrieval system that processes MIMIC-III patient records using multiple encoding strategies and hybrid retrieval methods.

## Overview

This project implements a sophisticated medical document retrieval system that processes 44,915 MIMIC-III patient records using four different text encoding strategies and three retrieval methods. The system combines vector similarity search, keyword matching, and LLM-based reranking to provide accurate medical document retrieval.

## Features

### Document Encoding Methods
- **Document-level encoding**: Full document processing with sentence transformer embeddings
- **Recursive character encoding**: Intelligent text chunking with semantic similarity grouping
- **Semantic encoding**: Advanced semantic chunking using LangChain's SemanticChunker
- **Entity-based encoding**: Section-aware chunking based on medical document structure

### Retrieval Methods
- **Cosine Similarity Search**: Vector-based retrieval using 384-dimensional embeddings
- **Keyword Matching**: Medical term-weighted keyword search with stop word filtering
- **LLM Reranking**: Google Gemini-powered relevance scoring and result refinement

### Technical Stack
- **Vector Database**: Qdrant for high-performance vector storage and retrieval
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2) for 384-dimensional vectors
- **LLM Integration**: Google Gemini for intelligent reranking
- **Evaluation**: BLEURT scoring for retrieval quality assessment

## Project Structure

```
├── 1_document_encoding.py          # Document-level encoding implementation
├── 2_recursive_encoding.py         # Recursive character chunking
├── 3_semantic_encoding.py          # Semantic chunking with LangChain
├── 4_entity-based_encoding.py      # Section-based medical document chunking
├── cosine_retrieval.py             # Vector similarity search
├── keyword_retrieval.py            # Keyword matching with medical term weighting
├── reranking_retrieval.py          # LLM-based reranking system
├── evaluate_retrievers.py          # Comprehensive evaluation framework
├── plot_results.py                 # Results visualization and analysis
├── extract_sections.py             # Medical section header extraction
└── requirements.txt                # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AbdulBasithMohammed/MIMIC-III-Medical-Document-Encoder-Retrieval-System.git
cd MIMIC-III-Medical-Document-Encoder-Retrieval-System
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Qdrant vector database:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

5. Set up environment variables:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Usage

### Document Encoding

Process MIMIC-III documents using different encoding strategies:

```bash
# Document-level encoding
python 1_document_encoding.py

# Recursive character encoding
python 2_recursive_encoding.py

# Semantic encoding
python 3_semantic_encoding.py

# Entity-based encoding
python 4_entity-based_encoding.py
```

### Document Retrieval

```bash
# Cosine similarity search
python cosine_retrieval.py

# Keyword matching
python keyword_retrieval.py

# LLM reranking
python reranking_retrieval.py
```

### Evaluation

Run comprehensive evaluation across all methods:

```bash
python evaluate_retrievers.py
```

Generate performance visualizations:

```bash
python plot_results.py
```

## Performance Results

The system achieves the following performance metrics:

- **BLEURT Scores**: 0.16-0.21 across different retrieval methods
- **Cosine Similarity Scores**: 0.20-0.78 range for relevant document retrieval
- **Processing Scale**: 44,915 MIMIC-III patient records
- **Encoding Methods**: 4 different text chunking strategies
- **Retrieval Methods**: 3-tier hybrid retrieval system

## Dataset

This project uses the MIMIC-III (Medical Information Mart for Intensive Care III) dataset, which contains de-identified health data from over 40,000 patients. The system processes patient discharge summaries and clinical notes for medical document retrieval.

**Note**: Access to MIMIC-III requires completion of the required training and data use agreement through PhysioNet.

## Evaluation Queries

The system is evaluated using 10 medical queries covering various clinical scenarios:

1. "When is the usual visit to the vascular surgery clinic after discharge from the hospital?"
2. "What are the common complications after heart surgery?"
3. "How long does it take to recover from a knee replacement surgery?"
4. "What medications are typically prescribed after a stroke?"
5. "What are the dietary restrictions after gallbladder surgery?"
6. "What are the warning signs of a potential blood clot after surgery?"
7. "How should patients manage pain after spinal fusion surgery?"
8. "What are the typical rehabilitation exercises after rotator cuff repair?"
9. "What are the signs of infection at a surgical incision site?"
10. "How soon can patients resume normal activities after hip replacement surgery?"

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-III dataset provided by PhysioNet
- SentenceTransformers library for embedding generation
- Qdrant for vector database functionality
- Google Gemini for LLM-based reranking
- BLEURT for evaluation metrics

## Contact

AbdulBasith Mohammed - [GitHub](https://github.com/AbdulBasithMohammed)

Project Link: [https://github.com/AbdulBasithMohammed/MIMIC-III-Medical-Document-Encoder-Retrieval-System](https://github.com/AbdulBasithMohammed/MIMIC-III-Medical-Document-Encoder-Retrieval-System)
