# Chest X-Ray AI Agent Documentation

## Project Overview
This project implements an AI-powered system for analyzing chest X-rays using a combination of computer vision, natural language processing, and retrieval-augmented generation (RAG) techniques. The system provides detailed radiology reports with medical context and interpretation.

## Project Structure
```
.
├── Data/                  # Directory containing medical literature PDFs
├── chroma_db/            # Vector database for storing embeddings
├── ollama_models/        # Local Ollama models directory
├── r1_smolagent_rag.py   # Main agent implementation
├── ingest_pdfs.py        # PDF processing and vector store creation
├── requirements.txt      # Project dependencies
└── .env                  # Environment variables
```

## Core Components

### 1. ChestXRayAgent
The main agent class that handles:
- Integration with the Qwen 2.5 14B model for reasoning
- Vector database interactions for medical context retrieval
- Report generation with structured medical analysis

### 2. PDF Ingestion Pipeline
The system includes a robust pipeline for processing medical literature:
- PDF document loading and chunking
- Text splitting with overlap for context preservation
- Vector embeddings generation
- ChromaDB vector store creation and persistence

## Technical Stack

### Core Dependencies
- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for similarity search
- **HuggingFace**: For embeddings and transformers
- **SmolAgents**: For agent-based reasoning
- **Ollama**: Local LLM deployment
- **Gradio**: For web interface (if implemented)

### Key Models
- **Reasoning Model**: Qwen 2.5 14B Instruct (8K context)
- **Embeddings Model**: sentence-transformers/all-mpnet-base-v2

## Pipeline Flow

1. **Data Ingestion**
   - PDF documents are loaded from the Data directory
   - Documents are split into chunks with overlap
   - Text chunks are embedded and stored in ChromaDB

2. **Analysis Process**
   - Input chest X-ray is processed
   - Grad-CAM activation maps are generated
   - Diagnosis and confidence scores are computed

3. **Report Generation**
   - Medical context is retrieved from vector database
   - Activation patterns are analyzed
   - Comprehensive report is generated using LLM
   - Report is formatted with metadata and timestamps

## Features
- Detailed radiology report generation
- Medical literature context integration
- Grad-CAM activation map analysis
- Confidence scoring
- Professional medical terminology
- Structured report format
- Vector-based similarity search

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env`
5. Run PDF ingestion:
   ```bash
   python ingest_pdfs.py
   ```

## Usage
The system can be used to analyze chest X-rays and generate detailed reports:

```python
agent = ChestXRayAgent()
report = agent.generate_report(
    diagnosis="Pneumonia",
    activations={"upper_left": 0.85, "lower_right": 0.72},
    confidence=0.92
)
```

## Future Improvements
- Integration with additional medical imaging modalities
- Enhanced visualization capabilities
- Real-time analysis features
- Integration with medical record systems
- Additional medical literature sources
- Improved confidence scoring mechanisms

## Notes
- The system requires local Ollama deployment for the Qwen model
- Medical literature should be regularly updated in the vector database
- Reports should be reviewed by medical professionals
- The system is designed as an assistive tool, not a replacement for medical expertise 