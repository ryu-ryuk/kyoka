<div align="center">
  <img width="831" height="302" alt="Image" src="https://github.com/user-attachments/assets/c42a7c98-7422-439e-9b81-7e8732a5b470" />
</div>

<h1 align="center">Kyoka</h1>

<p align="center"><i>
the name <b>kyoka</b> (狂歌, “mad poem”) reflects the system’s role in turning formal documents into intelligent, structured responses — just as <b>kyōka</b> poetry reimagines tradition with clarity and wit.
</i></p>

---

**Kyoka** is a high-performance RAG system that processes documents and answers complex queries with precise, source-cited responses. Built for insurance, legal, HR, and compliance domains.


## Features

- **Multi-format Support**: PDF, DOCX, and email document processing
- **Hybrid Retrieval**: Combines dense (Pinecone) and sparse (BM25) search for superior accuracy
- **Semantic Chunking**: Intelligent text splitting with contextual overlap
- **Re-ranking**: Uses Flashrank for refined result relevance
- **Structured Responses**: JSON output with answers, rationale, and source attribution
- **Async Processing**: Concurrent question handling for optimal performance
- **GPU Acceleration**: CUDA support for faster embedding generation
- **Intelligent Caching**: Document and retriever caching for repeated queries

## Architecture

The system operates in two phases:

1. **Document Ingestion**: Structured parsing → Semantic chunking → Vector embedding → Pinecone indexing
2. **Query Processing**: Hybrid retrieval → Re-ranking → LLM generation → Structured response

## Tech Stack

- **Backend**: FastAPI
- **Vector Database**: Pinecone
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: HuggingFace (intfloat/e5-small-v2)
- **Re-ranker**: Flashrank (ms-marco-TinyBERT-L-2-v2)
- **Document Processing**: PyMuPDF, python-docx
- **Framework**: LangChain

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   
   Create a `.env` file:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

## API Endpoints

### Process Document from URL
**POST** `/hackrx/run`

```json
{
  "documents": "https://neow.com/document.pdf",
  "questions": ["What is the grace period?", "What are the waiting periods?"]
}
```

### Upload Document File
**POST** `/hackrx/upload`

```bash
curl -X POST "http://localhost:8000/hackrx/upload" \
     -F "file=@document.pdf" \
     -F "questions=What is the grace period?" \
     -F "questions=What are the waiting periods?"
```

### Response Format
```json
{
  "answers": [
    {
      "query": "What is the grace period?",
      "answer": "A 15-day grace period is provided for installment premium payments...",
      "rationale": "Based on policy document analysis of premium payment terms",
      "source": "page 31, page 3"
    }
  ]
}
```

## Performance

- **Response Time**: ~8.5 seconds for 2 questions (with caching: ~4.5 seconds)
- **Supported Formats**: PDF, DOCX, email (.eml, .msg, .txt)
- **Concurrent Processing**: Up to 8 parallel questions
- **GPU Acceleration**: 3-5x faster embedding generation

## Configuration

Key parameters can be adjusted in `constants.py`:

```python
MAX_CHUNK_TOKENS = 500          # Chunk size for processing speed
MAX_CONTEXT_CHUNKS = 3          # Context chunks sent to LLM
CONCURRENT_QUESTION_LIMIT = 8   # Parallel question processing
GEMINI_MAX_OUTPUT_TOKENS = 300  # Response length limit
```

## Evaluation Criteria Compliance

- **Accuracy**: Hybrid retrieval + re-ranking ensures precise context matching
- **Token Efficiency**: Optimized chunking and context selection minimizes LLM usage
- **Latency**: Async processing and caching deliver sub-10-second responses
- **Reusability**: Modular design with configurable components
- **Explainability**: Structured responses with rationale and source traceability

## Health Check

```bash
curl http://localhost:8000/health
```

Returns system status, GPU availability, cache statistics, and configuration details.

## License

[License](LICENSE)
