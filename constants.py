# optimized constants for speed
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
EMBEDDING_DIMENSIONS = 384
RERANKER_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
LLM_MODEL_NAME = "gemini-1.5-flash"

# aggressive chunking for speed
MAX_CHUNK_TOKENS = 500  # smaller chunks = faster processing
CHUNK_OVERLAP_TOKENS = 100  # 20% overlap
BATCH_SIZE = 80  # larger batches = fewer api calls

# retrieval optimized for speed
VECTOR_RETRIEVAL_K = 6  # fewer docs = faster
BM25_RETRIEVAL_K = 6
MAX_RERANK_CANDIDATES = 10
MAX_CONTEXT_CHUNKS = 5  # less context = faster llm calls
ENSEMBLE_WEIGHTS = [0.8, 0.2]  # favor vector search for speed

# performance settings
ENCODING_BATCH_SIZE = 128  # larger batches
CONCURRENT_QUESTION_LIMIT = 8  # more concurrency
API_RATE_LIMIT_DELAY = 0.1  # faster rate limiting

# pinecone settings
PINECONE_INDEX_NAME = "hackrx-e5-small"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"
INDEX_CREATION_WAIT_TIME = 10  # less waiting

# llm settings for speed
GEMINI_MAX_OUTPUT_TOKENS = 350  # fewer tokens = faster
GEMINI_TEMPERATURE = 0.05  # more deterministic

# file processing
HEADING_FONT_SIZE_THRESHOLD = 14
HEADING_LENGTH_THRESHOLD = 150  # shorter threshold

# error messages
ERROR_MESSAGES = {
    "unsupported_file": "unsupported file type",
    "extraction_failed": "text extraction failed",
    "no_text_extracted": "no text found",
    "download_failed": "download failed",
    "processing_failed": "processing failed",
    "gemini_not_configured": "llm not configured",
}

SUPPORTED_FILE_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "message/rfc822": "email",
    "text/plain": "email",
}

CLAUSE_PATTERN = r"\b(?:clause|section|article)\s+(\d+(?:\.\d+)*)"
