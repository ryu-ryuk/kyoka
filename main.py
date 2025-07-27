import aiohttp
import asyncio
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import tiktoken
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import logging
from typing import List, Dict, Any, Optional
import time
from flashrank import Ranker, RerankRequest
import torch
import os
import hashlib
import json
import google.generativeai as genai
from datetime import datetime
import re
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()

# logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRX Query-Retrieval System",
    description="Advanced LLM-powered document query system with file upload support",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set!")

# global cache for processed documents
DOCUMENT_CACHE = {}
RETRIEVER_CACHE = {}


# pydantic models
class QueryRequest(BaseModel):
    documents: str  # URL to PDF in Azure Blob Storage as shown in the problem-statement
    questions: List[str]


class FileQueryRequest(BaseModel):
    questions: List[str]


class Answer(BaseModel):
    query: str
    answer: str
    rationale: str
    source: str


class QueryResponse(BaseModel):
    answers: List[Answer]


class SimpleQueryResponse(BaseModel):
    answers: List[str]


# initialize components with optimization
def initialize_components():
    """init all components once at startup."""
    global \
        embeddings, \
        sentence_model, \
        tokenizer, \
        pinecone_client, \
        index, \
        ranker, \
        gemini_model

    # TODO: MIGHT NEED UPDATE; using e5-small-v2 for better speed while maintaining quality
    model_name = "intfloat/e5-small-v2"  # 384 dimensions, fasterrr
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },  # cuddaaaaaaa
        encode_kwargs={"batch_size": 64},
    )

    sentence_model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        sentence_model = sentence_model.cuda()

    tokenizer = tiktoken.get_encoding("cl100k_base")

    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
    index = initialize_pinecone_index(pinecone_client)

    # lightweight ahh reranker for fast re-ranking
    try:
        ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", max_length=512)
    except:
        logger.warning("Could not initialize ranker, skipping re-ranking")
        ranker = None

    # init Gemini model
    if GEMINI_API_KEY != "GEMINI_API_KEY":
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini model initialized")
    else:
        gemini_model = None
        logger.warning("Gemini model not initialized - invalid API key")

    logger.info(f"We are ready!!. GPU: {torch.cuda.is_available()}")


def initialize_pinecone_index(
    pinecone_client: PineconeClient, index_name: str = "hackrx-e5-small"
):
    """init Pinecone index for e5-small-v2 (384 dimensions)."""  # reducing from 1024 to try and make it faster
    try:
        existing_indexes = pinecone_client.list_indexes().names()

        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=384,  # e5-small-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(15)  # TODO : update?
            logger.info(f"Created Pinecone index: {index_name}")
        else:
            logger.info(f"Using existing Pinecone index: {index_name}")

        return pinecone_client.Index(index_name)

    except Exception as e:
        logger.error(f"Pinecone index initialization error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Pinecone index setup failed: {str(e)}"
        )


async def download_pdf_cached(url: str) -> bytes:
    """download PDF with caching based on URL hash."""
    url_hash = hashlib.md5(url.encode()).hexdigest()

    if url_hash in DOCUMENT_CACHE:
        logger.info("Using cached PDF")
        return DOCUMENT_CACHE[url_hash]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400, detail="Failed to download PDF"
                    )

                pdf_content = await response.read()
                DOCUMENT_CACHE[url_hash] = pdf_content
                logger.info("PDF downloaded and cached")
                return pdf_content
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


def extract_structured_text_from_pdf(pdf_content: bytes, doc_id: str) -> List[Document]:
    """
    PDF extraction using PyMuPDF with structured chunking.
    Detects headings, paragraphs, and maintains document structure.
    """
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        documents = []

        for page_num, page in enumerate(doc):
            # get text with structure information
            text_dict = page.get_text("dict")

            # extract structured content
            page_content = []
            current_section = ""

            for block in text_dict["blocks"]:
                if "lines" in block:
                    block_text = ""
                    font_sizes = []

                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            text = span["text"].strip()
                            font_size = span["size"]
                            font_sizes.append(font_size)
                            line_text += text + " "

                        if line_text.strip():
                            block_text += line_text.strip() + "\n"

                    if block_text.strip():
                        # determine if this is likely a heading based on font size ####
                        avg_font_size = (
                            sum(font_sizes) / len(font_sizes) if font_sizes else 12
                        )
                        is_heading = avg_font_size > 14 or len(block_text.strip()) < 100

                        if is_heading and len(block_text.strip()) < 200:
                            current_section = block_text.strip()
                            page_content.append(f"SECTION: {current_section}")
                        else:
                            page_content.append(block_text.strip())

            # join content and split into semantic chunks
            full_text = "\n".join(page_content)

            # use semantic chunking with overlap
            chunks = create_semantic_chunks(
                full_text,
                doc_id,
                page_num + 1,
                current_section,
                max_tokens=800,
                overlap_tokens=160,  # 20% overlap ###
            )

            documents.extend(chunks)

        doc.close()
        logger.info(f"Extracted {len(documents)} structured chunks from PDF")
        return documents

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


def create_semantic_chunks(
    text: str,
    doc_id: str,
    page_num: int,
    section: str,
    max_tokens: int = 800,
    overlap_tokens: int = 160,
) -> List[Document]:
    """
    create semantic chunks with token-based limits and overlap.
    """
    try:
        # use RecursiveCharacterTextSplitter for smart chunking ##
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            length_function=lambda x: len(tokenizer.encode(x)),
            separators=["\n\nSECTION:", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
        )

        chunks = splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                # extract clause IDs or important identifiers
                clause_matches = re.findall(
                    r"\b(?:clause|section|article)\s+(\d+(?:\.\d+)*)", chunk.lower()
                )
                clause_ids = list(set(clause_matches)) if clause_matches else []

                # TODO: TESTING ; create metadata for better filtering and explanation
                metadata = {
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_id": f"{page_num}_{i}",
                    "section": section,
                    "clause_ids": clause_ids,
                    "token_count": len(tokenizer.encode(chunk)),
                    "chunk_index": i,
                }

                documents.append(
                    Document(page_content=chunk.strip(), metadata=metadata)
                )

        return documents

    except Exception as e:
        logger.error(f"Chunking error: {str(e)}")
        return []


async def setup_hybrid_retriever_cached(documents: List[Document], doc_id: str):
    """
    setup hybrid retrieval system with Dense (Pinecone) + Sparse (BM25).
    """
    if doc_id in RETRIEVER_CACHE:
        logger.info("Using cached hybrid retriever")
        return RETRIEVER_CACHE[doc_id]

    try:
        # dense Retrieval: Pinecone Vector Store
        vectorstore = PineconeVectorStore(
            index=index, embedding=embeddings, text_key="text"
        )

        # add documents with metadata indexing
        batch_size = 40
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                # add metadata for better filtering
                vectorstore.add_documents(batch)
                await asyncio.sleep(0.3)  # rate limitinggg
            except Exception as batch_error:
                logger.warning(f"Batch {i // batch_size} failed: {batch_error}")
                continue

        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10, "include_metadata": True}
        )

        # sparse Retrieval: BM25 for domain-specific terms
        bm25_retriever = BM25Retriever.from_documents(documents, k=10)

        # ensemble retriever with optimized weights for insurance documents
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[
                0.75,
                0.25,
            ],  # favor dense retrieval for semantic understanding #####
        )

        RETRIEVER_CACHE[doc_id] = ensemble_retriever
        logger.info("Hybrid retriever setup completed and cached")
        return ensemble_retriever

    except Exception as e:
        logger.error(f"Hybrid retriever setup error: {str(e)}")
        # fallback to BM25 only
        try:
            logger.info("Falling back to BM25-only retrieval")
            bm25_retriever = BM25Retriever.from_documents(documents, k=12)
            RETRIEVER_CACHE[doc_id] = bm25_retriever
            return bm25_retriever
        except Exception as fallback_error:
            logger.error(f"Fallback retriever setup failed: {str(fallback_error)}")
            raise HTTPException(
                status_code=500, detail=f"All retriever setups failed: {str(e)}"
            )


async def get_relevant_context_with_reranking(
    query: str, retriever, max_chunks: int = 5
) -> List[Document]:
    """
    get relevant context with lightweight cross-encoder re-ranking.
    """
    try:
        # get initial chunks using invoke
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, retriever.invoke, query
        )

        # re-rank using lightweight cross-encoder if available
        if ranker and len(chunks) > max_chunks:
            try:
                passages = [
                    {"id": i, "text": chunk.page_content}
                    for i, chunk in enumerate(chunks[:15])
                ]
                request = RerankRequest(query=query, passages=passages)
                ranked = ranker.rerank(request)

                # get top chunks with original metadata
                reranked_chunks = []
                for r in ranked[:max_chunks]:
                    original_chunk = chunks[r["id"]]
                    reranked_chunks.append(original_chunk)

                return reranked_chunks

            except Exception as rerank_error:
                logger.warning(
                    f"Re-ranking failed, using original order: {rerank_error}"
                )

        # return top chunks without re-ranking
        return chunks[:max_chunks]

    except Exception as e:
        logger.error(f"Context retrieval error: {str(e)}")
        return []


async def generate_structured_answer(
    query: str, chunks: List[Document]
) -> Dict[str, str]:
    """
    generate structured answer with rationale and source using Gemini.
    """
    if not gemini_model or not chunks:
        return {
            "answer": "Uh oh! Unable to generate answer - check API configuration",
            "rationale": "Gemini API not properly configured or no relevant context found",
            "source": "N/A :( ",
        }

    # create rich context with metadata
    context_parts = []
    sources = set()

    for chunk in chunks:
        metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
        page = metadata.get("page", "N/A")
        section = metadata.get("section", "")
        clause_ids = metadata.get("clause_ids", [])

        sources.add(f"page {page}")

        context_info = f"[Page {page}"
        if section:
            context_info += f", Section: {section}"
        if clause_ids:
            context_info += f", Clauses: {', '.join(clause_ids)}"
        context_info += "]"

        context_parts.append(f"{context_info}\n{chunk.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    source_summary = ", ".join(sorted(sources))

    prompt = f"""You are an expert at analyzing insurance policy documents. Based on the provided context, answer the question with precision and provide clear reasoning.

Context from policy document:
{context}

Question: {query}

Instructions:
- Provide a direct, factual answer based ONLY on the information in the context
- Include specific details like time periods, amounts, conditions, percentages, and limits
- If coverage conditions exist, mention them clearly
- Explain your reasoning in the rationale
- Be precise about sources

Please respond in valid JSON format:
{{
    "answer": "Direct answer to the question",
    "rationale": "Explanation of reasoning based on the context",
    "source": "Specific page numbers and sections referenced"
}}"""

    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.1,
                ),
            ),
        )

        # parse JSON response
        try:
            result = json.loads(response.text.strip())
        except:
            # fallback parsing if JSON fails
            answer_text = response.text.strip()
            result = {
                "answer": answer_text[:200] + "..."
                if len(answer_text) > 200
                else answer_text,
                "rationale": "Based on policy document analysis",
                "source": source_summary,
            }

        # ensure all required fields are present
        if not all(key in result for key in ["answer", "rationale", "source"]):
            result = {
                "answer": result.get("answer", "Answer could not be parsed properly"),
                "rationale": result.get("rationale", "Reasoning not available"),
                "source": result.get("source", source_summary),
            }

        return result

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return {
            "answer": "Error generating answer with Gemini API",
            "rationale": f"API Error: {str(e)}",
            "source": source_summary,
        }


# ENDPOINTS @@


@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query_url(request: QueryRequest):
    """Process PDF from URL and answer queries with structured responses."""
    return await process_queries(request.documents, request.questions, is_url=True)


@app.post("/hackrx/upload", response_model=QueryResponse)
async def run_query_upload(questions: List[str] = [], file: UploadFile = File(...)):
    """Process uploaded PDF file and answer queries."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # read file content
    file_content = await file.read()

    # create a unique identifier for the uploaded file
    file_hash = hashlib.md5(file_content).hexdigest()
    file_identifier = f"upload_{file_hash}"

    return await process_queries(
        file_identifier, questions, is_url=False, file_content=file_content
    )


@app.post("/hackrx/run/simple", response_model=SimpleQueryResponse)
async def run_query_simple(request: QueryRequest):
    """Process PDF and return simple string answers (for hackathon testing / fallback)."""
    full_response = await process_queries(
        request.documents, request.questions, is_url=True
    )
    simple_answers = [answer.answer for answer in full_response.answers]
    return SimpleQueryResponse(answers=simple_answers)


async def process_queries(
    document_identifier: str,
    questions: List[str],
    is_url: bool = True,
    file_content: bytes = None,
) -> QueryResponse:
    """the CORE processing function for both URL and file uploads."""
    start_time = time.time()

    try:
        # get PDF content
        if is_url:
            pdf_content = await download_pdf_cached(document_identifier)
            doc_id = hashlib.md5(document_identifier.encode()).hexdigest()
        else:
            pdf_content = file_content
            doc_id = document_identifier

        # extract structured text
        documents = extract_structured_text_from_pdf(pdf_content, doc_id)

        if not documents:
            raise HTTPException(
                status_code=400, detail="No text could be extracted from PDF"
            )

        # setup hybrid retriever
        retriever = await setup_hybrid_retriever_cached(documents, doc_id)

        # process all questions concurrently
        async def process_single_question(question: str) -> Answer:
            try:
                relevant_chunks = await get_relevant_context_with_reranking(
                    question, retriever
                )
                result = await generate_structured_answer(question, relevant_chunks)

                return Answer(
                    query=question,
                    answer=result["answer"],
                    rationale=result["rationale"],
                    source=result["source"],
                )
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                return Answer(
                    query=question,
                    answer=f"Error processing question: {str(e)}",
                    rationale="Processing failed",
                    source="Error",
                )

        # execute with controlled concurrency
        semaphore = asyncio.Semaphore(4)

        async def bounded_process_question(question: str) -> Answer:
            async with semaphore:
                return await process_single_question(question)

        tasks = [bounded_process_question(q) for q in questions]
        answers = await asyncio.gather(*tasks)

        processing_time = time.time() - start_time
        logger.info(
            f"Total processing time: {processing_time:.2f} seconds for {len(questions)} questions"
        )

        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with sys info."""
    return {
        "message": "HackRX Advanced Query-Retrieval System",
        "status": "healthy",
        "features": [
            "Structured PDF parsing with PyMuPDF",
            "Semantic chunking with overlap",
            "Hybrid retrieval (Dense + Sparse)",
            "Lightweight re-ranking",
            "Metadata indexing",
            "File upload support",
            "Async processing",
            "GPU acceleration",
        ],
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "cached_documents": len(DOCUMENT_CACHE),
        "cached_retrievers": len(RETRIEVER_CACHE),
        "gemini_configured": gemini_model is not None,
        "ranker_available": ranker is not None,
        "embedding_model": "intfloat/e5-small-v2",
        "pinecone_configured": PINECONE_API_KEY != "YOUR_API_KEY_HERE",
    }


# initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        logger.info("Starting component initialization...")
        initialize_components()
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
