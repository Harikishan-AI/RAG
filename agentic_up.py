# pip install langchain langchain-community langchain-core langchain-google-genai langchain-huggingface sentence-transformers chromadb tiktoken

import os
import zipfile
import tempfile
from typing import List, Optional, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# loaders / splitters / embeddings / vectorstores
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# LLM (you can swap to other LLMs)
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Helpers: load PDFs (folder/zip/single)
# -------------------------------
def collect_pdf_paths(input_path: str) -> List[str]:
    """
    Accepts: folder path, single pdf path, or .zip containing pdfs.
    Returns list of absolute pdf file paths.
    """
    temp_dir = None
    if input_path.lower().endswith(".zip"):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(temp_dir)
        input_path = temp_dir

    pdfs = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, f))
    elif input_path.lower().endswith(".pdf"):
        pdfs.append(input_path)

    if not pdfs:
        raise ValueError(f"No PDF files found at {input_path}")

    return pdfs


def load_and_chunk_pdfs(input_path: str,
                        chunk_size: int = 500,
                        chunk_overlap: int = 50,
                        encoding_name: str = "cl100k_base"):
    """
    Loads PDFs, returns a list of LangChain Document objects (chunked).
    """
    pdf_paths = collect_pdf_paths(input_path)
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        loaded = loader.load()
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(pdf_paths)} PDFs -> {len(chunks)} chunks")
    return chunks


# -------------------------------
# Build vectorstore + BM25 retriever
# -------------------------------
def build_retrievers(chunks,
                     persist_dir: str = "./chroma_store",
                     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds:
     - dense vectorstore (Chroma) using HuggingFaceEmbeddings
     - BM25 retriever built from same chunks (sparse)
    Returns (vectorstore, bm25_retriever, embeddings)
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # NOTE: If you want HNSW/ANN tuning in Chroma, configure client_settings here.
    # Example (some Chroma builds or deployments accept index settings):
    # client_settings = {"chroma_db_impl": "duckdb+parquet", "index": "hnsw", "hnsw:space":"cosine"}
    # vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir, client_settings=client_settings)
    #
    # If your Chroma doesn't support client_settings/index param, use default Chroma from_documents:
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)

    # BM25 (sparse) - good for keyword precision and ranking
    bm25_retriever = BM25Retriever.from_documents(chunks)

    return vectorstore, bm25_retriever, embeddings


# -------------------------------
# Hybrid retriever: combine BM25 + dense results
# -------------------------------
def hybrid_retriever_factory(vectorstore,
                            bm25_retriever,
                            chunks,
                            top_k: int = 6):
    """
    Returns a function hybrid_retriever(query, metadata_filter=None, k=6)
    which returns combined docs from BM25 and dense vectorstore.
    """
    def hybrid_retriever(query: str, metadata_filter: Optional[Dict] = None, k: int = top_k):
        # 1) BM25 (sparse)
        try:
            bm25_docs = bm25_retriever.invoke(query)  # returns list of Document
        except Exception:
            bm25_docs = []

        # 2) Dense semantic search
        # If your Chroma (vectorstore) supports metadata filters pass `filter=metadata_filter`.
        try:
            if metadata_filter:
                dense_docs = vectorstore.similarity_search(query, k=k, filter=metadata_filter)
            else:
                dense_docs = vectorstore.similarity_search(query, k=k)
        except TypeError:
            # fallback if filter arg not supported
            dense_docs = vectorstore.similarity_search(query, k=k)

        # 3) Combine: favor BM25 for top keyword matches, then dense for semantic
        half = max(1, k // 2)
        combined = []
        # use unique contents to avoid duplicates
        seen = set()
        for d in (bm25_docs[:half] + dense_docs[:half]):
            key = (d.page_content[:200])  # coarse dedupe by content prefix
            if key not in seen:
                combined.append(d)
                seen.add(key)
        return combined

    return hybrid_retriever


# -------------------------------
# Better RAG system prompt
# -------------------------------
RAG_SYSTEM_PROMPT = """
You are a concise, factual assistant answering questions using only the provided context.
Grounding rules:
- Use only the provided context to answer. If the answer is not present, say "I don't know".
- Prefer concise answers (2–6 sentences) unless user asks for depth.
- When using content from documents, include a short citation marker like [doc#] referencing the provided context.
- If multiple documents disagree, state that and summarize the disagreement.
- When code is provided, keep examples minimal and runnable.

Output format:
1) Direct answer (1-3 paragraphs)
2) Optional bullet list with key points
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

# -------------------------------
# Build end-to-end RAG chain
# -------------------------------
def build_rag_pipeline(input_path: str,
                       persist_dir: str = "./chroma_store",
                       embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):

    chunks = load_and_chunk_pdfs(input_path, chunk_size=500, chunk_overlap=200, encoding_name="cl100k_base")
    vectorstore, bm25_retriever, embeddings = build_retrievers(chunks, persist_dir=persist_dir, embedding_model=embedding_model)
    hybrid_retriever = hybrid_retriever_factory(vectorstore, bm25_retriever, chunks, top_k=6)

    # LLM model (Gemini / Google - you can swap)
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')  # or another LLM

    # Map the hybrid_retriever to a callable that produces joined context text
    def context_fn(q_and_filter):
        # q_and_filter can be either a string (query) or tuple (query, metadata_filter)
        if isinstance(q_and_filter, tuple):
            q, metadata_filter = q_and_filter
        else:
            q, metadata_filter = q_and_filter, None

        docs = hybrid_retriever(q, metadata_filter=metadata_filter, k=6)
        # attach coarse citation labels [doc1], [doc2] for traceability
        pieces = []
        for i, d in enumerate(docs, start=1):
            # prefer metadata source name if present
            src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            label = f"[doc{i}" + (f":{src}]" if src else "]")
            pieces.append(f"{label}\n{d.page_content}")
        return "\n\n".join(pieces) if pieces else "No relevant context found."

    rag_chain = (
        RunnableMap({
            "context": context_fn,
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain, vectorstore, bm25_retriever


# -------------------------------
# Example usage (multiple PDFs / folder)
# -------------------------------
if __name__ == "__main__":
    DATA_PATH = "./data2"  # folder with many PDFs or a .zip of PDFs
    rag_chain, vectorstore, bm25 = build_rag_pipeline(DATA_PATH, persist_dir="./chroma_store")

    # Optionally filter by metadata, e.g. {"source": "world_bank_report.pdf"} if your documents set metadata when loading
    metadata_filter_example = None  # e.g. {"source": "world_bank_report.pdf"}

    query = "By how much had the United States warmed since 1970?"
    # We pass (query, filter) to context_fn via rag_chain.invoke
    # rag_chain expects a single input - we gave RunnableMap mapping, so pass either query or (query, filter)
    answer = rag_chain.invoke((query, metadata_filter_example))
    print("\n📘 Answer:\n", answer)
