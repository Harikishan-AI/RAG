# Install dependencies first (if not installed)
# pip install langchain langchain-community langchain-core langchain-google-genai langchain-huggingface sentence-transformers python-dotenv pymilvus

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------------
# 1️⃣ Load and chunk your document
# -------------------------------
pdf_path = "sample_ingestion.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# -------------------------------
# 2️⃣ Create embeddings
# -------------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# -------------------------------
# 3️⃣ Connect to Milvus and upsert vectors
# -------------------------------
# Use env vars for Milvus connection
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = os.getenv("MILVUS_PORT", "19530")
milvus_user = os.getenv("MILVUS_USER")  # optional for Milvus/Attu
milvus_password = os.getenv("MILVUS_PASSWORD")  # optional
collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")

vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_args={
        "host": milvus_host,
        "port": milvus_port,
        **({"user": milvus_user} if milvus_user else {}),
        **({"password": milvus_password} if milvus_password else {}),
    },
    collection_name=collection_name,
    index_params={"metric_type": "IP"},  # cosine via inner product
)

# -------------------------------
# 4️⃣ Create retriever
# -------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 5️⃣ Define Gemini LLM and Prompt
# -------------------------------
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

RAG_SYSTEM_PROMPT = """
You are a helpful assistant answering questions using only the provided context.

Grounding rules:
- Use the context verbatim; do not invent facts beyond it.
- If the answer isn't clearly in the context, say "I don't know".
- Prefer concise answers (2–6 sentences) unless the user asks for depth.
- Cite sources as [source] when available from document metadata.
- Keep code blocks minimal and runnable when relevant.

Output format:
- Direct answer first.
- Optional short bullet list for key points.
- Citations at the end like: [source1]; [source2]
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

# -------------------------------
# 6️⃣ Combine everything into LCEL pipeline
# -------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# -------------------------------
# 7️⃣ Run query
# -------------------------------
query = "What is the LLaMA model?"
answer = rag_chain.invoke(query)

print("\n📘 Answer:\n", answer)
