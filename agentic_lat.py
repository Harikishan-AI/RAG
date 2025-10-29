import os, zipfile, tempfile, json
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END


# ---------- STATE ----------
class State(TypedDict):
    input: str
    messages: Annotated[Sequence[HumanMessage | AIMessage], "conversation history"]
    search_results: str
    rag_results: str
    thinking_trace: list[str]
    actions_completed: list[str]


# ---------- TAVILY SEARCH TOOL ----------
def tavily_search_tool(query: str) -> str:
    """Performs real-time web search using Tavily API."""
    try:
        tavily = TavilySearchResults(max_results=3)
        results = tavily.invoke({"query": query})
        formatted = "\n".join(
            [f"- {r.get('content', '')}" for r in results if isinstance(r, dict)]
        )
        return formatted if formatted else "No relevant web data found."
    except Exception as e:
        return f"Tavily search failed: {str(e)}"


# ---------- LLM ----------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)


# ---------- VECTOR DATABASE (RAG PIPELINE) ----------
def vector_database(input_path: str, persist_dir: str = "./chroma_store"):
    temp_dir = None
    docs = []

    # Handle ZIP input
    if input_path.lower().endswith(".zip"):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(input_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        input_path = temp_dir

    # Gather PDFs
    pdf_files = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))
    elif input_path.lower().endswith(".pdf"):
        pdf_files.append(input_path)

    # Load and chunk documents
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    bm25_retriever = BM25Retriever.from_documents(chunks)

    def hybrid_retriever(query, top_k=5):
        dense_docs = vectorstore.similarity_search(query, k=top_k)
        bm25_docs = bm25_retriever.invoke(query)
        combined = bm25_docs[: top_k // 2] + dense_docs[: top_k // 2]
        return combined

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that answers using the provided context if relevant."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    rag_chain = (
        RunnableMap({
            "context": lambda q: "\n\n".join([d.page_content for d in hybrid_retriever(q)]),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


rag_chain = vector_database("./data2")


# ---------- NODES ----------
def supervisor(state: State) -> State:
    """Router node — decides between RAG, Tavily, or both."""
    query = state["input"].lower()
    actions_completed = state.get("actions_completed", [])
    thinking_trace = state.get("thinking_trace", [])
    rag_results = state.get("rag_results", "")

    rag_intent = any(k in query for k in ["document", "pdf", "file", "report", "from the data", "rag"])
    tavily_intent = any(k in query for k in ["current", "today", "latest", "news", "from web", "internet", "real-time"])

    if not actions_completed:
        if rag_intent and tavily_intent:
            decision = "rag_and_search"
            thinking = "User query requires both document (RAG) and web (Tavily) info."
        elif rag_intent:
            decision = "rag"
            thinking = "Document-based query detected — running RAG retrieval."
        elif tavily_intent:
            decision = "search_tool"
            thinking = "Real-time info requested — using Tavily search."
        else:
            decision = "rag_and_search"
            thinking = "No clear type — performing both RAG and Tavily for completeness."
    elif "rag" in actions_completed and "search" not in actions_completed:
        if not rag_results or len(rag_results.strip()) < 60:
            decision = "search_tool"
            thinking = "RAG answer insufficient — switching to Tavily."
        else:
            decision = "final"
            thinking = "RAG provided sufficient info — skipping search."
    else:
        decision = "final"
        thinking = "Both retrievals complete — proceeding to synthesis."

    thinking_trace.append(thinking)
    print(f"\n🧭 Supervisor Decision: {decision}")
    print(f"💭 Reasoning: {thinking}")

    return {**state, "thinking_trace": thinking_trace, "last_action": decision}


def rag_node(state: State) -> State:
    print("\n📘 Running RAG Retrieval...")
    query = state["input"]
    try:
        answer = rag_chain.invoke(query)
    except Exception as e:
        answer = f"RAG retrieval failed: {str(e)}"
    actions_completed = state.get("actions_completed", [])
    actions_completed.append("rag")
    return {**state, "rag_results": answer, "actions_completed": actions_completed}


def search_tool_node(state: State) -> State:
    print("\n🌐 Running Tavily Search...")
    query = state["input"]
    result = tavily_search_tool(query)
    actions_completed = state.get("actions_completed", [])
    actions_completed.append("search")
    return {**state, "search_results": result, "actions_completed": actions_completed}


def rag_and_search_node(state: State) -> State:
    """Execute both RAG and Tavily concurrently."""
    print("\n🔗 Running Combined RAG + Tavily Search...")
    query = state["input"]
    rag_res = rag_chain.invoke(query)
    tavily_res = tavily_search_tool(query)
    actions_completed = state.get("actions_completed", [])
    actions_completed += ["rag", "search"]
    return {
        **state,
        "rag_results": rag_res,
        "search_results": tavily_res,
        "actions_completed": actions_completed
    }


def final_node(state: State) -> State:
    print("\n✨ Synthesizing Final Answer...")
    query = state["input"]
    rag_results = state.get("rag_results", "")
    search_results = state.get("search_results", "")
    thinking = "\n".join(state.get("thinking_trace", []))

    synthesis_prompt = f"""
You are a reasoning agent. Combine both document-based and real-time insights to answer clearly and accurately.

User Question:
{query}

RAG (document-based findings):
{rag_results if rag_results else "No RAG data available."}

Tavily (web search findings):
{search_results if search_results else "No Tavily data available."}
"""

    final_answer = llm.invoke(synthesis_prompt).content.strip()

    output = f"""
{'='*60}
🧩 THINKING TRACE
{'='*60}
{thinking}

{'='*60}
💡 FINAL ANSWER
{'='*60}
{final_answer}
"""
    return {**state, "messages": [AIMessage(content=output)]}


# ---------- GRAPH ----------
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor)
workflow.add_node("rag", rag_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("rag_and_search", rag_and_search_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda s: s.get("last_action", "final"),
    {
        "rag": "rag",
        "search_tool": "search_tool",
        "rag_and_search": "rag_and_search",
        "final": "final"
    },
)

workflow.add_edge("rag", "supervisor")
workflow.add_edge("search_tool", "supervisor")
workflow.add_edge("rag_and_search", "supervisor")
workflow.add_edge("final", END)

graph = workflow.compile()


# ---------- DEMO ----------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 MULTI-AGENT SYSTEM — HYBRID RAG + TAVILY SEARCH ROUTER")
    print("="*70)

    result = graph.invoke({
        "input": "What percentage of global carbon dioxide emissions has the United States contributed since 1850? and what is the latest ai news?",
        "messages": [],
        "search_results": "",
        "rag_results": "",
        "thinking_trace": [],
        "actions_completed": []
    })

    print(result["messages"][-1].content)
