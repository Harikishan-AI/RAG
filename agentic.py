import os, zipfile, tempfile, json
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_groq import ChatGroq

# Vector DB & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Document loaders and text splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap

# Tavily Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph
from langgraph.graph import StateGraph, START, END


# ---------- STATE ----------
class State(TypedDict):
    input: str
    messages: Annotated[Sequence[HumanMessage | AIMessage], "conversation history"]
    search_results: str
    rag_results: str
    thinking_trace: list[str]
    actions_completed: list[str]


# ---------- TOOLS ----------
@tool
def tavily_search_tool(query: str):
    """Search the internet using Tavily Search API for real-time results."""
    tavily = TavilySearchResults(max_results=3)
    return tavily.invoke({"query": query})


# ---------- LLM ----------
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)


# ---------- VECTOR DATABASE ----------
def vector_database(input_path: str, persist_dir: str = "./chroma_store"):
    temp_dir = None
    docs = []

    if input_path.lower().endswith(".zip"):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        input_path = temp_dir

    pdf_files = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))
    elif input_path.lower().endswith(".pdf"):
        pdf_files.append(input_path)

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
        combined = bm25_docs[:top_k // 2] + dense_docs[:top_k // 2]
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
    """Router node to decide flow: RAG, Tavily, or both."""
    query = state["input"].lower()
    actions_completed = state.get("actions_completed", [])
    thinking_trace = state.get("thinking_trace", [])
    rag_results = state.get("rag_results", "")

    # Check query intent
    rag_intent = any(k in query for k in ["document", "pdf", "file", "report", "from the data", "rag"])
    tavily_intent = any(k in query for k in ["current", "today", "latest", "news", "from web", "internet", "real-time"])

    # Decision logic
    if not actions_completed:
        if rag_intent and tavily_intent:
            decision = "rag_and_search"
            thinking = "User query requires both document (RAG) and web (Tavily) information."
        elif rag_intent:
            decision = "rag"
            thinking = "User query seems document-based. Starting with RAG retrieval."
        elif tavily_intent:
            decision = "search_tool"
            thinking = "User query requires real-time information. Using Tavily search."
        else:
            decision = "rag"
            thinking = "No specific type detected — defaulting to RAG."
    elif "rag" in actions_completed and "search" not in actions_completed:
        # Check if RAG result is weak → then search
        if not rag_results or len(rag_results.strip()) < 50 or "not available" in rag_results.lower():
            decision = "search_tool"
            thinking = "RAG output was weak — switching to Tavily search."
        else:
            decision = "final"
            thinking = "RAG provided sufficient info — skipping search."
    else:
        decision = "final"
        thinking = "Both actions done — moving to final synthesis."

    thinking_trace.append(thinking)
    print(f"\n🧭 Supervisor Decision: {decision}")
    print(f"💭 Thinking: {thinking}")

    return {
        **state,
        "thinking_trace": thinking_trace,
        "last_action": decision
    }


def rag_node(state: State) -> State:
    query = state["input"]
    print("\n📚 Executing RAG Retrieval...")
    answer = rag_chain.invoke(query)
    actions_completed = state.get("actions_completed", [])
    actions_completed.append("rag")
    return {**state, "rag_results": answer, "actions_completed": actions_completed}


def search_tool_node(state: State) -> State:
    query = state["input"]
    print("\n🌐 Executing Tavily Search...")
    result = tavily_search_tool.run(query)
    actions_completed = state.get("actions_completed", [])
    actions_completed.append("search")
    return {**state, "search_results": str(result), "actions_completed": actions_completed}


def rag_and_search_node(state: State) -> State:
    """Runs both RAG and Tavily in parallel (for combined questions)."""
    query = state["input"]
    print("\n🔗 Executing Combined RAG + Tavily...")
    rag_res = rag_chain.invoke(query)
    tavily_res = tavily_search_tool.run(query)
    actions_completed = state.get("actions_completed", [])
    actions_completed += ["rag", "search"]
    return {
        **state,
        "rag_results": rag_res,
        "search_results": str(tavily_res),
        "actions_completed": actions_completed
    }


def final_node(state: State) -> State:
    query = state["input"]
    thinking = "\n".join(state.get("thinking_trace", []))
    rag_results = state.get("rag_results", "")
    search_results = state.get("search_results", "")

    synthesis_prompt = f"""
Provide a concise and informative answer to the user's question using both available sources.

User Question: {query}

RAG Results (document-based):
{rag_results if rag_results else "No RAG data available."}

Tavily Results (web-based):
{search_results if search_results else "No Tavily data available."}
"""

    print("\n✨ Generating Final Answer...")
    final_answer = llm.invoke(synthesis_prompt).content.strip()

    output = f"""
{'='*60}
🧠 THINKING PROCESS
{'='*60}
{thinking}

{'='*60}
💬 FINAL ANSWER
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
    }
)

workflow.add_edge("rag", "supervisor")
workflow.add_edge("search_tool", "supervisor")
workflow.add_edge("rag_and_search", "supervisor")
workflow.add_edge("final", END)

graph = workflow.compile()


# ---------- DEMO ----------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 MULTI-AGENT SYSTEM (RAG + Tavily + Combined Routing)")
    print("="*60)

    result = graph.invoke({
        "input": "Summarize the climate report from my documents and include the latest temperature trends from the web.",
        "messages": [],
        "search_results": "",
        "rag_results": "",
        "thinking_trace": [],
        "actions_completed": []
    })

    print(result["messages"][-1].content)
