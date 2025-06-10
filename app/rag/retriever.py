# app/rag/retriever.py
from langchain.tools.retriever import create_retriever_tool as langchain_create_retriever_tool
from .vector_store import vector_store

def create_retriever_tool():
    """Create and return a retriever tool for querying similar projects."""
    retriever = vector_store.as_retriever()
    retriever_tool = langchain_create_retriever_tool(
        retriever,
        "retrieve_similar_projects",
        "Retrieve similar project specs based on a given query"
    )
    return retriever_tool

retriever_tool = create_retriever_tool()
__all__ = ["retriever_tool"]
