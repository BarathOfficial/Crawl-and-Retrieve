import os
from typing import Annotated, Sequence, List
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from crawler import crawler
from chunking_embedding import split_text, create_embeddings
from vector_db import store_Web_chunks, retriver, get_chroma_client, get_or_create_collection, build_prompt

import logging


PRESISTDBPATH = os.getenv("CHROMADB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MODEL_NAME = os.getenv("MODEL_NAME")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatGroq(model_name=MODEL_NAME)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def dynamic_rag_node(state: State):
    try:
        last_message = state["messages"][-1]
        query = last_message.content

        # Initialize Models and DB Client
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        client = get_chroma_client(PRESISTDBPATH)
        collection = get_or_create_collection(client, "web_chunks")

        # Try retrieving from DB first
        results = retriver(query, embedding_model, collection)
        context_docs = []
        if results and "documents" in results and "distances" in results and results["documents"]:
            # Only use documents that are somewhat relevant (L2 distance < 1.2)
            distances = results["distances"][0]
            docs = results["documents"][0]
            context_docs = [doc for doc, dist in zip(docs, distances) if dist < 1.2]

        if context_docs:
            prompt = build_prompt(query, context_docs)
            response = llm.invoke(prompt)

            logging.info("Retrieval Successful from DB")

            return {"messages": [response]}
        
        # Fallback: search the web
        search_results = tavily_client.search(query, max_results=2)
        search_urls = [r["url"] for r in search_results.get("results", [])]

        # searching
        visited = set()
        data = []
        crawled_urls = []
        
        for search_url in search_urls:
            crawler(url=search_url, visited=visited, data=data, urls=crawled_urls, depth=0, max_depth=0)

        for i in range(len(data)):
            if not data[i]:
                continue
            # chunking
            splits = split_text(data[i], text_splitter)
            if splits:
                # embedding
                embeddings = create_embeddings(splits, embedding_model)
                if embeddings is not None:
                    # storing
                    store_Web_chunks(splits, crawled_urls[i], embeddings, collection)

        # retrival
        results = retriver(query, embedding_model, collection)
        context_docs = []
        if results and "documents" in results and results["documents"]:
            context_docs = results["documents"][0]

        
        prompt = build_prompt(query, context_docs)
        response = llm.invoke(prompt)

        logging.info("Retrival Successfull")

        return {"messages": [response]}
    except Exception as e:
        logging.error(f"Retrival failure: {e}")

    

def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def router(state: State):
    last_msg = state["messages"][-1].content.lower()
    # If the LLM says it doesn't know, trigger the RAG node
    failure_phrases = [
        "don't know", "not found", "not aware", "do not have", "don't have", 
        "not capable", "cannot access", "unable to", "my knowledge cutoff", 
        "as an ai"
    ]
    if any(phrase in last_msg for phrase in failure_phrases):
        return "dynamic_rag"
    return END


workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("dynamic_rag", dynamic_rag_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router)
workflow.add_edge("dynamic_rag", END)

app = workflow.compile(checkpointer=MemorySaver())
