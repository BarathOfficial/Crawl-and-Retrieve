# Agentic Web Crawler & RAG Chatbot

An intelligent, autonomous AI chatbot built with **LangGraph** that dynamically decides when to search the web, crawl articles, and store the information locally in a vector database for future reference.

## Features

- **Agentic Routing:** The LLM evaluates your question. If it knows the answer, it responds instantly. If it doesn't (or its knowledge cutoff prevents it), it automatically routes the query to the dynamic RAG pipeline.
- **Automated Web Crawling:** Uses **Tavily** to find relevant URLs and **Trafilatura/BeautifulSoup** to scrape and extract the main content, ignoring ads and boilerplate.
- **Smart Chunking & Embedding:** Splits the extracted text into manageable chunks and generates local embeddings using Hugging Face's `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Persistent Vector Cache:** Stores all scraped knowledge locally in **ChromaDB**. Future questions on the same topic will pull instantly from the local database instead of crawling the web again!
- **Distance Thresholding:** Uses L2 distance filtering to ensure the database only provides context if it's actually highly relevant to the new query.

## Technology Stack

- **Orchestration:** LangGraph & LangChain
- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Search:** Tavily Search API
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Web Scraping:** Trafilatura & BeautifulSoup4
- **Package Manager:** `uv`

## Setup & Installation

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Website-Crawler
   ```

2. **Install dependencies:**
   *(Note: If you are running this inside a OneDrive synced folder on Windows, you must use the `--link-mode=copy` flag to avoid hardlink errors).*
   ```powershell
   uv sync --link-mode=copy
   ```

3. **Configure Environment Variables:**
   Ensure your `.env` file in the root directory contains your API keys:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   TAVILY_API_KEY="your_tavily_api_key_here"
   MODEL_NAME="llama-3.3-70b-versatile"
   CHROMADB_PATH="vector_db"
   EMBEDDING_MODEL="all-MiniLM-L6-v2"
   ```

## Usage

Run the chatbot loop directly through `uv`:

```powershell
uv run main.py
```

- Type your questions normally.
- If the bot doesn't know the answer, you'll see logs as it searches the web, downloads the pages, chunks the text, and saves it to your local `vector_db` folder.
- Type `exit` or `quit` to end the chat.

## Project Structure

- `main.py`: The entry point and interactive terminal UI.
- `llm.py`: Defines the LangGraph workflow, nodes, and the routing logic.
- `vector_db.py`: Handles ChromaDB initialization, storing chunks, retrieval, and prompt injection.
- `crawler.py`: Handles fetching HTML and extracting clean text.
- `chunking_embedding.py`: Handles splitting long texts and generating vector embeddings.

## Notes

- On the very first run, `sentence-transformers` will download the embedding model from Hugging Face. This might take a moment, but subsequent runs will be instant.
