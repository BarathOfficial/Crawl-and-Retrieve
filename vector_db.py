import logging
import chromadb

def get_chroma_client(db_path):
    try:
        client = chromadb.PersistentClient(path=db_path)
        logging.info(f"ChromaDB persistent client initialized at {db_path}")
        return client
    except Exception as e:
        logging.error(f"Error initializing ChromaDB client: {e}")
        raise e

def get_or_create_collection(client, collection_name):
    try:
        collection = client.get_or_create_collection(name=collection_name)
        return collection
    except Exception as e:
        logging.error(f"Error getting/creating collection {collection_name}: {e}")
        raise e

def store_Web_chunks(chunks, url, embeddings, collection):
    try:
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"{url}_{i}"),
            metadatas.append({
                "source": url,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "document_typr": "WebPages"
            })

        collection.add(
            documents= chunks,
            embeddings= embeddings.tolist(),
            metadatas= metadatas,
            ids= ids
        )
        logging.info("Chunks Stored in the ChromaDB successfully")
    except Exception as e:
        logging.error(f"Error storing chunks in the DB: {e}")


def retriver(query, model, collection, top_k=3):
    try:
        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results= top_k
        )
        logging.info("Data retrived successfully")
        return results
    except Exception as e:
        logging.error(f"Error retriving data from DB: {e}")
        return None
    

def build_prompt(query, context_docs):

    context = "\n\n".join(context_docs)

    prompt = f"""
        You are a helpful assistant.

        Answer ONLY using the context below.

        Context:
        {context}

        Question:
        {query}
    """
    return prompt