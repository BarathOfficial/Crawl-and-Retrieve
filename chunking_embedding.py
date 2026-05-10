import logging


def split_text(text, text_splitter):
    try:
        chunks = text_splitter.split_text(text)
        logging.info(f"Text Splitter successfully turned the data into chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error Chunking the data {e}")

def create_embeddings(chunks, model):
    try:
        embeddings = model.encode(chunks)
        logging.info(f"Embedding applied successfully")
        return embeddings
    except Exception as e:
        logging.error(f"Embedding Failed {e}")
    
