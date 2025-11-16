from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
CHROMA_DB_PATH = "data/chroma_db"
CHUNKS_METADATA_FILE = "data/chunks_metadata.json"

def load_documents(path, pattern):
    documents = []
    for file_path in Path(path).glob(pattern):
        try:
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
        except Exception as e:
            print(f'Failed to load file {file_path.name}: {e}')
    return documents

def chunk_text(documents, chunk_size=500, chunk_overlap=100):
    """Simple sliding-window text chunking"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def embed_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv('EMBEDDING_MODEL')
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

def build_index():
    print("Loading documents...")
    documents = load_documents(DATA_PATH, "*.txt")

    print("Chunking...")
    chunks = chunk_text(documents, chunk_size=300, chunk_overlap=40)
    
    print(f"Chunks created: {len(chunks)}")


    print("Embedding chunks...")
    embed_chunks(chunks)

    print("Index build complete.")

if __name__ == "__main__":
    build_index()
