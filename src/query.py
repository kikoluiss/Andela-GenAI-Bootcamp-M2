from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB path
CHROMA_DB_PATH = "data/chroma_db"

# Reranking configuration
RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', '20'))  # Number of candidates to retrieve before reranking
FINAL_TOP_K = int(os.getenv('FINAL_TOP_K', '3'))  # Final number of chunks after reranking
RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')  # Cross-encoder model for reranking
USE_RERANKING = os.getenv('USE_RERANKING', 'true').lower() == 'true'  # Enable/disable reranking

def load_index(embeddings):
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def rerank_documents(question, documents, top_k=FINAL_TOP_K):
    if not documents:
        return documents
    
    try:
        cross_encoder = CrossEncoder(RERANKER_MODEL)
        
        pairs = [[question, doc.page_content] for doc in documents]
        
        scores = cross_encoder.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = [doc for doc, score in scored_docs[:top_k]]
        
        return reranked_docs
    
    except Exception as e:
        print(f"Warning: Reranking failed ({e}), using original order")
        return documents[:top_k]

def generate_answer(user_question, related_documents):
    llm_rag = ChatOpenAI(
        model_name=os.getenv('LLM_MODEL'), 
        temperature=1
    )

    prompt_rag = ChatPromptTemplate.from_messages([
        ("system",
        "You are a FAQ assistant. Use only the provided context to answer."),

        ("human", "QUESTION: {user_question}\n\nContext:\n{context}")
    ])

    document_chain = create_stuff_documents_chain(
        llm_rag,
        prompt_rag
    )

    answer = document_chain.invoke({
        'user_question': user_question,
        'context': related_documents
    })

    return answer.strip()

def answer_question(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv('EMBEDDING_MODEL')
    )
    vectorstore = load_index(embeddings)
    
    if USE_RERANKING:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RERANK_TOP_K}
        )
        candidate_documents = retriever.invoke(user_question)
        related_documents = rerank_documents(user_question, candidate_documents, top_k=FINAL_TOP_K)
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": FINAL_TOP_K}
        )
        related_documents = retriever.invoke(user_question)

    system_answer = generate_answer(user_question, related_documents)

    chunks_related = []
    for document in related_documents:
        chunks_related.append({
            "source": document.metadata['source'],
            "content": document.page_content
        })

    return {
        "user_question": user_question,
        "system_answer": system_answer,
        "chunks_related": chunks_related
    }

if __name__ == "__main__":
    user_question = input("Ask a question: ")
    result = answer_question(user_question)
    print(json.dumps(result, indent=2))
