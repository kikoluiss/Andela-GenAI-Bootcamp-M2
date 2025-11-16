# RAG FAQ System - Project Report

## Architecture Overview

### System Architecture

The RAG FAQ System is a Python-based Retrieval Augmented Generation (RAG) application built using LangChain, ChromaDB, HuggingFace embeddings, and OpenAI's language models. The architecture follows a three-phase pipeline design with clear separation between indexing, querying, and evaluation:

```
┌─────────────────────────────────────────────────────────────┐
│                    Indexing Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Document     │  │ Text         │  │ Embedding    │       │
│  │ Loading      │→ │ Chunking     │→ │ Generation   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                          │                                  │
│                          ▼                                  │
│                  ┌──────────────┐                           │
│                  │ ChromaDB     │                           │
│                  │ Vector Store │                           │
│                  └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Query Pipeline                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Question     │  │ Similarity   │  │ Reranking    │       │
│  │ Embedding    │→ │ Search       │→ │ (Cross-      │       │
│  └──────────────┘  └──────────────┘  │  Encoder)    │       │
│                          │           └──────────────┘       │
│                          │                    │             │
│                          ▼                    ▼             │
│                  ┌──────────────┐    ┌──────────────┐       │
│                  │ Candidates   │    │ Top-K Chunks │       │
│                  │ (k=20)       │    │ (k=3)        │       │
│                  └──────────────┘    └──────────────┘       │
│                          │                    │             │
│                          │                    ▼             │
│                          │           ┌──────────────┐       │
│                          │           │ Answer       │       │
│                          │           │ Generation   │       │
│                          │           │ (RAG)        │       │
│                          │           └──────────────┘       │
│                          │                    │             │
│                          └────────────────────┘             │
│                                    │                        │
│                                    ▼                        │
│                          ┌──────────────┐                   │
│                          │ OpenAI LLM   │                   │
│                          └──────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Evaluation Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Answer       │  │ LLM-based    │  │ Score &      │       │
│  │ Evaluation   │→ │ Evaluation   │→ │ Justification│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Indexing Pipeline** (`build_index.py`)
   - **Document Loader**: Loads all `.txt` files from the `data/` directory using LangChain's `TextLoader`
   - **Text Chunker**: Splits documents into overlapping chunks using `RecursiveCharacterTextSplitter` (300 chars, 40 overlap)
   - **Embedding Generator**: Generates vector embeddings using HuggingFace Sentence-Transformers
   - **Vector Store**: Persists embeddings and metadata in ChromaDB for efficient similarity search

2. **Query Pipeline** (`query.py`)
   - **Question Embedding**: Embeds user questions using the same embedding model
   - **Initial Retriever**: Performs similarity search in ChromaDB to find top-k candidates (default: k=20)
   - **Reranker**: Uses cross-encoder model to rerank candidates by relevance to the question
   - **Final Selection**: Selects top-k (k=3) most relevant chunks after reranking
   - **Answer Generator**: Uses OpenAI LLM with RAG pattern to generate context-aware answers
   - **Response Formatter**: Returns structured JSON with question, answer, and source chunks

3. **Evaluation Pipeline** (`evaluator.py`)
   - **Answer Evaluator**: Uses LLM-based evaluation to score answer quality (0-10)
   - **Context Analyzer**: Evaluates how well answers use retrieved context
   - **Score Generator**: Provides scores and justifications for each answer

### Data Flow

1. **Indexing Flow**:
   ```
   Text Files → Document Loading → Chunking → Embedding → ChromaDB Storage
   ```

2. **Query Flow**:
   ```
   User Question → Embedding → Similarity Search → Candidate Retrieval (k=20) → 
   Reranking (Cross-Encoder) → Top-K Selection (k=3) → Context Injection → 
   LLM Generation → Answer + Source Chunks
   ```

3. **Evaluation Flow**:
   ```
   Question + Answer + Chunks → LLM Evaluation → Score + Justification
   ```

### Technology Stack

- **Language**: Python 3.7+
- **Framework**: LangChain 0.3.27
- **Vector Database**: ChromaDB (via langchain-chroma 0.2.6)
- **Embeddings**: HuggingFace Sentence-Transformers (via langchain-huggingface 0.3.1)
- **Reranking**: Cross-Encoder models (via sentence-transformers)
- **LLM**: OpenAI (via langchain-openai 0.3.35)
- **Document Processing**: LangChain Community 0.3.29
- **Configuration**: python-dotenv 1.1.1 for environment variables

## Prompt Technique(s) Used and Why

### Primary Technique: Retrieval Augmented Generation (RAG) with Context Injection

The application employs **Retrieval Augmented Generation (RAG)** as its core prompt engineering technique. This combines information retrieval with language model generation to provide accurate, context-aware answers.

#### Implementation

**Query Pipeline Prompt**:
```python
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "You are a FAQ assistant. Use only the provided context to answer."),
    
    ("human", "QUESTION: {user_question}\n\nContext:\n{context}")
])
```

**Evaluation Pipeline Prompt**:
```python
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate the answer generated by a RAG system. "
     "Provide: "
     "1. A score from 0–10; "
     "2. A short justification; "
     "Return JSON."),
    
    ("human", 
     "USER QUESTION: {user_question}\n\n"
     "SYSTEM ANSWER: {system_answer}\n\n"
     "CONTEXT:\n{context}")
])
```

#### Why RAG?

1. **Accuracy and Grounding**
   - **Problem**: LLMs can hallucinate or provide outdated information when relying solely on training data
   - **Solution**: RAG retrieves relevant context from the knowledge base before generating answers
   - **Result**: Answers are grounded in actual documents, reducing hallucinations and improving accuracy

2. **Domain-Specific Knowledge**
   - **Problem**: General-purpose LLMs may not have specific company policies, procedures, or FAQ information
   - **Solution**: RAG allows the system to use domain-specific documents (policies, FAQs, benefits) as context
   - **Result**: The system can answer questions about specific company information without fine-tuning

3. **Updatability**
   - **Problem**: Fine-tuned models require retraining to incorporate new information
   - **Solution**: RAG systems can be updated by simply adding new documents to the vector store
   - **Result**: Easy to maintain and update without model retraining

4. **Transparency and Traceability**
   - **Problem**: Black-box LLM responses don't show sources
   - **Solution**: RAG returns source chunks used to generate each answer
   - **Result**: Users can verify answers by checking source documents

#### Why Context Injection Pattern?

1. **Explicit Context Boundary**
   - The system prompt "Use only the provided context to answer" constrains the model
   - Prevents the model from using general knowledge when specific context is available
   - Ensures answers are based on retrieved documents

2. **Clear Structure**
   - Separating QUESTION and Context in the prompt provides clear structure
   - Makes it easy for the model to distinguish between the question and supporting context
   - Follows LangChain's recommended RAG pattern

3. **Error Prevention**
   - Explicit instruction to use only provided context reduces hallucination
   - If context is insufficient, the model can indicate this rather than making up information

#### Technical Implementation Details

- **Chunking Strategy**: `RecursiveCharacterTextSplitter` with 300-character chunks and 40-character overlap
  - Preserves semantic meaning by keeping sentences/paragraphs together
  - Overlap ensures context continuity across chunk boundaries

- **Retrieval Strategy**: Two-stage retrieval with reranking
  - **Stage 1**: Similarity search (cosine similarity) retrieves top-k candidates (default: k=20)
  - **Stage 2**: Cross-encoder reranking scores question-document pairs for better relevance
  - **Final Selection**: Top-k chunks (default: k=3) selected after reranking
  - Balances context richness with token efficiency
  - Provides enough information for accurate answers without overwhelming the context window
  - Reranking improves relevance by considering question-document pairs together

- **Embedding Model**: Sentence-Transformers (default: `all-MiniLM-L6-v2`)
  - Optimized for semantic similarity search
  - 384-dimensional vectors provide good balance of quality and speed

#### Model Parameters

- **Temperature: 1.0** - Higher temperature for more natural, diverse responses
- **Model**: Configurable via `LLM_MODEL` env var (default: `gpt-5-nano`)
- **Rationale**: FAQ answers benefit from natural language while maintaining accuracy through context grounding

## Challenges

### 1. Chunk Size and Overlap Optimization

**Challenge**: Finding optimal chunk size and overlap for different document types.

**Issues Encountered**:
- **Fixed Parameters**: Chunk size (300) and overlap (40) are hardcoded
- **Document Diversity**: Different document types (FAQs, policies, technical docs) may need different chunking strategies
- **Context Loss**: Small chunks may split important information across boundaries
- **Overhead**: Large overlap increases storage and retrieval costs

**Impact**:
- Some answers may miss context that was split across chunks
- Optimal chunking may vary by document type (short FAQs vs. long policy documents)
- Current settings work reasonably well but may not be optimal for all use cases

### 2. Retrieval Quality and Relevance

**Challenge**: Ensuring retrieved chunks are truly relevant to the question.

**Status**: **Partially Addressed** - Reranking has been implemented to improve relevance.

**Issues Encountered**:
- **Semantic Mismatch**: Embedding similarity doesn't always correlate with answer relevance
- **Keyword vs. Semantic**: Questions with specific keywords may not match semantically similar but different content
- **Fixed K**: Always retrieves exactly 3 chunks after reranking, even when fewer or more might be better

**Current Solution**:
- **Reranking Implemented**: Cross-encoder reranking now reranks candidates (top-20) before final selection
- **Improved Relevance**: Two-stage retrieval (similarity + reranking) provides better chunk selection
- **Configurable**: Reranking can be enabled/disabled and parameters adjusted via environment variables

**Remaining Issues**:
- No dynamic k-selection based on question complexity
- Reranking adds computational overhead (though minimal with efficient cross-encoder models)
- Still limited to fixed final k value

**Impact**:
- Relevance has improved with reranking implementation
- Some edge cases may still benefit from dynamic k-selection
- Reranking helps but doesn't solve all semantic mismatch issues

### 3. Context Window Limitations

**Challenge**: Balancing context richness with token limits.

**Issues Encountered**:
- **Token Limits**: Model context windows limit how much context can be included
- **Chunk Selection**: With k=3, some questions may need more context, others less
- **No Prioritization**: All retrieved chunks are treated equally in the prompt
- **Truncation Risk**: Very long chunks might be truncated or not fully utilized

**Impact**:
- Complex questions might need more than 3 chunks
- Simple questions waste tokens on unnecessary context
- Cannot adapt context size based on question complexity

### 4. Answer Quality and Evaluation

**Challenge**: Ensuring generated answers are accurate and well-grounded.

**Issues Encountered**:
- **Hallucination Risk**: Even with context, models may add information not in the retrieved chunks
- **Context Ignoring**: Model might ignore context and use general knowledge
- **Incomplete Answers**: Model might not use all relevant information from retrieved chunks
- **Subjective Evaluation**: LLM-based evaluation is subjective and may be inconsistent

**Impact**:
- Some answers may contain information not in source documents
- Evaluation scores may vary between runs
- Difficult to objectively measure answer quality

### 5. Error Handling and Robustness

**Challenge**: Handling failures gracefully across the pipeline.

**Issues Encountered**:
- **API Failures**: No retry logic for OpenAI API failures
- **Empty Retrieval**: No handling for cases where no relevant chunks are found
- **Embedding Failures**: No fallback if embedding generation fails
- **ChromaDB Errors**: Limited error handling for vector database issues
- **Silent Failures**: Some errors may not be clearly communicated to users

**Impact**:
- System may crash on transient API failures
- Poor user experience when errors occur
- Difficult to diagnose issues

### 6. Configuration and Flexibility

**Challenge**: System parameters are hardcoded and not easily configurable.

**Issues Encountered**:
- **Fixed Chunking**: Chunk size and overlap cannot be changed without code modification
- **Fixed Retrieval**: Top-k value (3) is hardcoded
- **Model Selection**: Model selection requires environment variable changes
- **No A/B Testing**: Cannot easily test different configurations

**Impact**:
- Cannot optimize for different document types
- Difficult to experiment with different retrieval strategies
- Limited flexibility for different use cases

### 7. Performance and Scalability

**Challenge**: System performance with larger document collections.

**Issues Encountered**:
- **Sequential Processing**: No parallelization in indexing or querying
- **No Caching**: Repeated queries require full retrieval and generation
- **Embedding Computation**: Embeddings are computed on-the-fly for queries
- **No Index Optimization**: No indexing strategies for faster retrieval

**Impact**:
- Slow indexing for large document collections
- Repeated queries are inefficient
- May not scale well to thousands of documents

## Improvements

### Short-Term Improvements (High Priority)

#### 1. Enhanced Chunking Strategy

**Improvement**: Make chunking configurable and document-type aware.

**Proposed Changes**:
- Add configuration file for chunk size and overlap per document type
- Implement semantic chunking (chunk by topic/section rather than character count)
- Add metadata to chunks (document type, section, importance)
- Support multiple chunking strategies (by paragraph, by section, by topic)

**Expected Benefits**:
- Better context preservation for different document types
- More relevant chunk retrieval
- Improved answer quality

#### 2. Improved Retrieval with Reranking

**Status**: **Partially Implemented** - Reranking has been added, dynamic k-selection remains.

**Implemented**:
- ✅ Cross-encoder reranking for retrieved chunks (using `ms-marco-MiniLM-L-6-v2`)
- ✅ Two-stage retrieval: similarity search (k=20) → reranking → final selection (k=3)
- ✅ Configurable via environment variables (`USE_RERANKING`, `RERANK_TOP_K`, `FINAL_TOP_K`)

**Remaining Improvements**:
- Add dynamic k-selection based on question complexity
- Implement hybrid search (combine keyword and semantic search)
- Add relevance threshold filtering
- Consider query-specific reranking model selection

**Expected Benefits** (from implemented reranking):
- ✅ More relevant chunks in context
- ✅ Better answer quality
- ✅ Reduced token waste on irrelevant chunks

**Future Benefits** (from remaining improvements):
- Adaptive context size based on question complexity
- Better handling of keyword-specific queries
- Filtering out low-relevance chunks even after reranking

#### 3. Robust Error Handling

**Improvement**: Add comprehensive error handling and retry logic.

**Proposed Changes**:
- Implement exponential backoff retry for API calls
- Add graceful degradation when retrieval fails
- Provide clear error messages to users
- Add logging for debugging
- Handle edge cases (empty retrieval, API timeouts)

**Expected Benefits**:
- Better reliability
- Improved user experience
- Easier debugging

#### 4. Configuration Management

**Improvement**: Make system parameters configurable.

**Proposed Changes**:
- Add `config.yaml` for chunking, retrieval, and model parameters
- Allow per-query parameter overrides
- Support environment-based configurations
- Add validation for configuration values

**Expected Benefits**:
- Easy experimentation with different settings
- Flexibility for different use cases
- Better maintainability

### Medium-Term Improvements

#### 5. Advanced Retrieval Strategies

**Improvement**: Implement more sophisticated retrieval methods.

**Proposed Changes**:
- Add metadata filtering (filter by document type, date, etc.)
- Implement query expansion (synonyms, related terms)
- Add query rewriting for better retrieval
- Support multi-vector retrieval (combine multiple embedding models)

**Expected Benefits**:
- Better retrieval accuracy
- More flexible querying
- Improved answer quality

#### 6. Answer Quality Improvements

**Improvement**: Enhance answer generation and validation.

**Proposed Changes**:
- Add answer validation against retrieved chunks
- Implement citation generation (explicit references to source chunks)
- Add confidence scores for answers
- Detect and flag potential hallucinations
- Support follow-up questions and conversation context

**Expected Benefits**:
- More accurate answers
- Better transparency
- Improved user trust

#### 7. Performance Optimization

**Improvement**: Optimize system performance and scalability.

**Proposed Changes**:
- Add query caching for repeated questions
- Implement batch processing for multiple queries
- Add parallel processing for indexing
- Optimize ChromaDB queries
- Add connection pooling for API calls

**Expected Benefits**:
- Faster response times
- Better scalability
- Reduced API costs

#### 8. Enhanced Evaluation

**Improvement**: Improve answer evaluation and metrics.

**Proposed Changes**:
- Add multiple evaluation metrics (relevance, accuracy, completeness)
- Implement ground truth comparison
- Add evaluation dataset creation tools
- Generate evaluation reports with statistics
- Track evaluation metrics over time

**Expected Benefits**:
- Better understanding of system performance
- Data-driven improvements
- Objective quality measurement

### Long-Term Improvements

#### 9. Multi-Modal Support

**Improvement**: Support different document types and formats.

**Proposed Changes**:
- Add support for PDF, DOCX, HTML documents
- Implement table extraction and processing
- Add image processing for diagrams/charts
- Support structured data (JSON, CSV)

**Expected Benefits**:
- Broader document support
- More comprehensive knowledge base
- Better user experience

#### 10. Advanced RAG Techniques

**Improvement**: Implement state-of-the-art RAG improvements.

**Proposed Changes**:
- Add query compression (summarize retrieved chunks)
- Implement parent-document retrieval (retrieve larger context)
- Add self-RAG (model decides when to retrieve)
- Implement RAG fusion (combine multiple retrieval strategies)

**Expected Benefits**:
- Better answer quality
- More efficient context usage
- State-of-the-art performance

#### 11. Web Interface and API

**Improvement**: Add web interface and REST API.

**Proposed Changes**:
- Build FastAPI REST API
- Create web UI for querying
- Add authentication and rate limiting
- Implement query history and analytics
- Support batch query processing via API

**Expected Benefits**:
- Easier integration
- Better user experience
- Scalable architecture

#### 12. Monitoring and Analytics

**Improvement**: Add comprehensive monitoring and analytics.

**Proposed Changes**:
- Track query patterns and popular questions
- Monitor answer quality metrics
- Add performance dashboards
- Implement alerting for system issues
- Generate usage reports

**Expected Benefits**:
- Better system understanding
- Proactive issue detection
- Data-driven optimization

### Code Quality Improvements

#### 13. Testing Enhancements

**Proposed Changes**:
- Add integration tests with mock vector stores
- Add unit tests for all functions
- Test edge cases (empty retrieval, API failures)
- Add property-based testing
- Achieve >80% test coverage

#### 14. Documentation

**Proposed Changes**:
- Add comprehensive docstrings
- Create API documentation
- Add architecture decision records (ADRs)
- Create developer onboarding guide
- Document configuration options

#### 15. Type Hints and Linting

**Proposed Changes**:
- Add complete type hints
- Configure mypy for type checking
- Add pre-commit hooks
- Enforce code style with black/isort
- Add CI/CD pipeline

## Conclusion

The RAG FAQ System successfully demonstrates a working Retrieval Augmented Generation implementation using LangChain, ChromaDB, and OpenAI. The architecture is clean and modular, with clear separation between indexing, querying, and evaluation phases. Key strengths include accurate, context-grounded answers, easy document updates, and transparent source attribution.

The RAG prompt technique proves effective for FAQ systems, providing accurate answers grounded in company-specific documents without requiring model fine-tuning. The system can answer questions about policies, benefits, payroll, and other company-specific information by retrieving relevant context from the knowledge base.

The system includes a cross-encoder reranking, which significantly improves retrieval quality by using a two-stage approach: initial similarity search retrieves candidates (k=20), followed by cross-encoder reranking to select the most relevant chunks (k=3). This enhancement addresses the previous challenge of retrieval quality and relevance, resulting in better answer accuracy.

Primary remaining areas for improvement center on dynamic k-selection, error handling and robustness, configuration flexibility, and performance optimization. With the proposed improvements, the system could evolve into a production-ready, scalable FAQ solution capable of handling diverse document types and complex queries while maintaining high answer quality and user trust.
