# RAG FAQ System

A Python application that uses Retrieval Augmented Generation (RAG) to answer questions from a knowledge base of documents. The system uses LangChain, ChromaDB for vector storage, HuggingFace embeddings, and OpenAI's language models to provide accurate, context-aware answers.

## Features

- **Document Indexing**: Build a searchable vector index from text documents
- **RAG-based Q&A**: Answer questions using retrieved context from the knowledge base
- **Reranking**: Two-stage retrieval with cross-encoder reranking for improved relevance
- **Answer Evaluation**: Evaluate the quality of generated answers with scoring
- **Chunk Retrieval**: Returns relevant document chunks used to generate each answer
- **Modular Architecture**: Clean separation between indexing, querying, and evaluation
- **Configurable**: Environment-based configuration for models, reranking, and retrieval parameters

## Setup

### Prerequisites

- Python 3.7 or higher
- OpenAI API key
- [OPTIONAL] HuggingFace account (for non-free embedding models)

### Installation

1. Clone or navigate to this repository:
   ```bash
   cd M2
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The required packages are:
   - `langchain==0.3.27` - LangChain framework for LLM orchestration
   - `langchain-community==0.3.29` - Community integrations
   - `langchain-openai==0.3.35` - OpenAI integration
   - `langchain-huggingface==0.3.1` - HuggingFace embeddings
   - `langchain-chroma==0.2.6` - ChromaDB vector store
   - `sentence-transformers==5.1.1` - Sentence transformers for embeddings
   - `python-dotenv==1.1.1` - Environment variable management

## Environment Variables

Create a `.env` file in the project root directory with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-5-nano  # or your preferred OpenAI model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # or your preferred embedding model
TOKENIZERS_PARALLELISM=false  # should be false for local experiments

# Reranking configuration (optional)
USE_RERANKING=true  # Enable/disable reranking (default: true)
RERANK_TOP_K=20  # Number of candidates to retrieve before reranking (default: 20)
FINAL_TOP_K=3  # Final number of chunks after reranking (default: 3)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Cross-encoder model for reranking
```

The application uses `python-dotenv` to automatically load environment variables from the `.env` file at startup.

**Security Note**: Never commit your `.env` file to version control. Add `.env` to your `.gitignore` file.

## Usage

### Indexing Pipeline

Build the vector index from your documents. This is a one-time setup step that must be completed before querying.

**Command:**
```bash
python src/build_index.py
```

**What it does:**
- Loads all `.txt` files from the `data/` directory
- Splits documents into chunks using LangChain's `RecursiveCharacterTextSplitter` (300 characters with 40 character overlap)
- Generates embeddings using the HuggingFace Sentence-Transformers model specified in `.env`
- Stores the indexed documents in `data/chroma_db/` for persistent vector storage

**Example output:**
```
Loading documents...
Chunking...
Chunks created: 245
Embedding chunks...
Index build complete.
```

**Note**: Make sure your documents are in the `data/` directory as `.txt` files before running this command. You only need to rebuild the index when you add or modify documents.

### Query Pipeline

Ask questions interactively to get answers from your knowledge base.

**Command:**
```bash
python src/query.py
```

**Example session:**
```bash
$ python src/query.py
Ask a question: How do I reset my HR portal password?
```

**What it does:**
- Embeds your question using the same Sentence-Transformers model
- Performs similarity search in ChromaDB to retrieve candidate chunks (default: top 20)
- **Reranks candidates** using a cross-encoder model for better relevance (if enabled)
- Selects the top 3 most relevant chunks after reranking
- Passes the retrieved context to the OpenAI LLM
- Generates an answer based on the retrieved context
- Returns the answer along with the source chunks used

**Example output:**
```json
{
  "user_question": "How do I reset my HR portal password?",
  "system_answer": "You can reset your password from the login screen by selecting 'Forgot Password.' Enter your registered email address and check your inbox for a reset link. The link expires after 30 minutes. If you do not receive the email, ensure that your spam filter is not blocking noreply@hrplus.com. Administrators can also manually reset passwords from the 'User Management' panel.",
  "chunks_related": [
    {
      "source": "data/faq_document.txt",
      "content": "You can reset your password from the login screen by selecting 'Forgot Password.' Enter your registered email address and check your inbox for a reset link. The link expires after 30 minutes. If you do not receive the email, ensure that your spam filter is not blocking noreply@hrplus.com."
    },
    {
      "source": "data/faq_document.txt",
      "content": "1. Account Access & Login Support\nHow do I reset my password?"
    },
    {
      "source": "data/faq_document.txt",
      "content": "is not blocking noreply@hrplus.com. Administrators can also manually reset passwords from the 'User Management' panel."
    }
  ]
}
```

**More example questions:**
- `"Can I work overtime without authorization?"`
- `"How many vacation days can I have per year?"`
- `"What is the company's policy on remote work?"`

### Evaluation Pipeline

Evaluate the quality of answers from sample queries using an LLM-based evaluator.

**Command:**
```bash
python src/evaluator.py
```

**What it does:**
- Loads sample queries from `outputs/sample_queries.json`
- Evaluates each answer using an LLM-based evaluator
- Returns scores (0-10) and justifications for each answer

**Example output:**
```json
{
  "score": 8,
  "justification": "The answer accurately addresses the question and uses relevant context from the retrieved chunks. It provides clear, actionable steps for password reset."
}
```

## Project Structure

```
m2/
├── src/                          # Source code directory
│   ├── build_index.py            # Indexing pipeline: loads documents, chunks text, 
│   │                             # generates embeddings, and stores in ChromaDB
│   ├── query.py                  # Query pipeline: performs similarity search with 
│   │                             # reranking, retrieves relevant chunks, and generates 
│   │                             # answers using RAG
│   ├── evaluator.py              # Evaluation pipeline: scores answer quality using 
│   │                             # LLM-based evaluation
│   └── test_core.py              # Test suite for validating query output format and 
│                                 # core functionality
├── data/                         # Data directory
│   ├── *.txt                     # Source documents (FAQ, policies, payroll, etc.)
│   │                             # Files: faq_document.txt, policies.txt, payroll.txt,
│   │                             # benefits.txt, onboarding.txt, time_tracking.txt
│   └── chroma_db/                # ChromaDB vector database (created after indexing)
│                                 # Contains persistent vector embeddings and metadata
├── outputs/                      # Output directory
│   └── sample_queries.json       # Sample queries with answers and chunks for evaluation
├── tests/                        # Test directory (currently empty)
├── reports/                      # Reports directory
│   └── REPORTS.md                # Project reports and documentation
├── requirements.txt              # Python package dependencies
├── .env                          # Environment variables (API keys, model configs)
│                                 # Not committed to version control
└── README.md                     # This file - project documentation
```

## How It Works

### Architecture

1. **Indexing Phase** (`build_index.py`):
   - Documents are loaded from `data/*.txt`
   - Text is split into overlapping chunks
   - Each chunk is embedded using HuggingFace embeddings
   - Embeddings are stored in ChromaDB vector store

2. **Query Phase** (`query.py`):
   - User question is embedded using the same embedding model
   - Similarity search retrieves top-k (default: 3) relevant chunks
   - Retrieved chunks are passed as context to OpenAI LLM
   - LLM generates answer based on the context

3. **Evaluation Phase** (`evaluator.py`):
   - Uses an LLM to evaluate answer quality
   - Provides scores and justifications
   - Helps assess system performance

### Technical Choices

#### Text Chunking
- **Method**: LangChain's `RecursiveCharacterTextSplitter`
- **Chunk Size**: 300 characters
- **Chunk Overlap**: 40 characters
- **Rationale**: RecursiveCharacterTextSplitter intelligently splits text by attempting to keep paragraphs, sentences, and words together when possible, which helps preserve semantic meaning. The overlap ensures context continuity between chunks.

#### Embeddings
- **Model**: Sentence-Transformers from HuggingFace (default: `all-MiniLM-L6-v2`)
- **Library**: `langchain-huggingface` integration
- **Rationale**: Sentence-Transformers provide high-quality semantic embeddings optimized for similarity search. The `all-MiniLM-L6-v2` model offers a good balance between performance and speed, generating 384-dimensional vectors.

#### Vector Search
- **Database**: ChromaDB
- **Search Method**: Similarity search (cosine similarity)
- **Initial Retrieval**: Retrieves top-k candidates (default: 20) using similarity search
- **Reranking**: Uses cross-encoder model (`ms-marco-MiniLM-L-6-v2`) to rerank candidates for better relevance
- **Final Top-K Retrieval**: k=3 (selects 3 most relevant chunks after reranking)
- **Rationale**: 
  - ChromaDB provides efficient similarity search with persistent storage
  - Reranking with cross-encoders improves relevance by considering question-document pairs together
  - Two-stage retrieval (similarity + reranking) provides better accuracy than similarity search alone
  - Final top 3 chunks balance context richness with token efficiency

### Retrieval Strategy

- **Search Type**: Two-stage retrieval with reranking
  1. **Initial Retrieval**: Similarity search (cosine similarity) retrieves top-k candidates (default: 20)
  2. **Reranking**: Cross-encoder model reranks candidates by relevance to the question
  3. **Final Selection**: Top 3 most relevant chunks after reranking
- **Chunk Size**: 300 characters
- **Chunk Overlap**: 40 characters
- **Reranking Model**: Cross-encoder (default: `ms-marco-MiniLM-L-6-v2`)

### Model Configuration

- **Embedding Model**: Configurable via `EMBEDDING_MODEL` env var (default: `sentence-transformers/all-MiniLM-L6-v2`)
- **LLM Model**: Configurable via `LLM_MODEL` env var (default: `gpt-5-nano`)
- **Temperature**: 1.0 (for more diverse responses)
- **Retrieval**: Top 3 chunks by similarity

## Running Tests

Run the test suite to validate core functionalities:

```bash
python src/test_core.py
```

The test suite validates:
- Query output format correctness
- Required fields in response
- Data type validation
- Chunk structure and content validation
- Response consistency across multiple calls
- Handling of different question types (short, long, special characters)

## Output Format

### Query Response

```json
{
  "user_question": "How do I reset my HR portal password?",
  "system_answer": "You can reset your password from the login screen...",
  "chunks_related": [
    {
      "source": "data/faq_document.txt",
      "content": "You can reset your password from the login screen..."
    },
    {
      "source": "data/faq_document.txt",
      "content": "1. Account Access & Login Support..."
    }
  ]
}
```

### Evaluation Response

```json
{
  "score": 8,
  "justification": "The answer accurately addresses the question and uses relevant context..."
}
```

## Data Files

The system expects text files in the `data/` directory. Example files included:
- `faq_document.txt` - Frequently asked questions
- `policies.txt` - Company policies
- `payroll.txt` - Payroll information
- `benefits.txt` - Employee benefits
- `onboarding.txt` - Onboarding information
- `time_tracking.txt` - Time tracking policies

## Known Limitations

### 1. Chunk Size and Overlap

- **Fixed Parameters**: Chunk size (300) and overlap (40) are hardcoded
- **Impact**: May not be optimal for all document types
- **Solution**: Consider making these configurable or document-specific

### 2. Retrieval Strategy

- **Simple Similarity**: Uses basic similarity search without reranking
- **Fixed Top-K**: Always retrieves exactly 3 chunks
- **No Metadata Filtering**: Cannot filter by document type or metadata

### 3. Answer Generation

- **No Citation**: Generated answers don't explicitly cite which chunks were used
- **Context Window**: Limited by the model's context window
- **No Confidence Scores**: Doesn't provide confidence metrics for answers

### 4. Evaluation

- **Subjective Scoring**: Evaluation relies on LLM judgment, which can be inconsistent
- **No Ground Truth**: No comparison against reference answers
- **Single Metric**: Only provides a single score without detailed breakdown

### 5. Error Handling

- **Limited Recovery**: Minimal error recovery for API failures
- **No Retry Logic**: Doesn't retry failed API calls
- **Silent Failures**: Some errors may not be clearly communicated

## Future Improvements

- **Configurable Chunking**: Make chunk size and overlap configurable
- **Advanced Retrieval**: Implement reranking, hybrid search, or metadata filtering
- **Answer Citations**: Add explicit citations to source chunks
- **Confidence Metrics**: Provide confidence scores for answers
- **Batch Processing**: Support batch query processing
- **Caching**: Cache frequent queries to reduce API calls
- **Web Interface**: Add a web UI for easier interaction
- **Multi-Model Support**: Support different embedding and LLM models
- **Evaluation Metrics**: Add more comprehensive evaluation metrics (BLEU, ROUGE, etc.)

## License

MIT
