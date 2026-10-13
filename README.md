# SentinelRAG

### A Reliability-Focused Retrieval-Augmented Generation System

SentinelRAG is a document-based question-answering system designed to improve the reliability and transparency of Retrieval-Augmented Generation (RAG). It enables users to upload documents, ask questions about their contents, and inspect the retrieved evidence supporting generated answers.

Unlike a basic RAG pipeline, SentinelRAG incorporates hybrid retrieval, cross-encoder reranking, answer verification, confidence-based decision-making, retry logic, and human-review escalation. It uses a local language model through Ollama to generate document-grounded responses.

The system also provides a web dashboard for document management, question answering, retrieval inspection, confidence monitoring, and audit logging.

## Key Features

- **Document Ingestion:** Upload and process PDF and TXT documents.
- **Hybrid Retrieval:** Combine semantic vector search with PostgreSQL full-text search.
- **Cross-Encoder Reranking:** Improve the relevance of retrieved document chunks.
- **Local LLM Inference:** Generate responses using Ollama and Llama 3.2.
- **Answer Verification:** Evaluate answer support using similarity and natural language inference signals.
- **Confidence-Based Decisions:** Determine whether to accept an answer, retry retrieval, or request human review.
- **Retry Mechanism:** Perform an additional retrieval attempt when the initial confidence is insufficient.
- **Document Management:** Upload, list, and delete documents.
- **Audit Logging:** Maintain records of queries and reliability decisions.
- **Interactive Dashboard:** View answers, confidence scores, latency, retry status, and retrieved sources.

## System Architecture

1. **Document Ingestion:** Extract text from uploaded PDF and TXT files.
2. **Text Chunking:** Divide extracted text into smaller chunks for retrieval.
3. **Embedding Generation:** Generate vector representations using Sentence Transformers.
4. **Hybrid Retrieval:** Retrieve relevant chunks using vector similarity and keyword search.
5. **Reranking:** Reorder retrieved chunks using a cross-encoder model.
6. **Answer Generation:** Generate a response using the local LLM and retrieved context.
7. **Verification:** Evaluate retrieval relevance and answer-support signals.
8. **Decision Making:** Accept the answer, retry retrieval, or escalate for human review.
9. **Audit Logging:** Record query activity and system decisions.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Backend | Python, FastAPI |
| Database | PostgreSQL, pgvector |
| Embedding Model | all-MiniLM-L6-v2 |
| Retrieval | Vector Search, PostgreSQL Full-Text Search |
| Reranking Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Answer Verification | NLI Cross-Encoder |
| LLM Runtime | Ollama |
| Language Model | Llama 3.2 |
| Container Tooling | Docker |

## Reliability Decision Pipeline

SentinelRAG uses confidence-based decisions to determine how to handle generated answers.

| Decision | Description |
|---|---|
| `ACCEPT` | The answer meets the configured confidence threshold. |
| `RETRY` | The system retries retrieval with a larger retrieval limit when initial confidence is low. |
| `HUMAN REVIEW` | The response is flagged for human review when confidence remains insufficient. |

These mechanisms are designed to reduce unsupported responses. Confidence scores are not guarantees of correctness, and the system does not eliminate all hallucinations.

## Getting Started

### Prerequisites

- Python and pip
- Node.js and npm
- PostgreSQL with the pgvector extension
- Docker Desktop
- Ollama

### 1. Clone the Repository

```bash
git clone https://github.com/AnuSaha545/SentinelRAG.git
cd SentinelRAG
