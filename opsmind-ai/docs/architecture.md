# OpsMind AI - Architecture

## System Overview
OpsMind AI is an enterprise RAG system that uses MongoDB Atlas Vector Search and Gemini to answer employee questions strictly based on uploaded Standard Operating Procedures (SOPs).

## Core Components
1. **Frontend (React + Vite)**: 
   - Uses Server-Sent Events (SSE) to display real-time streaming responses from the AI.
   - Provides an admin panel for uploading and processing SOPs.
   - State managed via React Context.

2. **Backend (Express + Node.js)**:
   - **File Upload:** Handles multipart data via Multer.
   - **PDF Parsing & Chunking:** Extracts text and chunks it using a 1000 character size and 100 character overlap.
   - **Embeddings:** Generates vector embeddings for each chunk via `@google/genai`.
   - **Vector Search:** Performs cosine similarity search via MongoDB Atlas Vector Search.
   - **RAG Engine:** Constructs a strict prompt using retrieved context and pipes it to Gemini 1.5 Flash.

## Hallucination Control
- **Strict Prompting:** The LLM is explicitly instructed to answer ONLY using the provided context and return "I don't know" otherwise.
- **Source Citations:** The prompt requires the LLM to cite the source name and page number for every piece of information used.

## MongoDB Vector Index Configuration
To enable the retrieval engine, create the following Atlas Vector Search index on the `chunks` collection:
```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 768,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```
