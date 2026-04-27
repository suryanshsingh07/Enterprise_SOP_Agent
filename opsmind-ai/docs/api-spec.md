# OpsMind AI - API Specification

## Base URL
`/api`

## 1. Admin / Document Management

### 1.1 Upload SOP Document
Upload a PDF document, chunk it, generate embeddings, and store in MongoDB.

- **Endpoint:** `POST /admin/upload`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `document`: File (PDF only)
- **Response:**
  - `201 Created`
  ```json
  {
    "message": "Document uploaded and processing started",
    "documentId": "60d5ecb54...345"
  }
  ```
  - `400 Bad Request` (No file uploaded)
  - `500 Internal Server Error`

## 2. Query / RAG System

### 2.1 Stream Query Response
Sends a user query, performs vector search for context, and streams the Gemini LLM response using Server-Sent Events (SSE).

- **Endpoint:** `POST /query/stream`
- **Content-Type:** `application/json`
- **Body:**
  ```json
  {
    "query": "How do I process a refund?"
  }
  ```
- **Response:** `200 OK` (Stream: `text/event-stream`)
  - Streams JSON objects prefixed with `data: `
  - Final chunk contains `__SOURCES__` metadata array.
  ```json
  data: {"text": "To process a refund..."}
  data: {"done": true}
  ```
