# OpsMind AI

A production-ready enterprise context-aware corporate knowledge brain using RAG (Retrieval-Augmented Generation).

## Features
- **Zero Hallucination Tolerance**: Strict grounding on provided SOPs.
- **Mandatory Source Citation**: Clickable references to exact PDF pages.
- **Scalable Architecture**: Modular Node.js backend and React frontend.
- **Streaming Responses**: Real-time SSE streaming for chat UX.

## Setup Instructions

### Prerequisites
- Node.js (v18+)
- MongoDB Atlas cluster (with Vector Search capabilities)
- Google Gemini API Key

### 1. MongoDB Atlas Configuration
1. Create a database named `opsmind`.
2. In the `chunks` collection, create an Atlas Vector Search index named `vector_index` using the JSON provided in `docs/architecture.md`.

### 2. Backend Setup
1. Navigate to `backend` directory.
2. Run `npm install` (requires Node.js to be installed on your machine).
3. Update `.env` with your `MONGO_URI` and `GEMINI_API_KEY`.
4. Run `npm run dev` to start the server on port 5000.

### 3. Frontend Setup
1. Navigate to `frontend` directory.
2. Run `npm install`.
3. Run `npm run dev` to start the Vite preview on port 5173.

## Example Queries + Outputs
**Test Case 1 (Found in SOP):**
- *Query:* "How do I process a refund?"
- *Output:* "To process a refund, navigate to the billing portal, locate the transaction ID, and click 'Refund'. (Source: Billing_SOP.pdf, Pg 4, Section 2)"

**Test Case 2 (Not in SOP):**
- *Query:* "What is the CEO's favorite color?"
- *Output:* "I don't know based on the provided SOPs."
