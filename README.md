# Document QA — RAG-Based Question Answering API

A Retrieval-Augmented Generation (RAG) based Document Question Answering system that allows users to ask questions about documents and receive AI-generated answers grounded in retrieved document content.

The project combines PDF text extraction, page-aware document processing, text chunking, embeddings, ChromaDB vector search, and Google Gemini for document-grounded answer generation.

## 🚀 Features

* PDF text extraction
* Page-number preservation
* Document chunking
* Embedding generation
* ChromaDB vector database
* Semantic document retrieval
* Retrieval evaluation
* Google Gemini-powered answer generation
* Source/page references
* FastAPI REST API
* Interactive Swagger/OpenAPI documentation
* Automated retrieval test questions and results

## 🏗️ Architecture

```text
                    PDF Document
                         │
                         ▼
                PDF Text Extraction
                         │
                         ▼
              Page Number Preservation
                         │
                         ▼
                  Text Chunking
                         │
                         ▼
                    Embeddings
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                │  Vector Store   │
                └────────┬────────┘
                         │
                    User Question
                         │
                         ▼
                Semantic Retrieval
                         │
                         ▼
               Retrieved Documents
                         │
                         ▼
                 Google Gemini
                         │
                         ▼
              Answer + Source Pages
```

## 🛠️ Technology Stack

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| API Framework        | FastAPI                        |
| Vector Database      | ChromaDB                       |
| LLM                  | Google Gemini                  |
| Embeddings           | Embedding Model                |
| API Documentation    | Swagger / OpenAPI              |
| Testing              | Python / JSON-based evaluation |
| Environment          | Python Virtual Environment     |

## 📂 Project Structure

```text
Document QA/
│
├── src/
│   ├── build_vector_db.py
│   ├── chunk_documents.py
│   ├── evaluate_retrieval.py
│   ├── extract_pdf.py
│   ├── generate_answer.py
│   └── test_retrieval.py
│
├── tests/
│   ├── evaluation_questions.json
│   ├── retrieval_questions.json
│   └── retrieval_results.json
│
├── api.py
├── evaluation_results.txt
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/srivasan16/document-qa-rag.git
cd document-qa-rag
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Never upload your actual API key to GitHub.

The repository includes `.env.example` as a safe template:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 📥 Document Processing Pipeline

The project processes the source PDF through multiple stages.

### 1. Extract PDF text

```powershell
python src/extract_pdf.py
```

This extracts document text while preserving page information.

### 2. Create document chunks

```powershell
python src/chunk_documents.py
```

The extracted document content is divided into smaller chunks suitable for embedding and retrieval.

### 3. Build the vector database

```powershell
python src/build_vector_db.py
```

The document chunks are converted into embeddings and stored in ChromaDB.

## 🔍 Retrieval Evaluation

The project includes retrieval evaluation scripts and test questions.

Retrieval questions are stored in:

```text
tests/retrieval_questions.json
```

Run the retrieval evaluation:

```powershell
python src/evaluate_retrieval.py
```

Retrieval results are stored in:

```text
tests/retrieval_results.json
```

The project also includes:

```text
evaluation_results.txt
```

for evaluation results and analysis.

## 🤖 Generate Answers

The RAG answer-generation pipeline uses the retrieved document context together with Google Gemini.

Run:

```powershell
python src/generate_answer.py
```

The system retrieves relevant document content and generates a grounded answer from the retrieved context.

## ▶️ Running the FastAPI Server

The FastAPI application is located in the project root:

```text
api.py
```

Start the API with:

```powershell
uvicorn api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## 📚 Swagger API Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger provides an interactive interface for testing the API.

## 🔍 Ask a Question

The main question-answering endpoint is:

```http
POST /ask
```

Example request:

```json
{
  "question": "What is the role of NITI Aayog?"
}
```

The system:

1. Receives the user's question.
2. Searches the ChromaDB vector database.
3. Retrieves relevant document chunks.
4. Uses the retrieved context as grounding information.
5. Sends the relevant context and question to Google Gemini.
6. Generates a document-grounded answer.
7. Returns the answer with source/page information.

## 🧪 Testing

The RAG pipeline was tested using multiple questions.

Testing covers:

* Document retrieval
* Retrieval relevance
* Answer generation
* Source/page references
* Gemini integration
* API response handling
* End-to-end RAG functionality

Test questions are maintained in:

```text
tests/evaluation_questions.json
tests/retrieval_questions.json
```

Retrieval results are maintained in:

```text
tests/retrieval_results.json
```

## 🔐 Security

API credentials are stored using environment variables.

The following sensitive files and local resources are excluded from Git:

```text
.env
chroma_db/
data/
*.pdf
venv/
__pycache__/
```

Never commit a real Gemini API key to GitHub.

## 🎯 Project Objective

The objective of this project is to build a practical Retrieval-Augmented Generation system capable of answering questions from document knowledge rather than relying only on the language model's pre-trained knowledge.

The RAG approach provides:

* Document-grounded responses
* Semantic information retrieval
* Source traceability
* Page-level references
* Reduced dependence on model-only knowledge

## 📊 Project Workflow

```text
PDF
 │
 ▼
Extract Text
 │
 ▼
Preserve Page Information
 │
 ▼
Chunk Documents
 │
 ▼
Generate Embeddings
 │
 ▼
Store in ChromaDB
 │
 ▼
User Question
 │
 ▼
Retrieve Relevant Chunks
 │
 ▼
Gemini
 │
 ▼
Grounded Answer
 │
 ▼
Source/Page References
```

## 🔮 Future Improvements

Possible future enhancements include:

* Web-based chat interface
* Multiple document support
* Document upload through the API
* Authentication
* Conversation history
* Advanced retrieval/reranking
* Automated evaluation metrics
* Cloud deployment
* Streaming responses
* Docker support

## 👨‍💻 Project Status

**Core RAG pipeline: Completed ✅**

The project currently supports:

* PDF document processing
* Page-aware text extraction
* Document chunking
* Embeddings
* ChromaDB vector retrieval
* Retrieval evaluation
* Gemini-based answer generation
* Source/page references
* FastAPI question answering
* Swagger API testing

The project is suitable as an educational and internship-level RAG implementation.

## 🔗 Repository

GitHub:

https://github.com/srivasan16/document-qa-rag

## 📄 License

This project is intended for educational and internship purposes.
