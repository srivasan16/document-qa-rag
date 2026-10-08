# Document QA — RAG-Based Question Answering API

A Retrieval-Augmented Generation (RAG) based Document Question Answering system that allows users to ask questions about documents and receive AI-generated answers grounded in retrieved document content.

## 🚀 Features

- PDF/document text extraction
- Page-number preservation
- Text chunking
- Embedding generation
- ChromaDB vector database
- Semantic document retrieval
- Google Gemini-powered answer generation
- Source/page references in answers
- FastAPI REST API
- Interactive Swagger/OpenAPI documentation
- API-based question answering and testing

## 🏗️ Architecture

```text
PDF Document
     │
     ▼
Document Ingestion
     │
     ▼
Text Extraction + Page Preservation
     │
     ▼
Text Chunking
     │
     ▼
Embeddings
     │
     ▼
ChromaDB Vector Database
     │
     │       User Question
     │             │
     │             ▼
     └──────► Semantic Retrieval
                   │
                   ▼
            Retrieved Context
                   │
                   ▼
            Google Gemini
                   │
                   ▼
          Answer + Sources
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI |
| Vector Database | ChromaDB |
| LLM | Google Gemini |
| API Documentation | Swagger / OpenAPI |
| Environment | Python Virtual Environment |

## 📂 Project Structure

```text
Document QA/
│
├── src/
│   ├── api.py
│   ├── generate_answer.py
│   ├── ingest.py
│   ├── retrieve.py
│   └── ...
│
├── data/
├── chroma_db/
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

> Update the example `src/` filenames if your actual project uses different names.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd "Document QA"
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

**Never upload your API key to GitHub.**

Add `.env` to `.gitignore`.

## 📥 Document Ingestion

Place the required document in the project's data/document directory.

Run the ingestion process:

```powershell
python src/ingest.py
```

The document is processed into text chunks and stored in ChromaDB for semantic retrieval.

## ▶️ Running the API

Start the FastAPI server:

```powershell
uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## 📚 Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger provides an interactive interface for testing the API.

## 🔍 Ask a Question

Use the `/ask` endpoint.

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

1. Receives the question.
2. Searches the ChromaDB vector database.
3. Retrieves relevant document chunks.
4. Sends the retrieved context to Gemini.
5. Generates an answer grounded in the retrieved information.
6. Returns the answer with source/page information.

## 🧪 Testing

The `/ask` endpoint has been tested with multiple questions to verify:

- Retrieval quality
- Answer generation
- Source/page references
- API response handling
- Gemini integration
- End-to-end RAG functionality

## 🔐 Security

API credentials are stored using environment variables.

Do not commit sensitive files such as:

```text
.env
```

to GitHub.

## 🎯 Project Objective

The objective is to build a practical Retrieval-Augmented Generation system capable of answering questions from document knowledge instead of relying only on the language model's pre-trained knowledge.

This improves:

- Document-grounded responses
- Source traceability
- Knowledge retrieval
- Question answering

## 🔮 Future Improvements

Possible future enhancements include:

- Web-based chat interface
- Multiple document support
- Document upload through the API
- Authentication
- Conversation history
- Advanced retrieval/reranking
- Automated evaluation metrics
- Cloud deployment
- Streaming responses
- Docker support

## 👨‍💻 Project Status

**Core RAG pipeline: Completed ✅**

The project supports document ingestion, vector retrieval, Gemini-based answer generation, source/page references, and FastAPI-based question answering.

## 📄 License

This project is intended for educational and internship purposes.
