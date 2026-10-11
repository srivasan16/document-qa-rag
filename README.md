# 🚀 LLM Cost Optimisation Study

> Cut the cost of a Retrieval-Augmented Generation (RAG) system without materially reducing answer quality.

---

## 📌 Overview

This project demonstrates practical techniques to optimize the cost of an LLM-powered RAG (Retrieval-Augmented Generation) application.

Instead of changing the language model, this project reduces token usage and inference cost by optimizing the retrieval pipeline.

The study compares different retrieval configurations and measures their impact on:

- Prompt size
- Response time
- Retrieval quality
- Generated answer quality

---

# 🎯 Objective

Reduce the overall LLM cost while maintaining nearly identical answer quality.

---

# 🛠️ Tech Stack

- Python 3.14
- Google Gemini API
- ChromaDB
- Sentence Transformers
- all-MiniLM-L6-v2
- LangChain Concepts
- Retrieval-Augmented Generation (RAG)

---

# 📂 Project Structure

```
LLM-Cost-Optimization-Study/

│
├── data/
│   └── chroma_db/
│
├── documents/
│
├── src/
│   ├── create_database.py
│   ├── generate_answer.py
│
├── requirements.txt
│
├── .env.example
│
└── README.md
```

---

# ⚙️ Cost Optimisation Strategy

This project optimizes cost by reducing the amount of context sent to the LLM.

### Before

- Top-K Retrieval = 3
- Larger Prompt
- More Tokens
- Higher Cost

### After

- Top-K Retrieval = 2
- Smaller Prompt
- Fewer Tokens
- Lower Cost

---

# 📊 Evaluation

| Metric | Before | After |
|---------|--------|-------|
| Top-K | 3 | 2 |
| Prompt Size | Larger | Smaller |
| Response Time | Higher | Lower |
| Answer Quality | Good | Good |
| Token Usage | Higher | Lower |
| Estimated Cost | Higher | Lower |

---

# 📈 Example Evaluation

### Question

> What is the NITI Frontier Tech Hub?

### Result

| Metric | Value |
|---------|-------|
| Retrieved Chunks | 2 |
| Prompt Size | ~4500 characters |
| Response Time | ~7 seconds |
| Answer Quality | Accurate |
| Cost Reduction | Improved |

---

# 🔍 Retrieval Pipeline

```
User Question
      │
      ▼
Sentence Transformer Embedding
      │
      ▼
ChromaDB Similarity Search
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Prompt Construction
      │
      ▼
Google Gemini
      │
      ▼
Final Answer
```

---

# 🚀 Features

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- ChromaDB Vector Database
- Google Gemini Integration
- SentenceTransformer Embeddings
- Response Time Measurement
- Prompt Size Analysis
- Cost Optimization Study

---

# 📊 Performance Metrics

Example Output

```
Top-K: 2

Prompt Size:
4511 characters

Response Time:
6.92 seconds

Retrieved Documents:
2

Answer Quality:
Good
```

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/LLM-Cost-Optimization-Study.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create environment file

```
GEMINI_API_KEY=YOUR_API_KEY
```

Run the project

```bash
python src/generate_answer.py
```

---

# 💡 Key Learnings

- Optimizing retrieval can reduce LLM cost.
- Smaller prompts reduce token consumption.
- Lower Top-K improves efficiency.
- Careful tuning maintains answer quality while improving performance.

---

# 📷 Sample Output

```
Question:
What is the NITI Frontier Tech Hub?

Retrieved Chunks:
2

Prompt Size:
4511 characters

Response Time:
6.92 seconds

Answer:
The NITI Frontier Tech Hub is an initiative focused on frontier technologies such as AI, bioengineering, renewable energy, and quantum computing. It supports technology-led growth through research, events, and quarterly insights.
```

---

# 🔮 Future Improvements

- Dynamic Top-K Selection
- Hybrid Search
- Query Expansion
- Response Caching
- Token Usage Dashboard
- Automatic Evaluation Metrics

---

# 👨‍💻 Author

**Srivasan S**

B.Tech Computer Science and Engineering

Aspiring AI Software Engineer

GitHub: https://github.com/srivasan16

---

# ⭐ Internship Project

This project was developed as part of the **AgenticX AI Internship** to demonstrate practical techniques for reducing LLM inference cost while maintaining response quality.

---