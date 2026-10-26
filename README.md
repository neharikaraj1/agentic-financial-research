# 🤖 Agentic AI Financial Research & Document Analysis System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2.28-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-0.2.16-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-0.5.15-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  A production-grade <strong>Agentic AI system</strong> that decomposes complex financial queries into multi-step reasoning tasks and dynamically routes them across retrieval, calculation, and analysis tools.
</p>

---

## 🌟 Why This Project?

| Feature | Detail |
|---------|--------|
| 🧠 **Multi-node LangGraph Agent** | 6-node stateful graph with conditional retry loops — not a simple chain |
| 🔍 **RAG + Live Data** | Semantic search over uploaded financials **plus** real-time stock data via yfinance |
| 🛡️ **Anti-hallucination Loop** | Validator node checks every answer; loops back to Reasoner if grounding fails |
| 📊 **RAGAs Evaluation** | Faithfulness, relevancy, precision & recall scoring built-in |
| 🐳 **Production-ready** | Dockerized, health checks, structured logging, pytest test suite |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Graph                     │
│                                                              │
│  [1] Query Planner ──► [2] Retriever (RAG / ChromaDB)       │
│                               │                             │
│                        [3] Tool Executor                     │
│                    ┌──────────┴──────────┐                  │
│              Calculator            Stock Data API           │
│                    └──────────┬──────────┘                  │
│                        [4] Reasoner                         │
│                               │                             │
│                        [5] Validator ──(fail)──► [4]        │
│                               │                             │
│                      [6] Response Generator                  │
└───────────────────────────────┬─────────────────────────────┘
                                │
                         Final Answer
                   (grounded, cited, formatted)
```

### Node Responsibilities

| Node | Role |
|------|------|
| **Query Planner** | Decomposes the query into a structured 3–6 step research plan |
| **Retriever** | Runs semantic similarity search against ChromaDB (RAG) |
| **Tool Executor** | Calls LangChain tools: financial calculator, live stock data |
| **Reasoner** | Synthesises all evidence into a chain-of-thought draft answer |
| **Validator** | Checks factual grounding; loops back if answer is unsupported |
| **Response Generator** | Formats the final markdown response with source citations |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agentic Orchestration | **LangGraph** |
| LLM / Embeddings | **LangChain + OpenAI GPT-4o** |
| Vector Database | **ChromaDB** (persisted to disk) |
| Backend API | **FastAPI** + Uvicorn |
| Containerisation | **Docker** + Docker Compose |
| Evaluation | **RAGAs** (faithfulness, relevancy, precision, recall) |
| Financial Data | yfinance (live stock prices & fundamentals) |

---

## Project Structure

```
agentic-financial-research/
├── app/
│   ├── main.py            # FastAPI application & lifespan
│   ├── config.py          # Pydantic settings (reads .env)
│   ├── agent.py           # LangGraph graph definition (6 nodes)
│   ├── tools.py           # LangChain tool definitions
│   ├── vector_store.py    # ChromaDB manager (add/search/delete)
│   ├── ingestion.py       # Document loading & chunking pipeline
│   ├── evaluation.py      # RAGAs evaluation runner
│   ├── schemas.py         # Pydantic request/response models
│   └── routers/
│       ├── documents.py   # /api/documents endpoints
│       ├── query.py       # /api/query endpoints
│       └── evaluation.py  # /api/evaluate endpoint
├── tests/
│   ├── test_tools.py      # Financial calculator unit tests
│   ├── test_ingestion.py  # Ingestion pipeline tests
│   └── test_api.py        # FastAPI endpoint integration tests
├── data/                  # Auto-created: chroma_db/, uploads/
├── logs/                  # Auto-created: app.log
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Option 1: Local (Python virtualenv)

```bash
# 1. Clone and enter the project
git clone https://github.com/neharikaraj1/agentic-financial-research.git
cd agentic-financial-research

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

> Open **http://localhost:8000/docs** for the interactive API docs.

---

### Option 2: Docker Compose

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 2. Build and start
docker-compose up --build

# 3. Check health
curl http://localhost:8000/health
```

---

## API Endpoints

### 📄 Document Ingestion
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload PDF, DOCX, TXT, XLSX |
| `POST` | `/api/documents/text` | Ingest raw text |
| `GET` | `/api/documents/` | List all ingested sources |
| `DELETE` | `/api/documents/{source}` | Remove a document |

### 🤖 Agent Query
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Full 6-node agentic pipeline |
| `POST` | `/api/query/quick` | Fast RAG-only retrieval |

### 📊 RAGAs Evaluation
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/evaluate` | Compute faithfulness, relevancy, precision, recall |

### 🔧 System
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/docs` | Swagger UI |

---

## Example Usage

### 1. Ingest a financial document
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@apple_10k_2024.pdf"
```

### 2. Run a financial research query
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Apple'\''s revenue growth trend and how does its P/E ratio compare to the sector?"}'
```

### 3. Evaluate RAG quality
```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
      "question": "What was Apple revenue in 2024?",
      "answer": "Apple reported $391B in revenue for FY2024.",
      "contexts": ["Apple Inc. reported total net sales of $391.0 billion for fiscal year 2024."],
      "ground_truth": "Apple revenue was approximately $391 billion in fiscal 2024."
    }]
  }'
```

---

## Available LangChain Tools

| Tool | Description |
|------|-------------|
| `rag_retrieval_tool` | Semantic search over ingested financial documents |
| `financial_calculator_tool` | P/E, ROE, margins, CAGR, D/E, EPS, current ratio |
| `stock_data_tool` | Live price, market cap, fundamentals via yfinance |
| `document_summary_tool` | Distil retrieved context into bullet summaries |

---

## RAGAs Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Are answer claims supported by retrieved context? |
| **Answer Relevancy** | Is the answer on-topic for the question? |
| **Context Precision** | Are the retrieved chunks actually relevant? |
| **Context Recall** | Does the retrieved context cover all needed facts? |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | LLM model name |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Document chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (chars) |
| `MAX_ITERATIONS` | `10` | Agent retry limit |
| `RAGAS_EVAL_ENABLED` | `true` | Enable RAGAs evaluation |

---

## License

MIT — feel free to use, modify, and distribute.
