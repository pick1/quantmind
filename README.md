# QuantMind Local

A **local** financial knowledge extraction and retrieval system that ingests academic papers, news, and financial documents, structures them into a queryable knowledge base, and surfaces insights through a Streamlit UI — running entirely on local hardware.

## Architecture

```
                    ┌──────────────────┐
                    │   Streamlit UI    │
                    │  (Query / Library │
                    │   Ingest / Set.)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │     Retrieval Pipeline       │
              │  Embed → Search → Rerank →   │
              │          Synthesize          │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────┴─────┐       ┌────┴─────┐       ┌─────┴────┐
    │  Ollama  │       │ ChromaDB │       │  SQLite   │
    │ (LLM +   │       │ (vectors)│       │(metadata) │
    │  embed)  │       │ 768-dims │       │           │
    └──────────┘       └──────────┘       └──────────┘
         │
    ┌────┴─────┐
    │ OpenRouter│
    │ (fallback)│
    └──────────┘
```

## Prerequisites

- **Python 3.11+**
- **Ollama** (primary inference backend) — [install](https://ollama.com/download)
- Pull required models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen3:14b
  ```

## Quick Start

```bash
# 1. Clone and enter the project
cd quantmind

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local Ollama)

# 5. Ingest some papers
python -m quantmind.ingest --arxiv-query "financial LLM RAG" --max 5

# 6. Launch the UI
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Usage

### 📥 Ingest Documents

Four ways to add documents:

```bash
# Single arXiv paper by ID
python -m quantmind.ingest --arxiv 2309.01234

# Bulk keyword search (top-5 by relevance)
python -m quantmind.ingest --arxiv-query "factor momentum risk" --max 10

# Local PDF file
python -m quantmind.ingest --pdf /path/to/paper.pdf

# Plain text
python -m quantmind.ingest --text "Your financial text here..."

# URL (auto-detects arXiv, PDF, or article page)
python -m quantmind.ingest --url https://arxiv.org/abs/2309.01234
python -m quantmind.ingest --url https://example.com/paper.pdf
```

Or use the UI: navigate to **Ingest** → choose arXiv / drag & drop PDF or text / paste link / plain text.

### 🔍 Query

Ask questions in plain English. The system:
1. Embeds your query via Ollama (nomic-embed-text)
2. Retrieves top-20 chunks from ChromaDB (cosine similarity)
3. Reranks to top-5 using a cross-encoder (ms-marco-MiniLM-L-6-v2)
4. Sends to the LLM with a strict citation prompt
5. Displays the answer with source citations

### 📚 Library

Browse all ingested documents. Filter by source type or domain tag. Paginated at 20 per page.

### ⚙️ Settings

Check provider status (Ollama / OpenRouter / unavailable), storage stats, and configuration values.

## Switching to Xavier

When the Jetson Xavier AGX arrives:

1. Install Ollama on the Xavier (JetPack / Linux)
2. Pull models on Xavier:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3:30b   # or deepseek-r1:32b using Xavier's 36GB
   ```
3. Update `.env` on the workhorse:
   ```bash
   OLLAMA_HOST=http://192.168.1.XXX:11434   # Xavier's static LAN IP
   OLLAMA_MODEL=qwen3:30b
   ```
4. **No code changes needed.** The LLM client reads `OLLAMA_HOST` at startup.

> **Note on embeddings:** nomic-embed-text produces identical 768-dim vectors regardless of host (workhorse vs. Xavier), so the ChromaDB collection stays valid. If you switch the embedding model, you must rebuild ChromaDB (delete `./data/chroma` and re-ingest).

## Project Structure

```
quantmind/
├── app.py                        # Streamlit entrypoint
├── pages/
│   ├── 1_Query.py                # Search + citations UI
│   ├── 2_Library.py              # Document browser
│   ├── 3_Ingest.py               # Ingestion forms (arXiv, files, link, text)
│   └── 4_Settings.py             # Provider + storage info
├── quantmind/
│   ├── __init__.py
│   ├── config.py                 # Env vars, constants
│   ├── llm_client.py             # Two-tier LLM (Ollama → OpenRouter)
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── __main__.py           # CLI entrypoint (--arxiv, --pdf, --text, --url)
│   │   ├── arxiv.py              # arXiv API fetch + parse
│   │   ├── pdf.py                # PDF extraction + chunking
│   │   ├── pipeline.py           # Orchestrates ingest flow (incl. URL auto-detect)
│   │   └── tagger.py             # LLM domain tagging
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py           # Query embedding
│   │   ├── searcher.py           # ChromaDB search
│   │   ├── reranker.py           # Cross-encoder reranking
│   │   ├── synthesizer.py        # LLM answer + citations
│   │   └── pipeline.py           # End-to-end retrieval
│   └── store/
│       ├── __init__.py
│       ├── chroma.py             # ChromaDB wrapper (768-dim)
│       └── sqlite.py             # SQLite document metadata
├── data/
│   ├── chroma/                   # ChromaDB persist (gitignored)
│   └── quantmind.db              # SQLite db (gitignored)
├── tests/
│   ├── test_llm_client.py
│   ├── test_ingest.py
│   └── test_retrieval.py
├── .env.example
├── requirements.txt
├── README.md
└── Dockerfile
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server (workhorse or Xavier LAN) |
| `OLLAMA_MODEL` | `qwen3:14b` | Completion model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (must output 768-dim) |
| `OPENROUTER_API_KEY` | — | OpenRouter key for fallback |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat-v3-5` | Fallback model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store path |
| `SQLITE_PATH` | `./data/quantmind.db` | Metadata store path |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Tech Stack

- **Python 3.11+** — language
- **Streamlit** — UI framework
- **ChromaDB** — local persistent vector store (768-dim, cosine similarity)
- **SQLite** — document metadata store
- **Ollama** — primary LLM + embedding backend
- **OpenRouter** — fallback LLM API (DeepSeek V3.5)
- **pdfplumber** — PDF text extraction
- **sentence-transformers** — cross-encoder reranking + CPU embedding fallback
- **httpx** — async HTTP (arXiv API)
- **ollama Python SDK** — Ollama integration
- **openai Python SDK** — OpenRouter integration

## License

MIT — built as a portfolio project.
