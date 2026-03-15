# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

Always use `uv` for all package management and running Python commands. Never use `pip` directly.

- Install dependencies: `uv sync`
- Add a dependency: `uv add <package>`
- Remove a dependency: `uv remove <package>`
- Run Python commands: `uv run <command>`

## Commands

**Setup:**
```bash
uv sync                          # Install dependencies
cp .env.example .env             # Then add your ANTHROPIC_API_KEY
```

**Run the server:**
```bash
./run.sh                         # Quick start
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The server must be started from the `backend/` directory (or via `run.sh`) because `config.py` sets `CHROMA_PATH = "./chroma_db"` relative to the working directory, and `app.py` loads docs from `"../docs"` and serves frontend from `"../frontend"`.

## Architecture

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot. The backend is a FastAPI server; the frontend is plain HTML/CSS/JS served as static files from the same server.

**Request flow:**
1. User submits a question via the frontend → `POST /api/query`
2. `app.py` delegates to `RAGSystem.query()`
3. `RAGSystem` builds a prompt and calls `AIGenerator.generate_response()` with Claude and a `search_course_content` tool
4. Claude decides whether to invoke the tool; if so, `ToolManager` routes the call to `CourseSearchTool`
5. `CourseSearchTool` calls `VectorStore.search()` which queries ChromaDB using sentence-transformer embeddings
6. Tool results are sent back to Claude for a final synthesized response
7. The exchange is stored in `SessionManager` (in-memory, keyed by `session_id`)

**Key components:**

| File | Responsibility |
|---|---|
| `backend/app.py` | FastAPI routes, startup document loading, static file serving |
| `backend/rag_system.py` | Orchestrator — wires all components together |
| `backend/ai_generator.py` | Anthropic API calls, tool execution loop |
| `backend/vector_store.py` | ChromaDB wrapper — two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `backend/document_processor.py` | Parses `.txt` course files into `Course`/`Lesson`/`CourseChunk` objects and chunks text |
| `backend/search_tools.py` | `Tool` ABC, `CourseSearchTool`, and `ToolManager` for Anthropic tool-calling integration |
| `backend/session_manager.py` | In-memory conversation history (resets on server restart) |
| `backend/models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |
| `backend/config.py` | All tuneable settings (model, chunk size, embedding model, etc.) |

**Course document format** (files in `docs/`):
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <title>
...
```

Documents are loaded at startup; existing courses (matched by title) are skipped. ChromaDB persists to `backend/chroma_db/`. To force a re-index, delete that directory or call `VectorStore.clear_all_data()`.

**Configuration** (`backend/config.py`):
- Model: `claude-sonnet-4-20250514`
- Embedding: `all-MiniLM-L6-v2` (via sentence-transformers)
- Chunk size: 800 chars, overlap: 100 chars
- Max search results: 5, max conversation history: 2 exchanges

**API endpoints:**
- `POST /api/query` — accepts `{query, session_id?}`, returns `{answer, sources, session_id}`
- `GET /api/courses` — returns course count and titles
- `GET /` — serves `frontend/index.html`
