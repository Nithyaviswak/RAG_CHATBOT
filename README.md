# RAG Chatbot

Full-stack Retrieval-Augmented Generation chatbot with a Flask backend and a static frontend.

## What it does
- Upload and index `PDF`, `TXT`, `DOC`, and `DOCX` files.
- Ask questions over selected documents with citation-style sources.
- Supports vector retrieval and long-context retrieval modes.

## Tech stack
- Backend: Flask, LangChain, FAISS, Gemini, sentence-transformers.
- Frontend: single-file HTML/CSS/JS app (no build step).
- Deployment: backend on Render, frontend on Netlify or Nginx container.

## Project structure
```text
rag-chatbot/
|-- backend/
|   |-- app/
|   |-- tests/
|   |-- .env.example
|   |-- Dockerfile
|   `-- run.py
|-- frontend/
|   |-- index.html
|   |-- entrypoint.sh
|   `-- Dockerfile
|-- docker-compose.yml
|-- netlify.toml
`-- requirements.txt
```

## Frontend API URL behavior
- Default backend origin: `https://rag-chatbot-53e0.onrender.com/`
- Frontend automatically normalizes to an API base ending with `/api`.
- Runtime override (Docker/Nginx): set `API_URL`.
  - Example: `API_URL=http://localhost:5000/api`
  - Example: `API_URL=https://rag-chatbot-53e0.onrender.com/`

## Local development (without Docker)

1. Create and activate a virtual environment.
2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
3. Create backend env file:
```bash
copy backend\.env.example backend\.env
```
4. Update required values in `backend/.env`:
```env
GEMINI_API_KEY=your_key_here
```
5. Start backend:
```bash
cd backend
python run.py
```
6. Open frontend:
```bash
cd frontend
python -m http.server 5500
```
Then browse to `http://localhost:5500`.

## Docker

1. Create env file:
```bash
copy backend\.env.example backend\.env
```
2. Start services:
```bash
docker compose up --build
```
3. Open:
- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:5000/api/health`

## Deploy notes

### Backend (Render)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn run:app --bind 0.0.0.0:$PORT`
- Set env vars from `backend/.env.example`
- Add your frontend domain in `CORS_ORIGINS`

### Frontend (Netlify)
- `netlify.toml` uses `frontend` as base and publishes the folder directly.
- Since this frontend has no build pipeline, default API URL is baked into `index.html`.
- Keep `frontend/index.html` default URL pointed at your deployed backend.

## Important backend environment variables

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API access key | empty |
| `EMBEDDING_PROVIDER` | `auto`, `gemini`, or `local` | `auto` |
| `LLM_MODEL` | Chat model name | `gemini-1.5-flash` |
| `CHAT_RETRIEVAL_MODE` | `vector`, `long_context`, `hybrid` | `vector` |
| `CORS_ORIGINS` | Allowed frontend origins | local dev origins |

## API endpoints
- `GET /api/health`
- `POST /api/upload`
- `GET /api/documents`
- `DELETE /api/document/<doc_id>`
- `POST /api/document/<doc_id>/process`
- `POST /api/chat`
- `POST /api/reset-memory`
- `GET /api/history?session_id=<id>`

## Run tests
```bash
cd backend
pytest
```

## Troubleshooting
- `Cannot reach backend`: confirm backend is running and CORS includes your frontend origin.
- Upload saved but not processed: check `GEMINI_API_KEY` and embedding provider config.
- Empty answers: ensure selected documents are processed and retrieval mode matches your setup.
