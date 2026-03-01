# RAG Chatbot

A full-stack Retrieval-Augmented Generation chatbot with:
- Flask backend (`/api`)
- LangChain-based document and chat services
- React frontend (single-file SPA, no build step)
- Dockerized local setup

## Project Structure

```text
rag-chatbot/
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- routes.py
|   |   |-- document_service.py
|   |   `-- chat_service.py
|   |-- tests/
|   |   `-- test_app.py
|   |-- run.py
|   |-- Dockerfile
|   |-- pytest.ini
|   `-- .env.example
|-- frontend/
|   |-- index.html
|   |-- Dockerfile
|   `-- entrypoint.sh
|-- docker-compose.yml
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Run Locally (without Docker)

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Copy env template:
   ```bash
   copy backend\.env.example backend\.env
   ```
4. Start backend:
   ```bash
   cd backend
   python run.py
   ```
5. Open `frontend/index.html` in a browser (or serve it with a static server).

## Run with Docker

1. Copy env template:
   ```bash
   copy backend\.env.example backend\.env
   ```
2. Start stack:
   ```bash
   docker compose up --build
   ```
3. Open frontend: `http://localhost:3000`
4. Backend API: `http://localhost:5000/api`

## Run Tests

```bash
cd backend
pytest
```

Coverage requires `pytest-cov`:

```bash
python -m pip install pytest-cov
pytest --cov=app --cov-report=html
```