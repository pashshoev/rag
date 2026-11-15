COMPOSE := docker-compose.yml
COMPOSE_FULL := docker-compose-full-stack.yml

.PHONY: run stop build run-full stop-full pull-llm api-local ui-local

# Default: API + UI + Qdrant in Docker; LLM on host Ollama (GPU on Mac)
run:
	docker compose -f $(COMPOSE) up -d --build

stop:
	docker compose -f $(COMPOSE) down

build:
	docker compose -f $(COMPOSE) build

# All-in-Docker demo (includes Ollama — CPU on Mac Docker Desktop)
run-full:
	docker compose -f $(COMPOSE_FULL) up -d --build

stop-full:
	docker compose -f $(COMPOSE_FULL) down

# Host Ollama: pull / switch model used by default stack
pull-llm:
	ollama pull $${LLM_MODEL:-llama3.2:1b}

# Host process with reload (needs local .venv + services already reachable)
api-local:
	uv run uvicorn rag.api.app:app --reload --host 0.0.0.0 --port 8000

ui-local:
	uv sync --group ui
	API_BASE_URL=$${API_BASE_URL:-http://localhost:8000} uv run --group ui streamlit run src/rag/ui/app.py
