# Ollama on macOS (host, outside Docker)

Native Ollama on Apple Silicon uses Metal GPU. Docker Desktop on macOS cannot pass through the GPU.

Official docs: [macOS](https://docs.ollama.com/macos) · [GPU](https://docs.ollama.com/gpu) · [FAQ](https://docs.ollama.com/faq)

## How it works

- Install adds `Ollama.app` (the server) and the `ollama` CLI. No models are included.
- The server runs in the background on `http://127.0.0.1:11434` and starts at login.
- `ollama pull` adds a model to the local catalog (`~/.ollama/models`). Disk only, nothing is loaded.
- Each request names a model. The server loads it into memory on the first request and unloads it after 5 minutes without requests.
- An idle server with no model loaded uses no GPU, ~0% CPU and ~175 MB of RAM.

## Install

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Pull a model

```bash
ollama pull llama3.2:1b
ollama list               # models in the catalog
ollama rm llama3.2:1b     # remove from disk
```

## Server

Start (if not running):

```bash
open -a Ollama
```

Stop: menu bar → Quit, or:

```bash
pkill -f '/Applications/Ollama.app'
```

To disable auto-start: System Settings → General → Login Items → **Ollama**.

## Loaded models

```bash
ollama ps                 # loaded models; Processor should be 100% GPU
ollama stop llama3.2:1b   # unload now
```

## Optional: chat in the terminal

```bash
ollama run llama3.2:1b
```

## Point this project at host Ollama

Set in `.env` (Compose default is the Docker value):

```bash
# make run (API in Docker → host Ollama)
LLM_BASE_URL=http://host.docker.internal:11434/v1

# make api-local (API on host)
# LLM_BASE_URL=http://localhost:11434/v1
```

Set matching `LLM_MODEL` (or `make pull-llm`).
