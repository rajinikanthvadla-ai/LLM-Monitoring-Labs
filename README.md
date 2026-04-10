# LLM Monitoring Lab

A complete, local Docker lab for learning **LLM observability** — the same signals and tools used in production AI systems.



1. **Start everything** (first run can take **5–10 minutes** while Langfuse v3, ClickHouse, Redis, and MinIO start):
   ```bash
   docker compose up -d
   ```
2. **Wait for the model** (watch logs until you see `Models ready!`):
   ```bash
   docker compose logs -f ollama-pull
   ```
   If you use **Ollama on the host** (not Docker) on port 11434, skip the above and run: `ollama pull llama3.2:1b`
3. **Confirm Langfuse** opens at [http://localhost:3000](http://localhost:3000) — if it spins, wait; v3 needs Postgres + ClickHouse + worker healthy.
4. **Python** (use **Python 3.10–3.13** if you can; 3.14 works but Langfuse may show a Pydantic warning you can ignore):
   ```bash
   cd apps
   pip install -r requirements.txt
   cp .env.example .env
   python 01_simple_chat.py
   ```
5. **Proof for submission**: open **Langfuse → Traces** and **Phoenix** ([http://localhost:6006](http://localhost:6006)) and show traces from the run.

If anything fails, jump to **[Troubleshooting](#troubleshooting)** below.

---

## What You'll Learn

By the end of this lab you'll understand:
- What signals to collect from LLM applications (and why each matters in prod)
- How Langfuse, Phoenix, Prometheus, and Grafana fit together
- How to trace RAG pipelines, agents, batch jobs, and error flows
- How to do A/B model comparisons before rolling out a new model

---

## Stack Overview

| Tool | Category | What it does |
|---|---|---|
| **Ollama** | Local LLM | Runs `llama3.2:1b` (and others) locally on your machine |
| **Langfuse** | LLM Observability | Traces, scores, evals, prompt mgmt, cost tracking |
| **Phoenix (Arize)** | LLM Tracing | OpenTelemetry-native traces, RAG eval, agent debugging |
| **Prometheus** | Metrics | Scrapes and stores time-series metrics |
| **Grafana** | Dashboards | Visualises Prometheus metrics in real time |
| **Pushgateway** | Metrics bridge | Accepts pushed metrics from batch/script jobs |

---

## Monitoring Concepts — What We Observe and Why

### 1. Latency
```
llm_request_duration_seconds{model, app}  →  p50 / p95 / p99
```
- **p50 (median)** — What a typical user experiences
- **p95** — The "slow 5%". SLAs are usually set here (e.g. p95 < 8s)
- **p99** — The pain of the unlucky 1%. Retries and timeouts live here
- **Where to look**: Grafana "Latency Percentiles" panel, Langfuse trace timeline

### 2. Token Usage = Cost
```
llm_tokens_total{model, token_type="prompt|completion"}
```
- Prompt tokens = what you send (context, system prompt, history)
- Completion tokens = what the model generates
- Cost = (prompt_tokens × price_in) + (completion_tokens × price_out)
- **Watch for**: prompt token creep (system prompts growing over time), context window abuse
- **Where to look**: Grafana "Token Usage" panel, Langfuse → Cost tab

### 3. Error Rate
```
llm_errors_total{model, error_type}   →  timeout | rate_limit | content_filter | model_error
```
- Each error type has a different remediation path:
  - `timeout` → add retry, check model health, scale up
  - `rate_limit` → add backoff, request quota increase
  - `content_filter` → review prompts, add pre-filtering
  - `model_error` → report to provider, add fallback model
- **Where to look**: Grafana "Errors by Type" panel, Langfuse filtered by `level=ERROR`

### 4. RAG Pipeline Stages
```
llm_rag_retrieval_seconds     →  how long vector search takes
llm_rag_generation_seconds    →  how long LLM generation takes
llm_rag_docs_retrieved        →  how many docs came back
```
- Retrieval > 200ms indicates a slow vector DB or bad index
- `docs_retrieved = 0` means the query didn't match anything → hallucination risk
- **Where to look**: Grafana "RAG" row, Langfuse → span tree (retrieval → generation)

### 5. Agent Metrics
```
llm_agent_iterations          →  how many reasoning loops
llm_tool_calls_total          →  which tools are called, success/fail
llm_tool_call_duration_seconds →  per-tool latency
```
- `iterations > 5` often means the agent is confused → cost spike alert
- Tool errors can cause the agent to loop forever
- Total tokens = sum across ALL iterations (compounding cost)
- **Where to look**: Langfuse → trace → span tree with per-iteration breakdown

### 6. User Feedback (Scores)
```python
langfuse.score(trace_id=..., name="user-feedback", value=1.0)  # thumbs up
langfuse.score(trace_id=..., name="user-feedback", value=0.0)  # thumbs down
```
- Ground truth for quality. Correlate with: which model, which prompt version, which user segment
- **Where to look**: Langfuse → Scores tab → filter by score name

---

## Quick Start

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Python **3.10–3.13** recommended (3.14 may show a harmless Langfuse/Pydantic warning)
- **8 GB RAM minimum** (16 GB recommended); Langfuse v3 adds ClickHouse, Redis, and MinIO

### Step 1 — Start the stack (from repo root)
```bash
cd /path/to/LLM-Monitoring-Labs
docker compose up -d
```
First boot: allow **several minutes** for Langfuse v3, worker, and dependencies to become ready.

### Step 2 — Wait for model download (~2 min for llama3.2:1b)
```bash
docker compose logs -f ollama-pull
# Wait until you see: "Models ready!"
```
If that container never appears, pull manually **from repo root** (service name `ollama`, not a guessed container name):
```bash
docker compose exec ollama ollama pull llama3.2:1b
```

### Step 3 — Install Python dependencies
```bash
cd apps
pip install -r requirements.txt
```

### Step 4 — Copy env file
```bash
cp .env.example .env
# Defaults are fine; timeouts are pre-set for large LLM spans (see LANGFUSE_TIMEOUT in .env.example)
```

### Step 5 — Open dashboards

| Dashboard | URL | Credentials |
|---|---|---|
| Langfuse | http://localhost:3000 | admin@lab.local / admin123 |
| Phoenix | http://localhost:6006 | (no login) |
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | (no login) |
| Pushgateway | http://localhost:9091 | (no login) |

### Step 6 — Run the labs

Each lab is a standalone Python script. Run them in order:

```bash
cd apps

# Lab 1: Simple chat with latency + token tracking
python 01_simple_chat.py

# Lab 2: RAG pipeline — per-stage latency (retrieval vs generation)
python 02_rag_pipeline.py

# Lab 3: Agent with tool calls — iteration count, tool latency
python 03_agent_with_tools.py

# Lab 4: Batch processor — pushgateway metrics, throughput
python 04_batch_processor.py

# Lab 5: Error simulation — retry patterns, circuit breaker concept
python 05_error_simulation.py

# Lab 6: A/B model comparison — latency, tokens, cost side by side
python 06_ab_comparison.py
```

The **load generator** runs automatically inside Docker (`metrics-app` service) and keeps Grafana populated with live data. You can also run it locally:
```bash
python load_generator.py
```

---

## Pull Additional Models (Optional)

Run **from the repository root** so `docker compose` finds `docker-compose.yml`:

```bash
# Faster model (compose usually pulls this via ollama-pull)
docker compose exec ollama ollama pull llama3.2:1b

# Better quality (requires ~4GB)
docker compose exec ollama ollama pull mistral:7b

# List what's installed
docker compose exec ollama ollama list
```

Then update `SMART_MODEL=mistral:7b` in your `.env` to use it in Lab 6.

---

## What Each Tool Covers

### Langfuse (http://localhost:3000)
Navigate to:
- **Traces** — every LLM call, with input/output and span tree
- **Sessions** — group traces into a conversation or batch run
- **Scores** — user feedback, eval scores attached to traces
- **Users** — per-user usage and cost
- **Prompts** — version-controlled prompt management
- **Dashboard** — token cost, request volume, error rate overview

### Phoenix (http://localhost:6006)
Navigate to:
- **Traces** — OpenTelemetry spans, auto-captured via OpenInference
- **Datasets** — replay traces as eval datasets
- **Experiments** — run evals against a dataset with different models/prompts

### Grafana (http://localhost:3001)
Dashboard: `LLM → LLM Monitoring Lab`

Panels:
- **Overview row** — total requests, error %, avg latency, total tokens, active sessions
- **Request Rate** — req/s broken down by app (chat, rag, agent)
- **Latency Percentiles** — p50/p95/p99 over time
- **Token Usage** — prompt vs completion tokens/s
- **Latency by Model** — compare models side by side
- **Errors by Type** — which errors are most common
- **RAG panel** — retrieval vs generation latency split
- **Retry Attempts** — retry pressure over time

---

## Project Structure

```
LLM-Monitoring-Labs/
├── docker-compose.yml          ← All services
├── .env.example                ← Config template
├── prometheus/
│   └── prometheus.yml          ← Scrape config
├── grafana/
│   ├── provisioning/           ← Auto-loads datasource + dashboard
│   └── dashboards/
│       └── llm-monitoring.json ← Pre-built dashboard
└── apps/
    ├── requirements.txt
    ├── config.py               ← Shared config (reads .env)
    ├── tracing.py              ← Langfuse + Phoenix setup
    ├── metrics.py              ← All Prometheus metric definitions
    ├── 01_simple_chat.py       ← Lab 1
    ├── 02_rag_pipeline.py      ← Lab 2
    ├── 03_agent_with_tools.py  ← Lab 3
    ├── 04_batch_processor.py   ← Lab 4
    ├── 05_error_simulation.py  ← Lab 5
    ├── 06_ab_comparison.py     ← Lab 6
    ├── load_generator.py       ← Continuous traffic + metrics server
    └── Dockerfile              ← For metrics-app service
```

---

## Stopping the Lab

```bash
docker compose down          # Stop containers (keeps data)
docker compose down -v       # Stop + delete all volumes (clean reset)
```

---

## Troubleshooting

| Symptom | What to do |
|--------|------------|
| `model 'llama3.2:1b' not found` | **Docker:** from repo root, `docker compose exec ollama ollama pull llama3.2:1b`. **Host Ollama:** `ollama pull llama3.2:1b`. Ensure `OLLAMA_BASE_URL` in `apps/.env` matches where Ollama listens (`http://localhost:11434/v1`). |
| `No such container` when using `docker exec ...` | Stack not running or wrong name. Use **`docker compose exec ollama ...`** from the **repo root**, or run `docker compose up -d`. |
| `Failed to export span batch due to timeout` | Usually **Langfuse OTLP** under heavy spans. Defaults are now **120s** (`LANGFUSE_TIMEOUT`, `OTEL_BSP_EXPORT_TIMEOUT_MS`). Increase in `apps/.env` if needed. |
| Phoenix `UNAVAILABLE` / export errors on port **4317** | Start Phoenix: `docker compose up -d phoenix`. Open [http://localhost:6006](http://localhost:6006). |
| Langfuse **404** on traces (older setups) | This repo uses **Langfuse v3** in Docker (OTLP route required by Python SDK v4). Run `docker compose up -d` with the current `docker-compose.yml`. |
| `cd apps` fails | You are not in the repo root. `cd` to `LLM-Monitoring-Labs` first, then `cd apps`. |
