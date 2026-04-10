"""
Shared Prometheus metrics definitions.
All apps import from here — single source of truth for metric names.

In production these metrics would tell you:
  - Is the system healthy? (error rate, latency)
  - Is it expensive? (token counts, cost)
  - Where is it slow? (per-stage latency in RAG/agent)
  - Is it retrying too much? (retry counters)
"""
from prometheus_client import Counter, Histogram, Gauge, REGISTRY

# ── Request-level metrics ─────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM API calls",
    ["model", "app", "status"],          # status: success | error
)

REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "End-to-end LLM call duration (seconds)",
    ["model", "app"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
)

# ── Token metrics ─────────────────────────────────────────────────────────────

TOKEN_USAGE = Counter(
    "llm_tokens_total",
    "Tokens consumed",
    ["model", "token_type"],              # token_type: prompt | completion
)

# ── Error metrics ─────────────────────────────────────────────────────────────

ERROR_COUNT = Counter(
    "llm_errors_total",
    "LLM call failures",
    ["model", "error_type"],              # timeout | rate_limit | model_error | unknown
)

RETRY_COUNT = Counter(
    "llm_retries_total",
    "Retry attempts",
    ["model", "reason"],
)

# ── RAG-specific metrics ──────────────────────────────────────────────────────

RAG_RETRIEVAL_LATENCY = Histogram(
    "llm_rag_retrieval_seconds",
    "Time to retrieve documents from the vector store",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RAG_GENERATION_LATENCY = Histogram(
    "llm_rag_generation_seconds",
    "Time to generate answer given retrieved context",
    buckets=[0.5, 1, 2, 5, 10, 20, 30],
)

RAG_DOCS_RETRIEVED = Histogram(
    "llm_rag_docs_retrieved",
    "Number of documents retrieved per query",
    buckets=[1, 2, 3, 5, 10],
)

# ── Agent-specific metrics ────────────────────────────────────────────────────

AGENT_ITERATIONS = Histogram(
    "llm_agent_iterations",
    "Number of reasoning iterations per agent run",
    buckets=[1, 2, 3, 5, 8, 13],
)

TOOL_CALL_COUNT = Counter(
    "llm_tool_calls_total",
    "Tool invocations by agent",
    ["tool_name", "status"],
)

TOOL_CALL_LATENCY = Histogram(
    "llm_tool_call_duration_seconds",
    "Time to execute a tool call",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
)

# ── Session / concurrency metrics ─────────────────────────────────────────────

ACTIVE_SESSIONS = Gauge(
    "llm_active_sessions",
    "Currently in-flight LLM requests",
)
