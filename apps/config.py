"""
Shared configuration — reads from .env or environment variables.
All apps import from here so settings stay in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=False)  # don't override already-set env vars (useful in Docker)


def _normalize_phoenix_http_endpoint(url: str) -> str:
    """Ensure OTLP HTTP exporter posts to .../v1/traces (required when endpoint= is explicit)."""
    u = url.strip().rstrip("/")
    return u if u.endswith("/v1/traces") else f"{u}/v1/traces"

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Models — llama3.2:1b is fast for the lab; swap for bigger models when needed
DEFAULT_MODEL          = os.getenv("DEFAULT_MODEL", "llama3.2:1b")
FAST_MODEL             = os.getenv("FAST_MODEL",    "llama3.2:1b")
SMART_MODEL            = os.getenv("SMART_MODEL",   "llama3.2:1b")  # swap to mistral:7b if pulled

# ── Langfuse ─────────────────────────────────────────────────────────────────
LANGFUSE_HOST          = os.getenv("LANGFUSE_HOST",       "http://localhost:3000")
LANGFUSE_PUBLIC_KEY    = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-labpublickey1234")
LANGFUSE_SECRET_KEY    = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-labsecretkey1234")
# SDK v4 sends OTLP to Langfuse; default 5s is too low for large LLM span payloads.
LANGFUSE_TIMEOUT       = int(os.getenv("LANGFUSE_TIMEOUT", "120"))

# ── Phoenix (Arize) ───────────────────────────────────────────────────────────
# Default: OTLP gRPC on 4317 (docker-compose exposes it; avoids HTTP 404 if UI port is wrong).
# Set PHOENIX_OTLP_PROTOCOL=http to use HTTP on port 6006 instead.
_phoenix_proto = os.getenv("PHOENIX_OTLP_PROTOCOL", "grpc").strip().lower()
PHOENIX_OTLP_PROTOCOL = "http" if _phoenix_proto == "http" else "grpc"
PHOENIX_OTLP_GRPC_ENDPOINT = os.getenv(
    "PHOENIX_OTLP_GRPC_ENDPOINT", "http://localhost:4317"
)
PHOENIX_COLLECTOR_ENDPOINT = _normalize_phoenix_http_endpoint(
    os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
)
PHOENIX_OTLP_TIMEOUT_SEC = float(os.getenv("PHOENIX_OTLP_TIMEOUT_SEC", "120"))
OTEL_BSP_EXPORT_TIMEOUT_MS = int(os.getenv("OTEL_BSP_EXPORT_TIMEOUT_MS", "120000"))

# ── Prometheus Pushgateway ────────────────────────────────────────────────────
PROMETHEUS_PUSHGATEWAY = os.getenv("PROMETHEUS_PUSHGATEWAY", "localhost:9091")

# ── Metrics HTTP server port (for load_generator.py) ─────────────────────────
METRICS_PORT           = int(os.getenv("METRICS_PORT", "8000"))
