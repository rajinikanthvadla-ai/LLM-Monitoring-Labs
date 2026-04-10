"""
Lab 1 — Simple Chat with Full Observability
============================================

WHAT WE MONITOR (and why it matters in production):

  Latency          — Users notice > 3s. SLAs typically set at p95 < 5s.
  Token counts     — Direct cost driver. prompt tokens * price + completion tokens * price.
  Model name       — Did the right model serve the request? Useful after model upgrades.
  Conversation ID  — Group related turns so you can replay full sessions in Langfuse.
  User feedback    — 👍/👎 scores tied to traces = ground truth for quality monitoring.
  Error type       — timeout vs model_unavailable vs content_filter → different remediation.

HOW TO RUN:
  1. From repo root: docker compose up -d  (starts Ollama + pulls llama3.2:1b via ollama-pull)
  2. cd apps && pip install -r requirements.txt
  3. python 01_simple_chat.py
  If "model not found": run ollama pull (local) OR from repo root docker compose exec ollama ollama pull
  If Phoenix export errors: from repo root, docker compose up -d phoenix

DASHBOARDS:
  Langfuse   → http://localhost:3000  (admin@lab.local / admin123)
  Phoenix    → http://localhost:6006
  Grafana    → http://localhost:3001  (admin / admin)
"""
import time
import uuid
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

import config
from tracing import setup_tracing, propagate_attributes
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, TOKEN_USAGE,
    ERROR_COUNT, ACTIVE_SESSIONS,
)

console = Console()
langfuse = setup_tracing("simple-chat")
ollama = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")


def _hint_for_chat_error(exc: BaseException, model: str) -> str:
    msg = str(exc).lower()
    if "not_found" in msg or "not found" in msg:
        if "model" in msg:
            return (
                f"\n[yellow]Fix:[/yellow] No model [cyan]{model}[/cyan] at "
                f"[cyan]{config.OLLAMA_BASE_URL}[/cyan]\n"
                "  [bold]A) Ollama on this machine (Docker not used):[/bold]\n"
                f"     ollama pull {model}\n"
                "  [bold]B) Ollama via this repo’s Docker Compose:[/bold] from the [cyan]repo root[/cyan] (not apps/):\n"
                "     docker compose up -d ollama ollama-pull\n"
                f"     docker compose exec ollama ollama pull {model}\n"
                "     ([dim]Use the service name [cyan]ollama[/cyan], not a container name; "
                '"No such container" means the stack is not running.)[/dim]'
            )
    return ""


def chat(user_message: str, session_id: str, model: str = config.DEFAULT_MODEL) -> str:
    """
    Single chat turn with full observability:
      - Langfuse trace (manual SDK)
      - Phoenix trace (auto via OpenInference)
      - Prometheus metrics
    """
    ACTIVE_SESSIONS.inc()
    try:
        with propagate_attributes(
            session_id=session_id,
            trace_name="simple-chat",
            metadata={"model": str(model)},
        ):
            with langfuse.start_as_current_observation(
                name="simple-chat",
                input=user_message,
            ) as root_span:
                llm_span = root_span.start_observation(
                    name="llm-call",
                    input={"message": user_message},
                )
                start = time.time()
                try:
                    response = ollama.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Be concise."},
                            {"role": "user",   "content": user_message},
                        ],
                        max_tokens=256,
                        temperature=0.7,
                    )

                    latency = time.time() - start
                    answer  = response.choices[0].message.content

                    # ── Prometheus ────────────────────────────────────────────
                    REQUEST_COUNT.labels(model=model, app="simple-chat", status="success").inc()
                    REQUEST_LATENCY.labels(model=model, app="simple-chat").observe(latency)
                    TOKEN_USAGE.labels(model=model, token_type="prompt").inc(
                        response.usage.prompt_tokens
                    )
                    TOKEN_USAGE.labels(model=model, token_type="completion").inc(
                        response.usage.completion_tokens
                    )

                    # ── Langfuse ────────────────────────────────────────────────
                    llm_span.update(
                        output=answer,
                        metadata={
                            "latency_s":         round(latency, 3),
                            "prompt_tokens":     response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens":      response.usage.total_tokens,
                        },
                    )
                    root_span.update(output=answer)

                    return answer, latency, response.usage

                except Exception as exc:
                    latency = time.time() - start
                    error_type = type(exc).__name__

                    REQUEST_COUNT.labels(model=model, app="simple-chat", status="error").inc()
                    ERROR_COUNT.labels(model=model, error_type=error_type).inc()
                    llm_span.update(output=str(exc), level="ERROR")
                    root_span.update(output=str(exc), level="ERROR")
                    raise
                finally:
                    llm_span.end()
    finally:
        ACTIVE_SESSIONS.dec()


def add_user_score(trace_id: str, score: float, comment: str = ""):
    """
    Simulate user feedback (thumbs up = 1.0, thumbs down = 0.0).
    In production: call this from your UI's feedback button.
    Visible in Langfuse → Traces → Scores tab.
    """
    langfuse.create_score(
        trace_id=trace_id,
        name="user-feedback",
        value=score,
        comment=comment,
    )


# ── Demo ──────────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "Explain machine learning in one sentence.",
    "Write a haiku about monitoring software.",
    "What is 17 × 23?",
    "Why is observability important for LLM applications?",
]

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Lab 1: Simple Chat[/bold cyan]\n"
        "Watch traces appear in [link=http://localhost:3000]Langfuse[/link] "
        "and [link=http://localhost:6006]Phoenix[/link] in real time.",
        border_style="cyan",
    ))

    session_id = str(uuid.uuid4())
    console.print(f"\n[dim]Session ID: {session_id}[/dim]\n")

    for question in SAMPLE_QUESTIONS:
        console.print(f"[bold yellow]Q:[/bold yellow] {question}")
        try:
            answer, latency, usage = chat(question, session_id)
            console.print(f"[bold green]A:[/bold green] {answer}")
            console.print(
                f"[dim]  latency={latency:.2f}s  "
                f"tokens={usage.prompt_tokens}->{usage.completion_tokens}[/dim]\n"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            hint = _hint_for_chat_error(e, config.DEFAULT_MODEL)
            if hint:
                console.print(hint)
            console.print()

    langfuse.flush()
    console.print("\n[bold]Done! Check Langfuse and Phoenix for traces.[/bold]")
