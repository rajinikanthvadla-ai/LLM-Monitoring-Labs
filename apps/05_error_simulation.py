"""
Lab 5 — Error Simulation & Retry Patterns
==========================================

REAL-WORLD SCENARIO: Production LLM APIs fail constantly.
  Rate limits hit during traffic spikes, timeouts occur under load,
  content filters block certain prompts, models occasionally return garbage.

WHAT WE MONITOR (error handling is a first-class observability concern):

  Error rate by type  — timeout vs rate_limit vs content_filter → different alerts.
  Retry count         — Too many retries = cascading failure risk.
  Retry success rate  — Is retrying actually helping?
  Circuit breaker     — After N consecutive failures, stop trying (fail fast).
  P99 latency         — Retries inflate tail latency dramatically.

ERRORS SIMULATED:
  timeout          — Request takes too long (15% of the time)
  rate_limit       — Too many requests (10% of the time)
  content_filter   — Prompt blocked by safety layer (5% of the time)
  model_error      — Model returns malformed response (5% of the time)
  success          — 65% happy path

HOW TO RUN:
  python 05_error_simulation.py
"""
import time
import random
import uuid
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from tracing import setup_tracing, propagate_attributes
from metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT, RETRY_COUNT, TOKEN_USAGE

console  = Console()
langfuse = setup_tracing("error-simulation")
ollama   = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")

# ── Fault injection wrapper ───────────────────────────────────────────────────

class RateLimitError(Exception):
    """Simulated 429 Too Many Requests"""

class ContentFilterError(Exception):
    """Simulated content policy violation"""

class ModelError(Exception):
    """Simulated malformed model response"""


def flakey_llm_call(prompt: str, model: str = config.DEFAULT_MODEL):
    """
    Wraps the real LLM call with random fault injection.
    Simulates a production API that isn't perfectly reliable.
    """
    fault = random.choices(
        ["timeout", "rate_limit", "content_filter", "model_error", "success"],
        weights=[15, 10, 5, 5, 65],
    )[0]

    if fault == "timeout":
        # Simulate a slow response that we'd abort in prod
        time.sleep(random.uniform(0.5, 1.5))
        raise TimeoutError("LLM request timed out after 5s")

    if fault == "rate_limit":
        raise RateLimitError("429 Rate limit exceeded. Retry after 2s.")

    if fault == "content_filter":
        raise ContentFilterError("Request blocked by content safety filter.")

    if fault == "model_error":
        raise ModelError("Model returned non-UTF8 response.")

    # Happy path — real LLM call
    return ollama.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )


def call_with_retry(
    prompt: str,
    model: str = config.DEFAULT_MODEL,
    max_retries: int = 3,
) -> dict:
    """
    Exponential backoff retry with different strategies per error type.

    Production patterns:
      timeout      → retry immediately (was just slow)
      rate_limit   → wait before retry (respect the limit)
      content_filter → DO NOT retry (won't change)
      model_error  → retry once (transient)
    """
    session_id = str(uuid.uuid4())

    with propagate_attributes(session_id=session_id, trace_name="error-simulation"):
        with langfuse.start_as_current_observation(
            name="error-simulation",
            input={"prompt": prompt, "max_retries": max_retries},
        ) as root_span:
            attempts = 0
            last_error = None
            t_total_start = time.time()

            for attempt in range(max_retries + 1):
                attempts = attempt + 1
                span = root_span.start_observation(
                    name=f"attempt-{attempts}",
                    input={"prompt": prompt},
                )
                t0 = time.time()

                try:
                    response = flakey_llm_call(prompt, model)
                    latency = time.time() - t0

                    REQUEST_COUNT.labels(model=model, app="error-sim", status="success").inc()
                    REQUEST_LATENCY.labels(model=model, app="error-sim").observe(time.time() - t_total_start)
                    TOKEN_USAGE.labels(model=model, token_type="prompt").inc(response.usage.prompt_tokens)
                    TOKEN_USAGE.labels(model=model, token_type="completion").inc(response.usage.completion_tokens)

                    span.update(
                        output=response.choices[0].message.content,
                        metadata={"latency_s": round(latency, 3), "attempt": attempts},
                    )
                    span.end()
                    root_span.update(
                        output=response.choices[0].message.content,
                        metadata={"attempts": attempts, "final_status": "success"},
                    )

                    return {
                        "answer":   response.choices[0].message.content,
                        "attempts": attempts,
                        "status":   "success",
                        "latency":  time.time() - t_total_start,
                    }

                except ContentFilterError as e:
                    ERROR_COUNT.labels(model=model, error_type="content_filter").inc()
                    span.update(output=str(e), level="ERROR")
                    span.end()
                    root_span.update(
                        output=str(e),
                        level="ERROR",
                        metadata={"final_status": "content_filtered"},
                    )
                    return {"answer": None, "attempts": attempts, "status": "content_filtered", "error": str(e),
                            "latency": time.time() - t_total_start}

                except (TimeoutError, RateLimitError, ModelError) as e:
                    error_type = type(e).__name__.replace("Error", "").lower()
                    last_error = e

                    ERROR_COUNT.labels(model=model, error_type=error_type).inc()
                    span.update(output=str(e), level="WARNING")
                    span.end()

                    if attempt < max_retries:
                        backoff = 0.5 * (2 ** attempt)
                        if isinstance(e, RateLimitError):
                            backoff = max(backoff, 2.0)

                        RETRY_COUNT.labels(model=model, reason=error_type).inc()
                        console.print(
                            f"  [yellow]Attempt {attempts} failed ({error_type}). "
                            f"Retrying in {backoff:.1f}s…[/yellow]"
                        )
                        time.sleep(backoff)
                    else:
                        break

            REQUEST_COUNT.labels(model=model, app="error-sim", status="error").inc()
            REQUEST_LATENCY.labels(model=model, app="error-sim").observe(time.time() - t_total_start)
            root_span.update(
                output=str(last_error),
                level="ERROR",
                metadata={"attempts": attempts, "final_status": "failed"},
            )
            return {"answer": None, "attempts": attempts, "status": "failed", "error": str(last_error),
                    "latency": time.time() - t_total_start}


# ── Demo ──────────────────────────────────────────────────────────────────────

PROMPTS = [
    "What is 2 + 2?",
    "Name three planets in our solar system.",
    "What color is the sky?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "Name the tallest mountain on Earth.",
    "What is a CPU?",
    "How many days in a year?",
    "What language is spoken in Brazil?",
    "Describe photosynthesis in one sentence.",
]

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Lab 5: Error Simulation[/bold cyan]\n"
        "Each call has a ~35% chance of failure with random error type.\n"
        "Watch retries, error rates, and tail latency in Grafana + Langfuse.",
        border_style="cyan",
    ))

    stats = {"success": 0, "content_filtered": 0, "failed": 0, "total_attempts": 0}

    table = Table(title="Results", show_header=True)
    table.add_column("Prompt", style="dim", max_width=35)
    table.add_column("Status")
    table.add_column("Attempts")
    table.add_column("Latency")

    for prompt in PROMPTS:
        result = call_with_retry(prompt, max_retries=3)
        stats[result["status"]] = stats.get(result["status"], 0) + 1
        stats["total_attempts"] += result["attempts"]

        status_color = {
            "success":          "green",
            "content_filtered": "yellow",
            "failed":           "red",
        }.get(result["status"], "white")

        table.add_row(
            prompt[:35],
            f"[{status_color}]{result['status']}[/{status_color}]",
            str(result["attempts"]),
            f"{result['latency']:.2f}s",
        )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {stats}")

    langfuse.flush()
    console.print("\n[bold]Done! Check Grafana error panels and Langfuse error traces.[/bold]")
