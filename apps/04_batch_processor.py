"""
Lab 4 — Batch Processing with Prometheus Pushgateway
======================================================

REAL-WORLD SCENARIO: Nightly batch job that processes 1000s of documents
  e.g. summarize customer support tickets, classify emails, extract entities.

WHAT WE MONITOR (batch jobs are "dark" — no live traffic, no SLAs, but still need monitoring):

  Throughput       — docs/second. Slow batch = misses the morning deadline.
  Success rate     — Failed items need retry or manual review.
  Cost projection  — Total tokens × price/token = batch cost.
  Queue drain rate — How fast is the backlog clearing?
  Worker health    — Are workers dying mid-batch?

KEY DIFFERENCE FROM REAL-TIME:
  Batch jobs can't serve a Prometheus scrape endpoint (they exit when done).
  Solution: push metrics to Prometheus Pushgateway before exit.

HOW TO RUN:
  python 04_batch_processor.py
  # Then check: http://localhost:9091 (Pushgateway) and http://localhost:9090 (Prometheus)
"""
import time
import random
import uuid
from datetime import datetime
from openai import OpenAI
from prometheus_client import (
    CollectorRegistry, Counter, Histogram, Gauge, push_to_gateway,
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn

import config
from tracing import setup_tracing, propagate_attributes

console  = Console()
langfuse = setup_tracing("batch-processor")
ollama   = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")

# ── Use a dedicated registry for push (avoids double-registering default metrics) ──
registry = CollectorRegistry()

batch_processed  = Counter("batch_docs_processed_total",  "Documents processed",  ["status"],    registry=registry)
batch_tokens     = Counter("batch_tokens_total",           "Tokens used in batch", ["token_type"],registry=registry)
batch_latency    = Histogram("batch_doc_latency_seconds",  "Per-doc processing time",             registry=registry,
                             buckets=[0.5, 1, 2, 5, 10, 20, 30])
batch_queue_size = Gauge("batch_queue_remaining",          "Docs still in queue",                 registry=registry)
batch_cost_usd   = Gauge("batch_estimated_cost_usd",       "Estimated cost of the batch run",     registry=registry)

# Rough cost estimate: llama3.2:1b is free locally, but we simulate an API cost
COST_PER_1K_TOKENS = 0.002  # $ per 1K tokens (mimics GPT-3.5 pricing for demo)


# ── Fake batch queue ──────────────────────────────────────────────────────────
SUPPORT_TICKETS = [
    "Customer says the app crashes when uploading files larger than 10MB.",
    "User wants to know how to export data to CSV format.",
    "Payment failed three times, customer is asking for a refund.",
    "Login page is not loading on Safari on iOS 17.",
    "Feature request: dark mode support for the dashboard.",
    "User received wrong order, wants immediate resolution.",
    "API key stopped working after password reset.",
    "Dashboard graphs are not refreshing in real time.",
    "Customer asking about GDPR data deletion policy.",
    "Mobile app is draining battery very fast.",
    "Two-factor auth codes are not being received via SMS.",
    "Report export is generating incorrect totals.",
]

SYSTEM_PROMPT = (
    "You are a customer support classifier. "
    "For the given ticket, respond with ONLY a JSON object: "
    '{"category": "<bug|feature|billing|question>", '
    '"priority": "<high|medium|low>", '
    '"summary": "<one sentence>"}'
)


def process_ticket(ticket: str, batch_id: str, ticket_num: int) -> dict:
    """Classify a single support ticket."""
    with propagate_attributes(session_id=batch_id, trace_name="batch-ticket-classification"):
        with langfuse.start_as_current_observation(
            name="batch-ticket-classification",
            input={"ticket": ticket, "ticket_num": ticket_num},
        ) as root:
            span = root.start_observation(name="classify")
            t0 = time.time()
            try:
                response = ollama.chat.completions.create(
                    model=config.DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": ticket},
                    ],
                    max_tokens=150,
                    temperature=0.1,
                )

                latency = time.time() - t0
                content = response.choices[0].message.content

                # Update push-registry metrics
                batch_processed.labels(status="success").inc()
                batch_latency.observe(latency)
                batch_tokens.labels(token_type="prompt").inc(response.usage.prompt_tokens)
                batch_tokens.labels(token_type="completion").inc(response.usage.completion_tokens)

                total_tokens = response.usage.total_tokens
                estimated_cost = (total_tokens / 1000) * COST_PER_1K_TOKENS
                batch_cost_usd.inc(estimated_cost)

                span.update(output=content, metadata={"latency_s": round(latency, 3)})
                span.end()

                return {"ticket": ticket, "result": content, "latency": latency, "tokens": total_tokens, "ok": True}

            except Exception as exc:
                latency = time.time() - t0
                batch_processed.labels(status="error").inc()
                span.update(output=str(exc), level="ERROR")
                span.end()
                return {"ticket": ticket, "result": str(exc), "latency": latency, "tokens": 0, "ok": False}


def run_batch(tickets: list[str]) -> dict:
    batch_id   = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    total      = len(tickets)
    results    = []

    console.print(f"\n[bold]Batch ID:[/bold] {batch_id}")
    console.print(f"[bold]Queue size:[/bold] {total} tickets\n")

    batch_queue_size.set(total)

    with Progress(
        SpinnerColumn(), "[progress.description]{task.description}",
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing tickets...", total=total)

        for i, ticket in enumerate(tickets):
            result = process_ticket(ticket, batch_id, i + 1)
            results.append(result)
            batch_queue_size.set(total - (i + 1))
            progress.advance(task)

    # ── Push final metrics to Pushgateway ─────────────────────────────────────
    try:
        push_to_gateway(
            config.PROMETHEUS_PUSHGATEWAY,
            job=f"batch-processor",
            grouping_key={"batch_id": batch_id},
            registry=registry,
        )
        console.print("\n[green]Metrics pushed to Prometheus Pushgateway.[/green]")
    except Exception as e:
        console.print(f"\n[yellow]Warning: Could not push to Pushgateway: {e}[/yellow]")

    success = sum(1 for r in results if r["ok"])
    total_tokens = sum(r["tokens"] for r in results)
    avg_latency  = sum(r["latency"] for r in results) / len(results)

    return {
        "batch_id":     batch_id,
        "total":        total,
        "success":      success,
        "failed":       total - success,
        "success_rate": f"{success/total*100:.1f}%",
        "total_tokens": total_tokens,
        "avg_latency":  avg_latency,
        "est_cost_usd": round((total_tokens / 1000) * COST_PER_1K_TOKENS, 6),
    }


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Lab 4: Batch Processor[/bold cyan]\n"
        "Metrics pushed to Pushgateway → http://localhost:9091",
        border_style="cyan",
    ))

    summary = run_batch(SUPPORT_TICKETS)

    console.print("\n[bold cyan]── Batch Summary ────────────────[/bold cyan]")
    for k, v in summary.items():
        console.print(f"  {k:<20} {v}")

    langfuse.flush()
    console.print("\n[bold]Done! Check Langfuse for batch traces grouped by session.[/bold]")
