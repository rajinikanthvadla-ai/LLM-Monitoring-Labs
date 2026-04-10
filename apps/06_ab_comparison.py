"""
Lab 6 — A/B Model Comparison
==============================

REAL-WORLD SCENARIO: You want to upgrade from model A to model B.
  Before switching 100% of traffic, run both in parallel and compare.

WHAT WE COMPARE:
  Latency          — Is the new model faster or slower?
  Token usage      — Does it use fewer tokens for the same output?
  Output length    — Does it give more/less verbose answers?
  Cost estimate    — What's the $ difference at scale?
  Quality score    — Simple heuristic: longer isn't always better.

IN PRODUCTION:
  - Shadow mode:  new model runs alongside old, responses discarded
  - Canary:       5% of real users get new model, compare metrics
  - Full rollout: shift 100% after metrics look good

HOW TO RUN:
  python 06_ab_comparison.py
  # Needs TWO models pulled. If only llama3.2:1b is available, it runs both sides with same model
  # to demonstrate the comparison framework.
"""
import time
import uuid
from dataclasses import dataclass, field
from openai import OpenAI
from prometheus_client import push_to_gateway, CollectorRegistry, Histogram, Counter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from tracing import setup_tracing, propagate_attributes
from metrics import REQUEST_COUNT, REQUEST_LATENCY, TOKEN_USAGE

console  = Console()
langfuse = setup_tracing("ab-comparison")
ollama   = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")

# ── Dedicated push registry for A/B metrics ───────────────────────────────────
reg = CollectorRegistry()
ab_latency = Histogram(
    "llm_ab_latency_seconds", "A/B latency per variant",
    ["model", "variant"], registry=reg,
    buckets=[0.5, 1, 2, 5, 10, 20, 30],
)
ab_tokens = Counter(
    "llm_ab_tokens_total", "A/B token usage",
    ["model", "variant", "token_type"], registry=reg,
)


@dataclass
class ModelResult:
    model:             str
    variant:           str          # "A" or "B"
    answer:            str
    latency:           float
    prompt_tokens:     int
    completion_tokens: int
    output_length:     int = field(init=False)

    def __post_init__(self):
        self.output_length = len(self.answer.split())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def est_cost_usd(self) -> float:
        return (self.total_tokens / 1_000_000) * 0.15  # $0.15/M tokens (GPT-4o pricing sim)


def call_model(prompt: str, model: str, variant: str, session_id: str) -> ModelResult:
    """Call one model variant and record everything."""
    with propagate_attributes(
        session_id=session_id,
        trace_name="ab-comparison",
        tags=[f"variant:{variant}", f"model:{model}"],
    ):
        with langfuse.start_as_current_observation(
            name="ab-comparison",
            input={"prompt": prompt, "variant": variant},
        ) as root_span:
            span = root_span.start_observation(name=f"model-{variant}")
            t0 = time.time()
            try:
                response = ollama.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7,
                )
                latency = time.time() - t0
                answer = response.choices[0].message.content

                REQUEST_COUNT.labels(model=model, app="ab-comparison", status="success").inc()
                REQUEST_LATENCY.labels(model=model, app="ab-comparison").observe(latency)
                TOKEN_USAGE.labels(model=model, token_type="prompt").inc(response.usage.prompt_tokens)
                TOKEN_USAGE.labels(model=model, token_type="completion").inc(response.usage.completion_tokens)

                ab_latency.labels(model=model, variant=variant).observe(latency)
                ab_tokens.labels(model=model, variant=variant, token_type="prompt").inc(response.usage.prompt_tokens)
                ab_tokens.labels(model=model, variant=variant, token_type="completion").inc(response.usage.completion_tokens)

                span.update(
                    output=answer,
                    metadata={
                        "latency_s":         round(latency, 3),
                        "prompt_tokens":     response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                )
                span.score(name="variant", value=0.0 if variant == "A" else 1.0)
                span.end()

                return ModelResult(
                    model=model, variant=variant, answer=answer, latency=latency,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

            except Exception as exc:
                latency = time.time() - t0
                REQUEST_COUNT.labels(model=model, app="ab-comparison", status="error").inc()
                span.update(output=str(exc), level="ERROR")
                span.end()
                return ModelResult(
                    model=model, variant=variant, answer=f"[ERROR] {exc}", latency=latency,
                    prompt_tokens=0, completion_tokens=0,
                )


def run_ab_test(
    prompts: list[str],
    model_a: str = config.FAST_MODEL,
    model_b: str = config.SMART_MODEL,
) -> dict:
    """
    Run every prompt through both models. Collect and compare results.
    """
    results_a, results_b = [], []
    session_id = str(uuid.uuid4())

    for i, prompt in enumerate(prompts):
        console.print(f"\n[dim]Prompt {i+1}/{len(prompts)}:[/dim] {prompt[:70]}")

        r_a = call_model(prompt, model_a, "A", session_id)
        r_b = call_model(prompt, model_b, "B", session_id)

        results_a.append(r_a)
        results_b.append(r_b)

        console.print(f"  [blue]Model A ({model_a}):[/blue] {r_a.latency:.2f}s, {r_a.total_tokens} tokens")
        console.print(f"  [magenta]Model B ({model_b}):[/magenta] {r_b.latency:.2f}s, {r_b.total_tokens} tokens")

    def avg(lst, key):
        vals = [getattr(r, key) for r in lst]
        return sum(vals) / len(vals) if vals else 0

    # Push to pushgateway
    try:
        push_to_gateway(
            config.PROMETHEUS_PUSHGATEWAY,
            job="ab-comparison",
            grouping_key={"session_id": session_id[:8]},
            registry=reg,
        )
    except Exception as e:
        console.print(f"[yellow]Pushgateway unavailable: {e}[/yellow]")

    return {
        "model_a": {
            "name":           model_a,
            "avg_latency_s":  round(avg(results_a, "latency"), 3),
            "avg_tokens":     round(avg(results_a, "total_tokens")),
            "avg_output_len": round(avg(results_a, "output_length")),
            "total_cost_usd": round(sum(r.est_cost_usd for r in results_a), 6),
        },
        "model_b": {
            "name":           model_b,
            "avg_latency_s":  round(avg(results_b, "latency"), 3),
            "avg_tokens":     round(avg(results_b, "total_tokens")),
            "avg_output_len": round(avg(results_b, "output_length")),
            "total_cost_usd": round(sum(r.est_cost_usd for r in results_b), 6),
        },
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    "Summarize the benefits of containerization in 2 sentences.",
    "What is the difference between supervised and unsupervised learning?",
    "Write a one-paragraph pitch for a time-tracking app.",
    "What are 3 best practices for REST API design?",
    "Explain why monitoring matters for AI systems.",
]

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Lab 6: A/B Model Comparison[/bold cyan]\n"
        f"Model A: [blue]{config.FAST_MODEL}[/blue]  vs  Model B: [magenta]{config.SMART_MODEL}[/magenta]\n"
        "(Same model used for both if only one is pulled — just demonstrates the framework)",
        border_style="cyan",
    ))

    summary = run_ab_test(TEST_PROMPTS)

    # Pretty comparison table
    table = Table(title="\nA/B Test Summary", show_header=True)
    table.add_column("Metric",       style="bold")
    table.add_column(f"Model A ({summary['model_a']['name']})", style="blue")
    table.add_column(f"Model B ({summary['model_b']['name']})", style="magenta")

    metrics = ["avg_latency_s", "avg_tokens", "avg_output_len", "total_cost_usd"]
    labels  = ["Avg Latency (s)", "Avg Tokens", "Avg Output Words", "Est. Cost ($)"]

    for m, l in zip(metrics, labels):
        va = str(summary["model_a"][m])
        vb = str(summary["model_b"][m])
        table.add_row(l, va, vb)

    console.print(table)

    langfuse.flush()
    console.print("\n[bold]Done! Filter Langfuse traces by tag 'variant:A' vs 'variant:B'.[/bold]")
