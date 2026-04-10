"""
Load Generator — Continuous traffic simulator + Prometheus metrics server
=========================================================================

Runs forever, sending requests to Ollama at a configurable rate.
Exposes metrics on :8000 for Prometheus to scrape.

This is what populates the Grafana dashboards with live data.

HOW IT WORKS:
  - Spawns N worker threads, each running one scenario in a loop
  - Every request updates Prometheus metrics (counters, histograms, gauges)
  - Prometheus scrapes /metrics every 15s
  - Grafana queries Prometheus every 10s and renders live charts

IN PRODUCTION this is replaced by your real application traffic.
"""
import os
import time
import random
import threading
import uuid
from openai import OpenAI
from prometheus_client import start_http_server

import config
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, TOKEN_USAGE, ERROR_COUNT,
    RETRY_COUNT, RAG_RETRIEVAL_LATENCY, RAG_GENERATION_LATENCY,
    RAG_DOCS_RETRIEVED, AGENT_ITERATIONS, TOOL_CALL_COUNT,
    TOOL_CALL_LATENCY, ACTIVE_SESSIONS,
)

ollama = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
MODEL  = config.DEFAULT_MODEL

PROMPTS = [
    "What is artificial intelligence?",
    "Explain Docker in one sentence.",
    "What is 15 + 27?",
    "Name the capital of Germany.",
    "What is HTTP?",
    "Why is sleep important?",
    "What is machine learning?",
    "Describe Python in 10 words.",
    "What is a REST API?",
    "How does TCP/IP work?",
]


def simple_chat_scenario():
    """Simulates a basic chat request."""
    prompt = random.choice(PROMPTS)
    ACTIVE_SESSIONS.inc()
    t0 = time.time()

    # Inject random errors for realism (20% chance)
    should_fail = random.random() < 0.20

    try:
        if should_fail:
            error_type = random.choice(["timeout", "rate_limit", "model_error"])
            ERROR_COUNT.labels(model=MODEL, error_type=error_type).inc()
            REQUEST_COUNT.labels(model=MODEL, app="chat", status="error").inc()
            REQUEST_LATENCY.labels(model=MODEL, app="chat").observe(random.uniform(0.1, 2.0))
            # Simulate retries on transient errors
            if error_type != "content_filter":
                RETRY_COUNT.labels(model=MODEL, reason=error_type).inc()
            time.sleep(random.uniform(0.2, 1.5))
            return

        response = ollama.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )

        latency = time.time() - t0
        REQUEST_COUNT.labels(model=MODEL, app="chat", status="success").inc()
        REQUEST_LATENCY.labels(model=MODEL, app="chat").observe(latency)
        TOKEN_USAGE.labels(model=MODEL, token_type="prompt").inc(response.usage.prompt_tokens)
        TOKEN_USAGE.labels(model=MODEL, token_type="completion").inc(response.usage.completion_tokens)

    except Exception:
        REQUEST_COUNT.labels(model=MODEL, app="chat", status="error").inc()
        ERROR_COUNT.labels(model=MODEL, error_type="unknown").inc()
        REQUEST_LATENCY.labels(model=MODEL, app="chat").observe(time.time() - t0)
    finally:
        ACTIVE_SESSIONS.dec()


def rag_scenario():
    """Simulates a RAG pipeline request."""
    ACTIVE_SESSIONS.inc()

    # Retrieval phase
    retrieval_latency = random.uniform(0.01, 0.12)
    RAG_RETRIEVAL_LATENCY.observe(retrieval_latency)
    RAG_DOCS_RETRIEVED.observe(random.randint(1, 5))
    time.sleep(retrieval_latency)

    # Generation phase
    t0 = time.time()
    try:
        response = ollama.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": random.choice(PROMPTS)}],
            max_tokens=150,
            temperature=0.3,
        )
        gen_latency = time.time() - t0
        total_latency = retrieval_latency + gen_latency

        RAG_GENERATION_LATENCY.observe(gen_latency)
        REQUEST_COUNT.labels(model=MODEL, app="rag", status="success").inc()
        REQUEST_LATENCY.labels(model=MODEL, app="rag").observe(total_latency)
        TOKEN_USAGE.labels(model=MODEL, token_type="prompt").inc(response.usage.prompt_tokens)
        TOKEN_USAGE.labels(model=MODEL, token_type="completion").inc(response.usage.completion_tokens)

    except Exception:
        REQUEST_COUNT.labels(model=MODEL, app="rag", status="error").inc()
        ERROR_COUNT.labels(model=MODEL, error_type="unknown").inc()
        REQUEST_LATENCY.labels(model=MODEL, app="rag").observe(time.time() - t0)
    finally:
        ACTIVE_SESSIONS.dec()


def agent_scenario():
    """Simulates an agent with tool calls."""
    ACTIVE_SESSIONS.inc()
    t0 = time.time()
    iterations = random.randint(1, 4)
    total_tokens = 0

    try:
        for i in range(iterations):
            # Each agent iteration calls the LLM
            response = ollama.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "What is 5 + 5?"}],
                max_tokens=50,
            )
            total_tokens += response.usage.total_tokens

            # Simulate tool call on iterations before last
            if i < iterations - 1:
                tool_name    = random.choice(["calculator", "search", "weather"])
                tool_latency = random.uniform(0.01, 0.5)
                TOOL_CALL_COUNT.labels(tool_name=tool_name, status="success").inc()
                TOOL_CALL_LATENCY.labels(tool_name=tool_name).observe(tool_latency)
                time.sleep(tool_latency)

        AGENT_ITERATIONS.observe(iterations)
        TOKEN_USAGE.labels(model=MODEL, token_type="prompt").inc(total_tokens // 2)
        TOKEN_USAGE.labels(model=MODEL, token_type="completion").inc(total_tokens // 2)
        REQUEST_COUNT.labels(model=MODEL, app="agent", status="success").inc()
        REQUEST_LATENCY.labels(model=MODEL, app="agent").observe(time.time() - t0)

    except Exception:
        REQUEST_COUNT.labels(model=MODEL, app="agent", status="error").inc()
        ERROR_COUNT.labels(model=MODEL, error_type="unknown").inc()
    finally:
        ACTIVE_SESSIONS.dec()


SCENARIOS = [
    (simple_chat_scenario, 0.60),   # 60% of traffic is simple chat
    (rag_scenario,         0.30),   # 30% is RAG
    (agent_scenario,       0.10),   # 10% is agent
]


def worker(worker_id: int, delay_between_requests: float = 5.0):
    """
    Each worker thread continuously runs scenarios.
    delay_between_requests controls simulated QPS per worker.
    """
    print(f"[worker-{worker_id}] started")
    while True:
        scenario_fn = random.choices(
            [s[0] for s in SCENARIOS],
            weights=[s[1] for s in SCENARIOS],
        )[0]
        try:
            scenario_fn()
        except Exception as e:
            print(f"[worker-{worker_id}] unhandled error: {e}")

        # Jitter: ±30% of delay
        jitter = delay_between_requests * random.uniform(0.7, 1.3)
        time.sleep(jitter)


if __name__ == "__main__":
    port = config.METRICS_PORT
    num_workers = int(os.getenv("NUM_WORKERS", "3"))
    req_delay   = float(os.getenv("REQUEST_DELAY_S", "4.0"))  # seconds between requests per worker

    print(f"Starting Prometheus metrics server on :{port}")
    start_http_server(port)

    print(f"Starting {num_workers} load generator workers (delay={req_delay}s between requests)")
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i, req_delay), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)  # stagger startup

    print("Load generator running. Metrics at http://localhost:8000/metrics")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopped.")
