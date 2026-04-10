"""
Lab 2 — RAG Pipeline with Stage-Level Monitoring
=================================================

WHAT WE MONITOR (RAG-specific — critical in prod):

  Retrieval latency   — Slow vector DB = slow answers. Usually p95 < 100ms target.
  # docs retrieved    — Too few = poor answers. Too many = expensive + slow.
  Relevance score     — Are retrieved docs actually relevant? (hallucination risk)
  Generation latency  — Time to produce answer given context.
  Context length      — Big context = more tokens = more cost + slower.
  Answer faithfulness — Did the model stay grounded? (needs eval step)
  Empty retrievals    — Query found nothing → fallback or escalate to human.

ARCHITECTURE:
  User Query → [Retriever] → [Ranked Docs] → [LLM with context] → Answer
                ↑ timed           ↑ counted           ↑ timed + token counted

HOW TO RUN:
  python 02_rag_pipeline.py
"""
import time
import uuid
import random
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from tracing import setup_tracing, propagate_attributes
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, TOKEN_USAGE, ERROR_COUNT,
    RAG_RETRIEVAL_LATENCY, RAG_GENERATION_LATENCY, RAG_DOCS_RETRIEVED,
    ACTIVE_SESSIONS,
)

console  = Console()
langfuse = setup_tracing("rag-pipeline")
ollama   = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")

# ── Fake knowledge base (mimics a vector store) ───────────────────────────────
KNOWLEDGE_BASE = [
    {
        "id": "doc-1",
        "content": "Langfuse is an open-source LLM observability platform. "
                   "It provides tracing, evals, and prompt management.",
        "topic": "monitoring",
    },
    {
        "id": "doc-2",
        "content": "Phoenix by Arize is an open-source tool for LLM tracing using OpenTelemetry. "
                   "It supports RAG evaluation and agent debugging.",
        "topic": "monitoring",
    },
    {
        "id": "doc-3",
        "content": "Ollama runs large language models locally on your machine. "
                   "It supports llama3, mistral, gemma, and many other models.",
        "topic": "local-llm",
    },
    {
        "id": "doc-4",
        "content": "Prometheus collects metrics by scraping HTTP endpoints. "
                   "Grafana visualises those metrics using PromQL queries.",
        "topic": "metrics",
    },
    {
        "id": "doc-5",
        "content": "RAG (Retrieval-Augmented Generation) reduces hallucination by "
                   "grounding the LLM in retrieved documents from a knowledge base.",
        "topic": "rag",
    },
    {
        "id": "doc-6",
        "content": "Token limits in LLMs constrain how much context can be included. "
                   "GPT-4 supports 128k tokens; llama3.2 supports 128k tokens too.",
        "topic": "llm",
    },
    {
        "id": "doc-7",
        "content": "Latency SLAs for LLM APIs: p50 < 3s, p95 < 8s, p99 < 15s "
                   "are common targets for interactive chat applications.",
        "topic": "sla",
    },
]


def fake_vector_search(query: str, top_k: int = 3) -> list[dict]:
    """
    Simulates a vector search with realistic latency.
    In production: replace with Chroma, Pinecone, Weaviate, pgvector, etc.

    Real monitoring concern: slow vector DB = user sees slow first byte.
    """
    # Simulate network + compute time (5–80ms typical for vector DBs)
    time.sleep(random.uniform(0.005, 0.08))

    # Naive keyword overlap scoring (fake relevance)
    query_words = set(query.lower().split())
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_words  = set(doc["content"].lower().split())
        score      = len(query_words & doc_words) / max(len(query_words), 1)
        scored.append({**doc, "relevance_score": round(score, 3)})

    scored.sort(key=lambda d: d["relevance_score"], reverse=True)
    return scored[:top_k]


def rag_query(user_query: str, model: str = config.DEFAULT_MODEL) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.
    Every stage is timed and traced.
    """
    ACTIVE_SESSIONS.inc()
    session_id = str(uuid.uuid4())

    result = {}
    try:
        with propagate_attributes(
            session_id=session_id,
            trace_name="rag-pipeline",
            metadata={"model": str(model)},
        ):
            with langfuse.start_as_current_observation(
                name="rag-pipeline",
                input=user_query,
            ) as root_span:
                try:
                    # ── Stage 1: Retrieval ────────────────────────────────────
                    retrieval_span = root_span.start_observation(
                        name="retrieval",
                        input={"query": user_query},
                    )
                    t0 = time.time()

                    docs = fake_vector_search(user_query, top_k=3)
                    retrieval_latency = time.time() - t0

                    RAG_RETRIEVAL_LATENCY.observe(retrieval_latency)
                    RAG_DOCS_RETRIEVED.observe(len(docs))

                    retrieval_span.update(
                        output={"docs_count": len(docs), "top_score": docs[0]["relevance_score"] if docs else 0},
                        metadata={"latency_s": round(retrieval_latency, 4)},
                    )
                    retrieval_span.end()

                    if not docs or docs[0]["relevance_score"] == 0:
                        ERROR_COUNT.labels(model=model, error_type="empty_retrieval").inc()
                        root_span.update(output="No relevant documents found.", level="WARNING")
                        return {"answer": "I don't have information on that topic.", "docs": [], "latency": 0}

                    # ── Stage 2: Build context ─────────────────────────────────
                    context = "\n\n".join(
                        f"[Doc {i+1}] (relevance={d['relevance_score']})\n{d['content']}"
                        for i, d in enumerate(docs)
                    )

                    prompt = (
                        f"Answer the question using ONLY the provided context. "
                        f"If the context doesn't contain the answer, say so.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {user_query}\n\n"
                        f"Answer:"
                    )

                    # ── Stage 3: Generation ───────────────────────────────────
                    gen_span = root_span.start_observation(
                        name="generation",
                        input={"prompt_length": len(prompt)},
                    )
                    t1 = time.time()

                    response = ollama.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.3,
                    )

                    gen_latency = time.time() - t1
                    answer = response.choices[0].message.content

                    RAG_GENERATION_LATENCY.observe(gen_latency)
                    TOKEN_USAGE.labels(model=model, token_type="prompt").inc(response.usage.prompt_tokens)
                    TOKEN_USAGE.labels(model=model, token_type="completion").inc(response.usage.completion_tokens)

                    gen_span.update(
                        output=answer,
                        metadata={
                            "latency_s":         round(gen_latency, 3),
                            "prompt_tokens":     response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        },
                    )
                    gen_span.end()

                    total_latency = retrieval_latency + gen_latency
                    REQUEST_COUNT.labels(model=model, app="rag", status="success").inc()
                    REQUEST_LATENCY.labels(model=model, app="rag").observe(total_latency)

                    root_span.update(
                        output=answer,
                        metadata={
                            "retrieval_latency_s": round(retrieval_latency, 3),
                            "generation_latency_s": round(gen_latency, 3),
                            "total_latency_s": round(total_latency, 3),
                            "docs_retrieved": len(docs),
                        },
                    )

                    result = {
                        "answer":            answer,
                        "docs":              docs,
                        "retrieval_latency": retrieval_latency,
                        "gen_latency":       gen_latency,
                        "total_latency":     total_latency,
                        "usage":             response.usage,
                    }

                except Exception as exc:
                    REQUEST_COUNT.labels(model=model, app="rag", status="error").inc()
                    ERROR_COUNT.labels(model=model, error_type=type(exc).__name__).inc()
                    root_span.update(output=str(exc), level="ERROR")
                    raise
    finally:
        ACTIVE_SESSIONS.dec()

    return result


# ── Demo ──────────────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "What is Langfuse used for?",
    "How does RAG reduce hallucination?",
    "What models does Ollama support?",
    "What are typical LLM latency SLAs?",
    "Explain the difference between Prometheus and Grafana.",
    "What is the meaning of life?",   # Intentional no-match → shows empty retrieval
]

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Lab 2: RAG Pipeline[/bold cyan]\n"
        "Watch per-stage latency in [link=http://localhost:3000]Langfuse[/link] traces.\n"
        "Retrieval vs Generation breakdown is visible in [link=http://localhost:3001]Grafana[/link].",
        border_style="cyan",
    ))

    for query in SAMPLE_QUERIES:
        console.print(f"\n[bold yellow]Q:[/bold yellow] {query}")
        result = rag_query(query)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[dim]Answer[/dim]", result["answer"][:200])
        table.add_row("[dim]Retrieval[/dim]", f"{result.get('retrieval_latency', 0)*1000:.0f}ms")
        table.add_row("[dim]Generation[/dim]", f"{result.get('gen_latency', 0)*1000:.0f}ms")
        table.add_row("[dim]Docs used[/dim]", str(len(result.get("docs", []))))
        console.print(table)

    langfuse.flush()
    console.print("\n[bold]Done! Inspect per-stage spans in Langfuse.[/bold]")
