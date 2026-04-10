"""
Lab 3 — LLM Agent with Tool Calls
===================================

WHAT WE MONITOR (agents are the hardest to debug — here's why):

  Iteration count    — Agents run in a loop. 10+ iterations = runaway agent → cost explosion.
  Tool call latency  — Each tool call adds latency. Slow tools = slow agent.
  Tool errors        — A broken tool causes the agent to loop or give up.
  Total tokens       — Multi-turn agents consume tokens every iteration (compounding cost).
  Tool selection     — Is the agent picking the right tool? Bad selection = wasted calls.
  Final answer rate  — What % of runs reach a conclusion vs. timing out?

TOOLS SIMULATED:
  calculator   — Math operations (always fast, rarely fails)
  weather      — Fake weather lookup (simulates network latency, occasional failure)
  search       — Fake document search (variable latency)
  database     — Fake DB query (slow, simulates prod bottleneck)

HOW TO RUN:
  python 03_agent_with_tools.py
"""
import json
import time
import random
import uuid
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

import config
from tracing import setup_tracing, propagate_attributes
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, TOKEN_USAGE, ERROR_COUNT,
    AGENT_ITERATIONS, TOOL_CALL_COUNT, TOOL_CALL_LATENCY, ACTIVE_SESSIONS,
)

console  = Console()
langfuse = setup_tracing("agent-tools")
ollama   = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")

MAX_ITERATIONS = 6

# ── Tool implementations ──────────────────────────────────────────────────────

def tool_calculator(expression: str) -> str:
    """Evaluate a safe math expression."""
    time.sleep(0.01)  # near-instant
    try:
        # Whitelist safe chars only
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: unsafe expression"
        return str(round(eval(expression), 6))  # noqa: S307 — safe input validated above
    except Exception as e:
        return f"Error: {e}"


def tool_weather(city: str) -> str:
    """Return fake weather for a city (simulates external API latency)."""
    time.sleep(random.uniform(0.1, 0.4))  # network call
    if random.random() < 0.1:             # 10% failure rate
        raise TimeoutError(f"Weather API timed out for {city}")
    conditions = ["sunny", "cloudy", "rainy", "windy", "snowy"]
    temp = random.randint(-5, 35)
    return f"{city}: {random.choice(conditions)}, {temp}°C"


def tool_search(query: str) -> str:
    """Fake knowledge base search."""
    time.sleep(random.uniform(0.05, 0.2))
    snippets = {
        "python": "Python is a high-level programming language known for readability.",
        "llm":    "Large language models are transformer-based neural networks trained on text.",
        "docker": "Docker containers package software and its dependencies together.",
        "default":"No specific information found. Try a more specific query.",
    }
    for kw, snippet in snippets.items():
        if kw in query.lower():
            return snippet
    return snippets["default"]


def tool_database(sql_query: str) -> str:
    """Simulate a slow database query (common prod bottleneck)."""
    time.sleep(random.uniform(0.3, 1.2))  # DB queries are slow
    if "users" in sql_query.lower():
        return "users table: 42,891 rows, last updated 2 minutes ago"
    if "orders" in sql_query.lower():
        return "orders table: 1.2M rows, avg order value $87.40"
    return "Query executed. 0 rows returned."


TOOLS = {
    "calculator": tool_calculator,
    "weather":    tool_weather,
    "search":     tool_search,
    "database":   tool_database,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Input: arithmetic expression string.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. '2 * (3 + 4)'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search knowledge base for information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "database",
            "description": "Run a plain-English database query to look up business data.",
            "parameters": {
                "type": "object",
                "properties": {"sql_query": {"type": "string"}},
                "required": ["sql_query"],
            },
        },
    },
]


def run_agent(task: str, model: str = config.DEFAULT_MODEL) -> dict:
    """
    ReAct-style agent loop.
    Observes: iterations, tool calls, tokens, total latency.
    """
    ACTIVE_SESSIONS.inc()
    session_id = str(uuid.uuid4())

    messages = [
        {"role": "system", "content": (
            "You are a helpful agent. Use tools when needed. "
            "When you have a final answer, respond WITHOUT calling any tool."
        )},
        {"role": "user", "content": task},
    ]

    total_prompt_tokens     = 0
    total_completion_tokens = 0
    iterations              = 0
    agent_start             = time.time()

    try:
        with propagate_attributes(session_id=session_id, trace_name="agent-run"):
            with langfuse.start_as_current_observation(name="agent-run", input=task) as root_span:
                try:
                    for iteration in range(MAX_ITERATIONS):
                        iterations = iteration + 1
                        iter_span = root_span.start_observation(
                            name=f"iteration-{iterations}",
                            input={"messages": len(messages)},
                        )

                        response = ollama.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=TOOL_SCHEMAS,
                            tool_choice="auto",
                            max_tokens=512,
                        )

                        total_prompt_tokens     += response.usage.prompt_tokens
                        total_completion_tokens += response.usage.completion_tokens

                        msg = response.choices[0].message

                        # No tool call → agent reached a final answer
                        if not msg.tool_calls:
                            iter_span.update(
                                output=msg.content,
                                metadata={"final_answer": True},
                            )
                            iter_span.end()
                            break

                        # Process tool calls
                        tool_results = []
                        for tc in msg.tool_calls:
                            tool_name = tc.function.name
                            tool_args = json.loads(tc.function.arguments)

                            tool_span = root_span.start_observation(
                                name=f"tool:{tool_name}",
                                input=tool_args,
                            )
                            t0 = time.time()

                            try:
                                result = TOOLS[tool_name](**tool_args)
                                tool_latency = time.time() - t0

                                TOOL_CALL_COUNT.labels(tool_name=tool_name, status="success").inc()
                                TOOL_CALL_LATENCY.labels(tool_name=tool_name).observe(tool_latency)
                                tool_span.update(
                                    output=result,
                                    metadata={"latency_s": round(tool_latency, 3)},
                                )
                                tool_span.end()
                                tool_results.append({"tool_call_id": tc.id, "content": result})

                            except Exception as tool_err:
                                tool_latency = time.time() - t0
                                TOOL_CALL_COUNT.labels(tool_name=tool_name, status="error").inc()
                                ERROR_COUNT.labels(model=model, error_type=f"tool_{tool_name}").inc()
                                tool_span.update(
                                    output=str(tool_err),
                                    level="ERROR",
                                    metadata={"latency_s": round(tool_latency, 3)},
                                )
                                tool_span.end()
                                tool_results.append({"tool_call_id": tc.id, "content": f"Error: {tool_err}"})

                        # Append assistant message + tool results to conversation
                        messages.append(msg)
                        for tr in tool_results:
                            messages.append({"role": "tool", **tr})

                        iter_span.update(metadata={"tool_calls": len(msg.tool_calls)})
                        iter_span.end()

                    else:
                        # Hit MAX_ITERATIONS without a final answer
                        msg.content = "I was unable to complete the task within the iteration limit."
                        ERROR_COUNT.labels(model=model, error_type="max_iterations_exceeded").inc()

                    total_latency = time.time() - agent_start

                    # ── Record metrics ────────────────────────────────────────
                    AGENT_ITERATIONS.observe(iterations)
                    TOKEN_USAGE.labels(model=model, token_type="prompt").inc(total_prompt_tokens)
                    TOKEN_USAGE.labels(model=model, token_type="completion").inc(total_completion_tokens)
                    REQUEST_COUNT.labels(model=model, app="agent", status="success").inc()
                    REQUEST_LATENCY.labels(model=model, app="agent").observe(total_latency)

                    root_span.update(
                        output=msg.content,
                        metadata={
                            "iterations":          iterations,
                            "total_latency_s":     round(total_latency, 3),
                            "total_prompt_tokens": total_prompt_tokens,
                            "total_comp_tokens":   total_completion_tokens,
                        },
                    )

                    return {
                        "answer":     msg.content,
                        "iterations": iterations,
                        "latency":    total_latency,
                        "tokens":     total_prompt_tokens + total_completion_tokens,
                    }

                except Exception as exc:
                    REQUEST_COUNT.labels(model=model, app="agent", status="error").inc()
                    ERROR_COUNT.labels(model=model, error_type=type(exc).__name__).inc()
                    root_span.update(output=str(exc), level="ERROR")
                    raise
    finally:
        ACTIVE_SESSIONS.dec()


# ── Demo ──────────────────────────────────────────────────────────────────────

TASKS = [
    "What is 15% of 847?",
    "What's the weather in Tokyo and London? Which is warmer?",
    "Search for information about LLMs, then tell me 2 key facts.",
    "How many users are in our database?",
    "Calculate the compound interest on $1000 at 5% for 3 years, then look up what Python is.",
]

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Lab 3: Agent with Tools[/bold cyan]\n"
        "Watch iteration spans, tool call timings, and token accumulation in Langfuse.",
        border_style="cyan",
    ))

    for task in TASKS:
        console.print(f"\n[bold yellow]Task:[/bold yellow] {task}")
        result = run_agent(task)
        console.print(f"[bold green]Answer:[/bold green] {result['answer']}")
        console.print(
            f"[dim]  iterations={result['iterations']}  "
            f"latency={result['latency']:.2f}s  "
            f"tokens={result['tokens']}[/dim]"
        )

    langfuse.flush()
    console.print("\n[bold]Done! Check Langfuse for per-iteration spans.[/bold]")
