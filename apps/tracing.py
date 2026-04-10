"""
Shared tracing setup — Langfuse + Phoenix (OpenTelemetry).

Import setup_tracing() once at the top of each app to get both platforms wired up.
Phoenix auto-instruments the OpenAI client via OpenInference.
Langfuse is used manually (decorator / SDK calls) for richer LLM metadata.
"""
import config
from langfuse import Langfuse, propagate_attributes
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpOTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.openai import OpenAIInstrumentor


def _phoenix_span_exporter():
    t = config.PHOENIX_OTLP_TIMEOUT_SEC
    if config.PHOENIX_OTLP_PROTOCOL == "http":
        return HttpOTLPSpanExporter(
            endpoint=config.PHOENIX_COLLECTOR_ENDPOINT,
            timeout=t,
        )
    return GrpcOTLPSpanExporter(
        endpoint=config.PHOENIX_OTLP_GRPC_ENDPOINT,
        insecure=True,
        timeout=t,
    )


def setup_tracing(project_name: str = "llm-monitoring-lab") -> Langfuse:
    """
    Call once at startup.
    Returns a configured Langfuse client.
    Phoenix is wired up as a side effect via OTel auto-instrumentation.
    """
    # ── Phoenix via OpenTelemetry ─────────────────────────────────────────────
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            _phoenix_span_exporter(),
            export_timeout_millis=float(config.OTEL_BSP_EXPORT_TIMEOUT_MS),
        )
    )
    trace_api.set_tracer_provider(tracer_provider)

    # Auto-instrument every openai.ChatCompletion call → Phoenix sees it
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    # ── Langfuse ─────────────────────────────────────────────────────────────
    langfuse = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
        timeout=config.LANGFUSE_TIMEOUT,
    )
    # Langfuse v4 exposes propagate_attributes on the package, not the client; some labs expect
    # langfuse.propagate_attributes(...) — alias so both import styles work.
    setattr(langfuse, "propagate_attributes", propagate_attributes)

    return langfuse


__all__ = ["setup_tracing", "propagate_attributes"]
