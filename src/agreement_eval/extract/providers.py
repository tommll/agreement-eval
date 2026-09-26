"""Model providers.

Three of them:

`AnthropicProvider`  - the Claude API.
`LlamaCppProvider`   - a local llama.cpp server (OpenAI-compatible /v1). Runs
                       GGUF weights on CPU with grammar-constrained decoding,
                       so a 4B model returns schema-valid JSON every time.
                       Reads receipt images too when the model is served with
                       its vision projector.
`MockProvider`       - a deterministic simulator with configurable, optionally
                       *correlated* error modes; makes the pipeline runnable
                       offline and gives the analysis a ground truth about what
                       agreement does and does not imply.

All three return the same `ExtractionResult`, so a run can mix them: a rater is
a (provider, model, prompt) triple and the analysis never asks where it came from.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import ExtractionConfig, JudgeConfig
from ..documents import Document
from ..normalize import is_null
from ..schemas import DatasetSchema
from .prompts import get_prompt

# USD per million tokens (input, output). Used for run cost accounting only.
PRICES: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    price = PRICES.get(model)
    if not price:
        return None
    return input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]


@dataclass
class ExtractionResult:
    values: Dict[str, Optional[str]] = field(default_factory=dict)
    evidence: Dict[str, Optional[str]] = field(default_factory=dict)
    raw_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _media_type(path: Path) -> str:
    return {".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(
        path.suffix.lower(), "image/jpeg"
    )


def _image_block(doc: Document) -> Optional[Dict[str, Any]]:
    if doc.image_path and Path(doc.image_path).exists():
        path = Path(doc.image_path)
        data = base64.standard_b64encode(path.read_bytes()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": _media_type(path), "data": data}}
    if doc.image_uri:
        return {"type": "image", "source": {"type": "url", "url": doc.image_uri}}
    return None


def _split_value_evidence(payload: Dict[str, Any], schema: DatasetSchema, with_evidence: bool):
    values: Dict[str, Optional[str]] = {}
    evidence: Dict[str, Optional[str]] = {}
    for name in schema.field_names:
        item = payload.get(name)
        if with_evidence and isinstance(item, dict):
            raw_value, raw_evidence = item.get("value"), item.get("evidence")
        else:
            raw_value, raw_evidence = item, None
        values[name] = None if is_null(raw_value) else str(raw_value).strip()
        evidence[name] = None if is_null(raw_evidence) else str(raw_evidence).strip()
    return values, evidence


class Provider:
    """Interface shared by real and mock providers."""

    def extract(self, doc: Document, schema: DatasetSchema, cfg: ExtractionConfig) -> ExtractionResult:
        raise NotImplementedError


class AnthropicProvider(Provider):
    def __init__(self, client: Any = None) -> None:
        import anthropic  # imported lazily so mock-only runs need no SDK config

        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()

    def extract(self, doc: Document, schema: DatasetSchema, cfg: ExtractionConfig) -> ExtractionResult:
        prompt = get_prompt(cfg.prompt)
        content: List[Dict[str, Any]] = []
        if cfg.input_mode in ("image", "both"):
            block = _image_block(doc)
            if block is None and cfg.input_mode == "image":
                return ExtractionResult(error=f"config {cfg.id} needs an image, none available for {doc.doc_id}")
            if block is not None:
                content.append(block)
        if cfg.input_mode in ("text", "both"):
            content.append({"type": "text", "text": prompt.user(schema, doc, cfg.max_ocr_chars)})
        elif not content:
            content.append({"type": "text", "text": prompt.user(schema, doc, cfg.max_ocr_chars)})

        output_config: Dict[str, Any] = {
            "format": {"type": "json_schema", "schema": schema.json_schema(prompt.with_evidence)}
        }
        if cfg.effort:
            output_config["effort"] = cfg.effort

        started = time.perf_counter()
        try:
            # No temperature/top_p: sampling parameters are rejected by the
            # current Claude models. Variation across configs comes from the
            # model and the prompt, which is what we actually want to measure.
            response = self.client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                system=prompt.system(schema),
                messages=[{"role": "user", "content": content}],
                output_config=output_config,
            )
        except self._anthropic.APIError as exc:
            return ExtractionResult(error=f"{type(exc).__name__}: {exc}",
                                    latency_ms=int((time.perf_counter() - started) * 1000))
        latency_ms = int((time.perf_counter() - started) * 1000)

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            return ExtractionResult(error=f"refusal: {getattr(details, 'category', None)}", latency_ms=latency_ms)

        text = next((b.text for b in response.content if b.type == "text"), None)
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        result = ExtractionResult(
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(cfg.model, in_tok or 0, out_tok or 0),
        )
        if text is None:
            result.error = f"no text block in response (stop_reason={getattr(response, 'stop_reason', None)})"
            return result
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            result.error = f"unparseable JSON: {exc}"
            result.raw_output = {"text": text}
            return result

        result.raw_output = payload
        result.values, result.evidence = _split_value_evidence(payload, schema, prompt.with_evidence)
        return result


# --------------------------------------------------------------------------
# Local llama.cpp server (OpenAI-compatible)
# --------------------------------------------------------------------------

DEFAULT_LLAMACPP_BASE_URL = "http://localhost:8080/v1"


class LlamaCppError(RuntimeError):
    pass


def llamacpp_chat_json(
    system: str,
    user: str,
    json_schema: Dict[str, Any],
    *,
    base_url: str = DEFAULT_LLAMACPP_BASE_URL,
    model: str = "local",
    max_tokens: int = 1024,
    schema_name: str = "extraction",
    timeout: float = 900.0,
    retries: int = 2,
    options: Optional[Dict[str, Any]] = None,
    image_data_uri: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """POST one chat completion and return (text, usage, timings).

    Two server-specific details matter and are not optional:

    * `response_format: json_schema` compiles the schema to a GBNF grammar and
      constrains decoding. This is what makes a 4B local model usable as a rater
      at all - without it, a meaningful share of responses are unparseable and
      the "error" would be the harness's, not the model's.
    * `chat_template_kwargs.enable_thinking: false`. Qwen3-family templates
      default to thinking on; the reasoning goes to `reasoning_content`, the
      answer never arrives inside `max_tokens`, and `content` comes back empty.
      At local decode speeds that is also the difference between 30 s and
      several minutes per document.

    `image_data_uri` attaches an image as an OpenAI-style `image_url` part. The
    server only accepts it when started with the model's `--mmproj` projector.
    """
    user_content: Any = user
    if image_data_uri:
        user_content = [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": user},
        ]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
        },
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool((options or {}).get("enable_thinking", False))},
    }
    for key in ("temperature", "top_p", "top_k", "min_p", "seed", "repeat_penalty"):
        if options and options.get(key) is not None:
            payload[key] = options[key]
    if options and options.get("extra_body"):
        payload.update(options["extra_body"])

    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:  # 4xx is a request bug - do not retry
            detail = exc.read()[:400].decode(errors="replace")
            raise LlamaCppError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                raise LlamaCppError(f"{type(exc).__name__} calling {url}: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
    else:  # pragma: no cover - loop always breaks or raises
        raise LlamaCppError(str(last_error))

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    if not text.strip():
        reasoning = (message.get("reasoning_content") or "")[:120]
        raise LlamaCppError(
            f"empty content (finish_reason={choice.get('finish_reason')}); "
            f"reasoning_content={reasoning!r} - raise max_tokens or keep thinking disabled"
        )
    timings = dict(body.get("timings") or {})
    timings["finish_reason"] = choice.get("finish_reason")
    return text, body.get("usage") or {}, timings


class LlamaCppProvider(Provider):
    """Extraction against a local llama.cpp server.

    config params:
        base_url          default http://localhost:8080/v1
        temperature/top_p/top_k/min_p/seed   sampling; unlike the Claude models,
                          local models accept these, which makes "same model,
                          different sampling" a usable rater axis
        enable_thinking   default False (see llamacpp_chat_json)
        timeout, retries

    input_mode image/both needs a vision model served with its --mmproj
    projector (scripts/serve_local.sh qwen-vl) and a downloaded image
    (`ingest --download-images`): the stored image_uri is a signed HF URL that
    expires, so only a local image_path is sent.
    """

    def extract(self, doc: Document, schema: DatasetSchema, cfg: ExtractionConfig) -> ExtractionResult:
        image_data_uri = None
        prompt_doc = doc
        if cfg.input_mode in ("image", "both"):
            if not (doc.image_path and Path(doc.image_path).exists()):
                return ExtractionResult(
                    error=f"config {cfg.id} needs a local image for {doc.doc_id}; run ingest --download-images"
                )
            path = Path(doc.image_path)
            image_data_uri = (f"data:{_media_type(path)};base64,"
                              + base64.standard_b64encode(path.read_bytes()).decode())
            if cfg.input_mode == "image":
                # Drop the OCR text so the prompt points the model at the image.
                prompt_doc = replace(doc, ocr_text=None)
        params = dict(cfg.params or {})
        prompt = get_prompt(cfg.prompt)
        started = time.perf_counter()
        try:
            text, usage, timings = llamacpp_chat_json(
                prompt.system(schema),
                prompt.user(schema, prompt_doc, cfg.max_ocr_chars),
                schema.json_schema(prompt.with_evidence),
                base_url=params.get("base_url", DEFAULT_LLAMACPP_BASE_URL),
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                schema_name=f"{schema.name}_extraction",
                timeout=float(params.get("timeout", 900.0)),
                retries=int(params.get("retries", 2)),
                options=params,
                image_data_uri=image_data_uri,
            )
        except LlamaCppError as exc:
            return ExtractionResult(error=str(exc), latency_ms=int((time.perf_counter() - started) * 1000))
        latency_ms = int((time.perf_counter() - started) * 1000)

        result = ExtractionResult(
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            cost_usd=0.0,  # local weights: the cost is wall-clock and electricity
        )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # A grammar guarantees the *shape* of the JSON, not that the model
            # stops talking: inside a string literal a small model can
            # degenerate (repeating the system prompt, runs of digits) and run
            # to the token cap, leaving the string unterminated. Say which of
            # the two it was, because the fixes differ - raise max_tokens for a
            # genuinely long value, lower temperature for a degeneration.
            truncated = timings.get("finish_reason") == "length"
            reason = (f"output hit max_tokens={cfg.max_tokens} and was cut off mid-value "
                      f"(model degenerated or the value is very long)" if truncated
                      else f"unparseable JSON: {exc}")
            result.error = reason
            result.raw_output = {"text": text, "finish_reason": timings.get("finish_reason")}
            return result
        # Keep the server's own timings: prefill and decode are bound by
        # different things on a laptop, and the report shows them per config.
        result.raw_output = {"values": payload, "timings": {
            k: timings.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second")
        }}
        result.values, result.evidence = _split_value_evidence(payload, schema, prompt.with_evidence)
        return result


# --------------------------------------------------------------------------
# Mock provider
# --------------------------------------------------------------------------

_DIGIT = re.compile(r"\d")


def _corrupt(value: str, field_type: str, rng: random.Random) -> str:
    """Produce a plausible wrong answer - the kind of error a model makes."""
    if field_type == "money":
        digits = [i for i, ch in enumerate(value) if ch.isdigit()]
        if not digits:
            return value
        choice = rng.random()
        if choice < 0.4:                      # picked the subtotal, not the total
            try:
                return f"{max(float(re.sub(r'[^0-9.]', '', value)) - rng.choice([0.6, 1.2, 2.4]), 0):.2f}"
            except ValueError:
                return value
        idx = rng.choice(digits)              # OCR digit confusion
        return value[:idx] + rng.choice("0123456789") + value[idx + 1:]
    if field_type == "date":
        parts = re.split(r"([/\-.])", value)
        if len(parts) >= 5:                   # day/month swap
            parts[0], parts[2] = parts[2], parts[0]
            return "".join(parts)
        return value
    if field_type in ("org", "text"):
        tokens = value.split()
        if len(tokens) > 2:
            return " ".join(tokens[:-1]) if rng.random() < 0.5 else " ".join(tokens[1:])
        return value.title()
    if field_type == "address":
        tokens = value.split(",")
        return ",".join(tokens[:-1]) if len(tokens) > 1 else value[: max(len(value) - 6, 4)]
    if field_type == "phone":
        idx = next((i for i, ch in enumerate(value) if ch.isdigit()), None)
        return value if idx is None else value[:idx] + rng.choice("0123456789") + value[idx + 1:]
    return value + " (?)"


_HALLUCINATIONS = {
    "money": ["0.00", "10.00", "5.30"],
    "date": ["01/01/2019", "12/12/2018"],
    "org": ["CASH SALES", "TAX INVOICE"],
    "address": ["KUALA LUMPUR, MALAYSIA"],
    "phone": ["03-1234 5678"],
    "text": ["N.A."],
    "integer": ["1"],
}


class MockProvider(Provider):
    """Deterministic fake model.

    params:
        error_rate      probability of a wrong-but-populated value
        miss_rate       probability of dropping a populated value to null
        halluc_rate     probability of inventing a value where gold is null
        cluster         configs sharing a cluster draw the *same* corruption,
                        so their errors correlate - the failure mode that
                        makes "they all agree" untrustworthy
        systematic_rate probability that a field is "genuinely ambiguous": every
                        config, whatever its cluster, reads the same wrong value.
                        This is what puts wrong answers inside unanimous items
        seed            base seed
    """

    def extract(self, doc: Document, schema: DatasetSchema, cfg: ExtractionConfig) -> ExtractionResult:
        params = cfg.params or {}
        error_rate = float(params.get("error_rate", 0.12))
        miss_rate = float(params.get("miss_rate", 0.05))
        halluc_rate = float(params.get("halluc_rate", 0.07))
        systematic_rate = float(params.get("systematic_rate", 0.04))
        cluster = str(params.get("cluster", cfg.id))
        base_seed = str(params.get("seed", 17))

        values: Dict[str, Optional[str]] = {}
        for name in schema.field_names:
            ftype = schema.field_type(name)
            gold = doc.gold.get(name)
            # Two independent random streams: one private to this config (does
            # it err at all?) and one shared across the cluster (which wrong
            # value does it produce?).
            own = random.Random(f"{base_seed}|{cfg.id}|{doc.doc_id}|{name}")
            shared = random.Random(f"{base_seed}|{cluster}|{doc.doc_id}|{name}")
            # A third stream shared by *every* config: ambiguous fields where
            # all raters make the same mistake and still look unanimous.
            universal = random.Random(f"{base_seed}|ALL|{doc.doc_id}|{name}")
            if universal.random() < systematic_rate:
                values[name] = (universal.choice(_HALLUCINATIONS.get(ftype, ["?"]))
                                if is_null(gold) else _corrupt(str(gold), ftype, universal))
                continue
            if is_null(gold):
                values[name] = shared.choice(_HALLUCINATIONS.get(ftype, ["?"])) if own.random() < halluc_rate else None
                continue
            roll = own.random()
            if roll < miss_rate:
                values[name] = None
            elif roll < miss_rate + error_rate:
                values[name] = _corrupt(str(gold), ftype, shared)
            else:
                values[name] = str(gold)

        prompt = get_prompt(cfg.prompt)
        evidence = {name: (f"line: {values[name]}" if values[name] else None)
                    for name in schema.field_names} if prompt.with_evidence else {}
        digest = hashlib.sha1(f"{cfg.id}|{doc.doc_id}".encode()).hexdigest()
        return ExtractionResult(
            values=values,
            evidence=evidence,
            raw_output={"mock": True, "values": values, "digest": digest[:8]},
            latency_ms=int(params.get("latency_ms", 5)),
            input_tokens=len((doc.ocr_text or "")) // 4,
            output_tokens=40,
            cost_usd=0.0,
        )


PROVIDERS = {"anthropic": AnthropicProvider, "llamacpp": LlamaCppProvider, "mock": MockProvider}


def build_provider(name: str, **kwargs: Any) -> Provider:
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise KeyError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}") from None
    return cls(**kwargs)
