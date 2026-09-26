"""Local llama.cpp provider - request shape and failure handling.

Stubs the HTTP layer: these run with no server, and they pin the two details
that silently produce garbage when they are wrong (grammar-constrained JSON,
and thinking left enabled on a Qwen-family chat template).
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from agreement_eval.config import ExtractionConfig, JudgeConfig
from agreement_eval.documents import Document
from agreement_eval.extract import providers
from agreement_eval.extract.providers import LlamaCppError, LlamaCppProvider, llamacpp_chat_json
from agreement_eval.judge import LLMJudge, JudgeItem
from agreement_eval.schemas import get_schema

SCHEMA = get_schema("sroie")
DOC = Document(
    doc_id="d1", dataset="sroie",
    gold={"company": "ACME SDN BHD", "date": "15/01/2019", "address": "1 MAIN ST", "total": "9.00"},
    ocr_text="ACME SDN BHD\n1 MAIN ST\nDATE: 15/01/2019\nTOTAL 9.00",
)


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub(monkeypatch, payload, captured=None):
    def fake_urlopen(request, timeout=None):
        if captured is not None:
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
        return _Response(payload)
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)


def _completion(content, usage=None):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 500, "completion_tokens": 70},
        "timings": {"prompt_n": 500, "prompt_per_second": 22.5,
                    "predicted_n": 70, "predicted_per_second": 4.6},
    }


def test_request_uses_grammar_constrained_json_and_disables_thinking(monkeypatch):
    captured = {}
    values = {"company": "ACME SDN BHD", "date": "15/01/2019", "address": "1 MAIN ST", "total": "9.00"}
    _stub(monkeypatch, _completion(json.dumps(values)), captured)

    cfg = ExtractionConfig(id="q1", provider="llamacpp", model="Qwen3.5-4B",
                           params={"temperature": 0.8, "top_p": 0.95, "seed": 7})
    result = LlamaCppProvider().extract(DOC, SCHEMA, cfg)

    body = captured["body"]
    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"]["required"] == list(SCHEMA.field_names)
    # Thinking on a Qwen chat template returns an empty `content`; never default it on.
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    # Sampling knobs are forwarded - unlike the Claude path, local models take them.
    assert body["temperature"] == 0.8 and body["top_p"] == 0.95 and body["seed"] == 7
    assert result.ok and result.values == values
    assert result.cost_usd == 0.0
    assert result.raw_output["timings"]["predicted_per_second"] == 4.6


def test_base_url_is_configurable(monkeypatch):
    captured = {}
    _stub(monkeypatch, _completion(json.dumps({f: None for f in SCHEMA.field_names})), captured)
    cfg = ExtractionConfig(id="q2", provider="llamacpp",
                           params={"base_url": "http://localhost:8081/v1"})
    LlamaCppProvider().extract(DOC, SCHEMA, cfg)
    assert captured["url"] == "http://localhost:8081/v1/chat/completions"


def test_empty_content_reports_the_thinking_trap(monkeypatch):
    payload = _completion("")
    payload["choices"][0]["message"]["reasoning_content"] = "Thinking Process: ..."
    payload["choices"][0]["finish_reason"] = "length"
    _stub(monkeypatch, payload)
    result = LlamaCppProvider().extract(DOC, SCHEMA, ExtractionConfig(id="q3", provider="llamacpp"))
    assert not result.ok
    assert "empty content" in result.error and "thinking" in result.error


def test_truncated_output_is_reported_as_a_token_cap_not_a_parse_bug(monkeypatch):
    # Valid prefix, unterminated string, finish_reason=length: what a degenerating
    # small model produces inside a JSON grammar.
    payload = _completion('{"company": "ACME SDN BHD", "date": "000000000000')
    payload["choices"][0]["finish_reason"] = "length"
    _stub(monkeypatch, payload)
    cfg = ExtractionConfig(id="q7", provider="llamacpp", max_tokens=400)
    result = LlamaCppProvider().extract(DOC, SCHEMA, cfg)
    assert not result.ok
    assert "max_tokens=400" in result.error
    assert result.raw_output["finish_reason"] == "length"


def test_http_error_is_not_retried_and_surfaces_the_server_message(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                                     __import__("io").BytesIO(b'{"error":{"message":"grammar failed"}}'))
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)

    result = LlamaCppProvider().extract(DOC, SCHEMA, ExtractionConfig(id="q4", provider="llamacpp"))
    assert calls["n"] == 1  # a 400 is a bug in the request, not a transient failure
    assert not result.ok and "grammar failed" in result.error


def test_connection_error_is_retried_then_reported(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)

    cfg = ExtractionConfig(id="q5", provider="llamacpp", params={"retries": 2})
    result = LlamaCppProvider().extract(DOC, SCHEMA, cfg)
    assert calls["n"] == 3
    assert not result.ok and "connection refused" in result.error


def test_vision_config_without_local_image_is_rejected_rather_than_downgraded(monkeypatch):
    cfg = ExtractionConfig(id="q6", provider="llamacpp", input_mode="image")
    result = LlamaCppProvider().extract(DOC, SCHEMA, cfg)
    assert not result.ok and "--download-images" in result.error


@pytest.mark.parametrize("input_mode, ocr_in_prompt", [("image", False), ("both", True)])
def test_vision_config_sends_image_as_data_uri(monkeypatch, tmp_path, input_mode, ocr_in_prompt):
    captured = {}
    values = {"company": "ACME SDN BHD", "date": "15/01/2019", "address": "1 MAIN ST", "total": "9.00"}
    _stub(monkeypatch, _completion(json.dumps(values)), captured)
    image = tmp_path / "d1.jpg"
    image.write_bytes(b"\xff\xd8fake-jpeg")
    doc = Document(doc_id="d1", dataset="sroie", gold=DOC.gold, ocr_text=DOC.ocr_text, image_path=str(image))

    cfg = ExtractionConfig(id="q7", provider="llamacpp", input_mode=input_mode)
    result = LlamaCppProvider().extract(doc, SCHEMA, cfg)

    image_part, text_part = captured["body"]["messages"][1]["content"]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert ("TOTAL 9.00" in text_part["text"]) is ocr_in_prompt
    assert result.ok and result.values["total"] == "9.00"


def test_local_judge_speaks_the_same_endpoint(monkeypatch):
    captured = {}
    verdicts = {"verdicts": [{"id": "p000000", "equivalent": True, "confidence": 0.9, "reason": "same amount"}]}
    _stub(monkeypatch, _completion(json.dumps(verdicts)), captured)

    judge = LLMJudge(JudgeConfig(provider="llamacpp", model="Qwen3.5-4B",
                                 params={"base_url": "http://localhost:8080/v1"}))
    out = judge.judge_batch([JudgeItem("p000000", "d1", "total", "money", "pair", "a", "b", "9.00", "RM9")])
    assert out[0]["equivalent"] is True
    assert captured["body"]["response_format"]["json_schema"]["name"] == "judge_verdicts"
    assert captured["body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_chat_helper_raises_on_unreachable_server(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("nope")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LlamaCppError):
        llamacpp_chat_json("sys", "user", {"type": "object"}, retries=0)
