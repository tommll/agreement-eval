import json
from types import SimpleNamespace

import pytest

from agreement_eval.config import ExtractionConfig
from agreement_eval.documents import Document
from agreement_eval.extract.providers import AnthropicProvider, MockProvider, estimate_cost
from agreement_eval.schemas import get_schema

SCHEMA = get_schema("sroie")
DOC = Document(
    doc_id="d1", dataset="sroie",
    gold={"company": "ACME SDN BHD", "date": "15/01/2019", "address": "1 MAIN ST", "total": "9.00"},
    ocr_text="ACME SDN BHD\n1 MAIN ST\nDATE: 15/01/2019\nTOTAL 9.00",
)


class _StubMessages:
    def __init__(self, payload, usage=(100, 20)):
        self.payload = payload
        self.usage = usage
        self.captured = None

    def create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]),
        )


class _StubClient:
    def __init__(self, payload):
        self.messages = _StubMessages(payload)


def test_anthropic_provider_request_shape():
    client = _StubClient({"company": "ACME SDN BHD", "date": "15/01/2019",
                          "address": "1 MAIN ST", "total": "9.00"})
    cfg = ExtractionConfig(id="c1", model="claude-opus-5", prompt="detailed", effort="medium")
    result = AnthropicProvider(client=client).extract(DOC, SCHEMA, cfg)

    sent = client.messages.captured
    assert sent["model"] == "claude-opus-5"
    assert sent["output_config"]["effort"] == "medium"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["format"]["schema"]["required"] == list(SCHEMA.field_names)
    # Sampling parameters are rejected by current models and must never be sent.
    assert "temperature" not in sent and "top_p" not in sent
    assert result.ok and result.values["total"] == "9.00"
    assert result.cost_usd == pytest.approx(estimate_cost("claude-opus-5", 100, 20))


def test_anthropic_provider_effort_omitted_when_none():
    client = _StubClient({f: None for f in SCHEMA.field_names})
    cfg = ExtractionConfig(id="c1", model="claude-haiku-4-5", effort=None)
    AnthropicProvider(client=client).extract(DOC, SCHEMA, cfg)
    assert "effort" not in client.messages.captured["output_config"]


def test_evidence_prompt_unwraps_value_and_evidence():
    payload = {f: {"value": "x", "evidence": f"line for {f}"} for f in SCHEMA.field_names}
    client = _StubClient(payload)
    cfg = ExtractionConfig(id="c1", prompt="evidence")
    result = AnthropicProvider(client=client).extract(DOC, SCHEMA, cfg)
    assert result.values["total"] == "x"
    assert result.evidence["total"] == "line for total"
    assert client.messages.captured["output_config"]["format"]["schema"]["properties"]["total"]["type"] == "object"


def test_null_like_model_output_becomes_none():
    client = _StubClient({"company": "N/A", "date": "", "address": None, "total": "9.00"})
    result = AnthropicProvider(client=client).extract(DOC, SCHEMA, ExtractionConfig(id="c1"))
    assert result.values["company"] is None and result.values["date"] is None
    assert result.values["address"] is None and result.values["total"] == "9.00"


def test_image_config_without_an_image_is_an_error_not_a_silent_text_call():
    client = _StubClient({})
    cfg = ExtractionConfig(id="c1", input_mode="image")
    result = AnthropicProvider(client=client).extract(DOC, SCHEMA, cfg)
    assert not result.ok and "image" in result.error


def test_mock_provider_is_deterministic():
    cfg = ExtractionConfig(id="m1", provider="mock", params={"error_rate": 0.5})
    a = MockProvider().extract(DOC, SCHEMA, cfg)
    b = MockProvider().extract(DOC, SCHEMA, cfg)
    assert a.values == b.values


def test_mock_cluster_errors_are_correlated():
    doc = Document(doc_id="dX", dataset="sroie", gold=DOC.gold, ocr_text=DOC.ocr_text)
    shared = {"error_rate": 1.0, "miss_rate": 0.0, "halluc_rate": 0.0,
              "systematic_rate": 0.0, "cluster": "same"}
    a = MockProvider().extract(doc, SCHEMA, ExtractionConfig(id="a", provider="mock", params=shared))
    b = MockProvider().extract(doc, SCHEMA, ExtractionConfig(id="b", provider="mock", params=shared))
    assert a.values == b.values                    # same cluster -> same wrong answers
    assert a.values["company"] != doc.gold["company"]


def test_mock_systematic_errors_hit_every_config():
    doc = Document(doc_id="dY", dataset="sroie", gold=DOC.gold, ocr_text=DOC.ocr_text)
    params = {"error_rate": 0.0, "miss_rate": 0.0, "halluc_rate": 0.0, "systematic_rate": 1.0}
    values = [
        MockProvider().extract(doc, SCHEMA, ExtractionConfig(
            id=cid, provider="mock", params={**params, "cluster": cid})).values
        for cid in ("a", "b", "c")
    ]
    # Every config reads the same ambiguous field the same wrong way: the items
    # look unanimous and are unanimously wrong.
    assert values[0] == values[1] == values[2]
    assert values[0]["company"] != doc.gold["company"]
