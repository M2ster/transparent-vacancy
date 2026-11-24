import json
from pathlib import Path

import httpx
import pytest

from vacancy_audit.__main__ import main
from vacancy_audit.llm import ReviewError, analyze
from vacancy_audit.models import Issue
from vacancy_audit.rules import fields, merge_issues, rules, validate_evidence


def issue(kind, evidence, *, topic="location"):
    return Issue.model_validate(
        {
            "topic": topic,
            "type": kind,
            "evidence": [{"field": path, "quote": value} for path, value in evidence],
            "problem": "Нужна ясность.",
            "recommendation": "Уточните условия.",
        }
    )


@pytest.fixture
def vacancy():
    return {
        "id": "a",
        "job-name": "Архивист",
        "duty": "Принимать документы.",
        "requirement": {"experience": 0, "education": "Не указано"},
        "salary_min": 70000,
        "salary_max": 0,
        "schedule": "Пн–пт 09:00–18:00",
        "employment": "Полная занятость",
        "addresses": {
            "address": [{"location": "Калининград", "contact_person": "Не передавать"}]
        },
    }


def test_rules_do_not_treat_zero_or_unfilled_optional_field_as_issue(vacancy):
    assert rules(fields(vacancy)) == []
    vacancy.update(salary_min=0, salary_max=0, currency="RUB")
    del vacancy["schedule"]
    assert {(item.topic, item.type) for item in rules(fields(vacancy))} == {
        ("salary", "missing"),
        ("schedule", "missing"),
    }
    vacancy["conditions"] = "Оплата 45000 рублей. Работа по будням с 9 до 18."
    assert rules(fields(vacancy)) == []


def test_inverted_salary_and_deduplication(vacancy):
    vacancy.update(salary_min=90000, salary_max=60000)
    report = rules(fields(vacancy))
    assert len(report) == 1
    assert report[0].type == "contradiction"
    assert [citation.quote for citation in report[0].evidence] == ["90000", "60000"]
    assert merge_issues(report + report) == report


def test_evidence_uses_exact_paths_and_quotes(vacancy):
    data = fields(
        vacancy | {"workPlaceType": "Удалённо", "conditions": "Работа в офисе."}
    )
    validate_evidence(
        issue("ambiguity", [("addresses.address[0].location", "Калининград")]), data
    )
    validate_evidence(
        issue(
            "contradiction",
            [
                ("workPlaceType", "Удалённо"),
                ("conditions", "в офисе"),
            ],
        ),
        data,
    )
    for bad in [
        issue("contradiction", [("workPlaceType", "Удалённо")]),
        issue(
            "contradiction",
            [("workPlaceType", "Удалённо"), ("workPlaceType", "далённо")],
        ),
        issue("ambiguity", [("addresses.address[1].location", "Калининград")]),
        issue("ambiguity", [("addresses.address[0].location", "Москва")]),
        issue("ambiguity", [("salary_min", "700")], topic="salary"),
        issue("missing", [("requirement.experience", "")], topic="requirements"),
        issue("missing", [("addresses.address[0].location", "")]),
    ]:
        with pytest.raises(ValueError):
            validate_evidence(bad, data)
    validate_evidence(issue("missing", [("schedule", "")], topic="schedule"), {})


def ollama_response(issues):
    return httpx.Response(
        200, json={"message": {"content": json.dumps({"issues": issues})}}
    )


def test_ollama_retries_invalid_evidence_and_uses_valid_response(vacancy):
    calls = []
    response_issue = issue("ambiguity", [("duty", "Принимать")], topic="duties")

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.url.path == "/api/chat"
        assert payload["think"] is False
        assert payload["options"]["num_ctx"] == 4096
        assert payload["options"]["temperature"] == 0
        assert payload["keep_alive"] == "30m"
        assert payload["format"]["properties"]["issues"]
        text = payload["messages"][1]["content"]
        assert "requirement.experience" in text
        assert "addresses.address[0].location" in text
        assert "contact_person" not in text
        if len(calls) == 1:
            return ollama_response(
                [issue("ambiguity", [("duty", "Отправлять")]).model_dump()]
            )
        evidence_options = payload["format"]["$defs"]["Evidence"]["anyOf"]
        assert {
            "field": {"const": "duty"},
            "quote": {"enum": ["Принимать документы."]},
        } in [entry["properties"] for entry in evidence_options]
        return ollama_response([response_issue.model_dump()])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze(
            client,
            fields(vacancy),
            url="http://ollama.test",
            model="qwen3:4b",
            context=4096,
            predict=500,
        )
    assert result == [response_issue]
    assert len(calls) == 2


def test_context_overflow_stops_before_ollama():
    def handler(_):
        pytest.fail("Ollama must not receive truncated input")

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ReviewError, match="бюджет контекста"),
    ):
        analyze(
            client,
            {"duty": "длинный текст " * 1500},
            url="http://ollama.test",
            model="qwen3:4b",
            context=4096,
            predict=500,
        )


def test_local_pipeline_all_unique_rows_and_atomic_failure(
    monkeypatch, tmp_path: Path, vacancy
):
    old_client = httpx.Client
    calls = []
    fail = False

    def handler(request):
        assert request.url.host == "ollama.test"
        calls.append(request)
        if fail:
            return httpx.Response(200, json={"message": {"content": "{}"}})
        return ollama_response([])

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: old_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    source, output = tmp_path / "input.json", tmp_path / "report.json"
    source.write_text(
        json.dumps(
            [
                vacancy,
                vacancy | {"id": "b", "vac_url": "https://example.org/v"},
                vacancy,
            ]
        )
    )
    arguments = [
        "--input",
        str(source),
        "--output",
        str(output),
        "--ollama-url",
        "http://ollama.test",
        "--limit",
        "1",
    ]
    assert main(arguments) == 0
    previous = output.read_bytes()
    assert json.loads(previous) == [
        {"id": "a", "url": None, "issues": []},
        {"id": "b", "url": "https://example.org/v", "issues": []},
    ]
    assert len(calls) == 2
    fail = True
    assert main(arguments) == 1
    assert len(calls) == 4
    assert output.read_bytes() == previous
    source.write_text('{"results": {}}')
    assert main(arguments) == 1
    assert output.read_bytes() == previous


def test_api_failure_keeps_previous_report(monkeypatch, tmp_path):
    real_client = httpx.Client

    def timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(timeout), **kwargs),
    )
    output = tmp_path / "report.json"
    output.write_text("[]\n")
    assert main(["--api-url", "https://portal.test", "--output", str(output)]) == 1
    assert output.read_text() == "[]\n"
