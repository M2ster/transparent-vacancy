import json
from pathlib import Path

import httpx
import pytest

from vacancy_audit.__main__ import main
from vacancy_audit.llm import ReviewError, analyze
from vacancy_audit.models import Issue
from vacancy_audit.rules import (
    combine_issues,
    fields,
    merge_issues,
    missing_issue,
    rules,
    validate_evidence,
)


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
    duty = vacancy.pop("duty")
    assert {(item.topic, item.type) for item in rules(fields(vacancy))} == {
        ("duties", "missing"),
        ("salary", "missing"),
        ("schedule", "missing"),
    }
    vacancy.update(
        duty=duty, conditions="Оплата 45000 рублей. Работа по будням с 9 до 18."
    )
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


@pytest.mark.parametrize(
    "path, quote, value",
    [
        ("duty", "", "Не указано"),
        ("duty", "Вести архив.", "Вести архив."),
        ("salary_min", "0", 0),
    ],
)
def test_missing_rejects_nonempty_quotes_and_unrelated_fields(path, quote, value):
    with pytest.raises(ValueError):
        validate_evidence(
            issue("missing", [(path, quote)], topic="salary"), {path: value}
        )


@pytest.mark.parametrize(
    "topic, path, data",
    [
        ("salary", "salary_min", {"salary_min": 0, "salary_currency": "RUB"}),
        (
            "requirements",
            "requirement.education",
            {"requirement": {"education": "н/д"}},
        ),
        ("location", "addresses.address[2].location", {}),
        ("schedule", "schedule", {"schedule": "не указан"}),
    ],
)
def test_missing_accepts_absent_or_placeholder_field(topic, path, data):
    validate_evidence(issue("missing", [(path, "")], topic=topic), data)


@pytest.mark.parametrize("text_salary", ["", "Оплата 45000 рублей в месяц."])
def test_shared_prose_leaves_missing_decision_to_one_model_call(text_salary):
    text = f"Предоставляем ДМС. {text_salary}".strip()
    data = {"job-name": "Архивист", "conditions": text}
    topics = ["duties", "requirements", "salary", "schedule", "location", "employment"]
    calls = []

    def handler(request):
        calls.append(request)
        payload = json.loads(request.content)
        schema = payload["format"]["properties"]["coverage"]
        assert schema["required"] == topics
        assert None in schema["properties"]["salary"]["enum"]
        coverage = dict.fromkeys(topics)
        coverage["salary"] = text_salary or None
        return ollama_response([], coverage=coverage)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        inferred = analyze(
            client,
            data,
            url="http://ollama.test",
            model="qwen3:4b",
            context=None,
            predict=900,
        )
    assert len(calls) == 1
    assert combine_issues(data, inferred) == [
        missing_issue(topic)
        for topic in topics
        if not (text_salary and topic == "salary")
    ]


def ollama_response(issues, **parts):
    return httpx.Response(
        200, json={"message": {"content": json.dumps({"issues": issues, **parts})}}
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


@pytest.mark.parametrize("context, length", [(6144, 3600), (8192, 6000), (None, 10000)])
def test_context_overflow_stops_before_ollama(context, length):
    def handler(_):
        pytest.fail("Ollama must not receive truncated input")

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ReviewError, match="бюджет контекста"),
    ):
        analyze(
            client,
            {"duty": "a" * length},
            url="http://ollama.test",
            model="qwen3:4b",
            context=context,
            predict=900,
        )


@pytest.mark.parametrize(
    "length, context, expected",
    [(1500, None, 6144), (3600, None, 8192), (6000, None, 12288), (6000, 10000, 10000)],
)
def test_context_selects_smallest_supported_size_or_explicit_limit(
    vacancy, length, context, expected
):
    calls = []
    data = fields(vacancy | {"duty": "a" * length})

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload["options"]["num_ctx"])
        assert json.loads(payload["messages"][1]["content"])["duty"] == "a" * length
        return ollama_response([])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            analyze(
                client,
                data,
                url="http://ollama.test",
                model="qwen3:4b",
                context=context,
                predict=900,
            )
            == []
        )
    assert calls == [expected]


@pytest.mark.parametrize(
    "schedule", ["Полная занятость, 4/3", "Смены 3/3, 07:30–16:30"]
)
def test_specific_schedule_and_salary_are_not_ambiguous(vacancy, schedule):
    data = fields(vacancy | {"schedule": schedule})
    inferred = [
        issue("ambiguity", [("schedule", schedule)], topic="schedule"),
        issue("ambiguity", [("salary_max", "0")], topic="salary"),
        issue(
            "contradiction",
            [("salary_min", "70000"), ("salary_max", "0")],
            topic="salary",
        ),
    ]
    assert combine_issues(data, inferred) == []


def test_generic_shift_and_specific_flexibility(vacancy):
    assert [
        (item.topic, item.type)
        for item in rules(fields(vacancy | {"schedule": "Сменный график"}))
    ] == [
        ("schedule", "ambiguity"),
    ]
    data = fields(
        vacancy
        | {
            "schedule": "Гибкий график",
            "conditions": "Начало работы между 07:00 и 10:00.",
        }
    )
    assert rules(data) == []
    inferred = issue("ambiguity", [("schedule", "Гибкий график")], topic="schedule")
    assert combine_issues(data, [inferred]) == []


def test_clear_conflicts_use_both_fields_and_keep_the_right_topic(vacancy):
    data = fields(
        vacancy
        | {
            "requirement": {
                "experience": 0,
                "qualification": "Опыт монтажа от 18 месяцев.",
            },
            "workPlaceType": "Исключительно удалённый режим.",
            "conditions": "Очная работа каждый рабочий день.",
        }
    )
    inferred = [
        issue(
            "ambiguity",
            [
                ("workPlaceType", "удалённый режим"),
                ("conditions", "каждый рабочий день"),
            ],
            topic="duties",
        )
    ]
    items = combine_issues(data, inferred)
    assert [(item.topic, item.type) for item in items] == [
        ("requirements", "contradiction"),
        ("location", "contradiction"),
    ]
    for item in items:
        validate_evidence(item, data)


def test_optional_experience_and_hybrid_onboarding_do_not_conflict(vacancy):
    data = fields(
        vacancy
        | {
            "requirement": {
                "education": "Не указано",
                "experience": 0,
                "qualification": "Желателен стаж от 2 лет.",
            },
            "workPlaceType": "После вводной недели полностью дистанционно.",
            "conditions": "Во время обучения ежедневно присутствовать в офисе.",
        }
    )
    assert rules(data) == []


def test_office_address_disclosed_later_is_location_ambiguity(vacancy):
    deferred = "Адрес офиса сообщим позже."
    data = fields(vacancy | {"conditions": deferred})
    inferred = issue(
        "contradiction",
        [("conditions", deferred), ("employment", "Полная занятость")],
        topic="employment",
    )
    assert [(item.topic, item.type) for item in combine_issues(data, [inferred])] == [
        ("location", "ambiguity")
    ]


@pytest.mark.parametrize(
    ("keep_valid", "retry_fails"), [(True, False), (False, False), (True, True)]
)
def test_retry_does_not_erase_verified_issues(vacancy, keep_valid, retry_fails, caplog):
    valid = issue("ambiguity", [("duty", "Принимать")], topic="duties")
    first = [valid.model_dump()] if keep_valid else []
    responses = iter(
        [
            ollama_response(first + [{"topic": "bad"}]),
            httpx.Response(503) if retry_fails else ollama_response([]),
        ]
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: next(responses))
    ) as client:
        if keep_valid:
            assert analyze(
                client,
                fields(vacancy),
                url="http://ollama.test",
                model="qwen3:4b",
                context=4096,
                predict=500,
                vacancy_id="test-vacancy",
            ) == [valid]
        else:
            with pytest.raises(ReviewError, match="дважды"):
                analyze(
                    client,
                    fields(vacancy),
                    url="http://ollama.test",
                    model="qwen3:4b",
                    context=4096,
                    predict=500,
                    vacancy_id="test-vacancy",
                )
    assert ("частично" in caplog.text) == keep_valid
    assert ("test-vacancy" in caplog.text) == keep_valid


@pytest.mark.parametrize("retry_case", ["fixed", "partial", "duplicate", "timeout"])
def test_retry_only_warns_if_findings_remain_unconfirmed(vacancy, retry_case, caplog):
    data = fields(vacancy)
    first = issue("ambiguity", [("duty", "Принимать")], topic="duties")
    replacement = issue("ambiguity", [("schedule", "09:00")], topic="schedule")
    bad = {"topic": "bad"}
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return ollama_response([first.model_dump(), bad])
        if retry_case == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if retry_case == "duplicate":
            return ollama_response([first.model_dump()])
        retry = [replacement.model_dump()]
        if retry_case == "partial":
            retry.append(bad)
        return ollama_response(retry)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze(
            client,
            data,
            url="http://ollama.test",
            model="qwen3:4b",
            context=None,
            predict=900,
            vacancy_id="test-vacancy",
        )
    assert result == (
        [first, replacement] if retry_case in {"fixed", "partial"} else [first]
    )
    assert ("test-vacancy" in caplog.text and "частично" in caplog.text) == (
        retry_case != "fixed"
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
                vacancy
                | {
                    "id": "b",
                    "job-name": "  Реставратор  ",
                    "vac_url": "https://example.org/v",
                },
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
        {"id": "a", "url": None, "job_title": "Архивист", "issues": []},
        {
            "id": "b",
            "url": "https://example.org/v",
            "job_title": "Реставратор",
            "issues": [],
        },
    ]
    assert len(calls) == 2
    fail = True
    assert main(arguments) == 1
    assert len(calls) == 4
    assert output.read_bytes() == previous
    source.write_text(json.dumps([vacancy | {"duty": "a" * 15000}]))
    assert main(arguments) == 1
    assert len(calls) == 4
    assert output.read_bytes() == previous
    source.write_text('{"results": {}}')
    assert main(arguments) == 1
    assert output.read_bytes() == previous


def test_api_report_uses_source_job_title(monkeypatch, tmp_path, vacancy):
    real_client = httpx.Client

    def handler(request):
        if request.url.host == "portal.test":
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "meta": {"total": 2},
                    "results": {
                        "vacancies": [
                            {"vacancy": vacancy},
                            {"vacancy": vacancy | {"id": "b", "job-name": "  "}},
                        ]
                    },
                },
            )
        assert request.url.host == "ollama.test"
        return ollama_response([])

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--api-url",
                "https://portal.test",
                "--output",
                str(output),
                "--ollama-url",
                "http://ollama.test",
            ]
        )
        == 0
    )
    assert [row["job_title"] for row in json.loads(output.read_bytes())] == [
        "Архивист",
        None,
    ]


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
