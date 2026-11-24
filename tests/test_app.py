import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ROOT = Path(__file__).resolve().parents[1]


def test_editor_loads_report_and_filters_issues():
    app = AppTest.from_file(ROOT / "app.py").run()
    assert not app.exception
    assert app.caption[-1].value == "Найдено 3 из 3 вакансий"

    location = {
        "topic": "location",
        "type": "contradiction",
        "evidence": [{"field": "workPlaceType", "quote": "Удалённый режим."}],
        "problem": "Форматы работы расходятся.",
        "recommendation": "Укажите один формат.",
    }
    schedule = {
        "topic": "schedule",
        "type": "ambiguity",
        "evidence": [{"field": "schedule", "quote": "Сменный график"}],
        "problem": "Часы работы неясны.",
        "recommendation": "Укажите время смен.",
    }
    report = [
        {"id": "control", "url": None, "issues": []},
        {"id": "shift", "url": None, "job_title": " ", "issues": [schedule]},
        {
            "id": "office",
            "url": "https://example.org/vacancy",
            "job_title": "Архивист",
            "issues": [location, schedule],
        },
    ]
    app.file_uploader[0].upload("report.json", json.dumps(report).encode()).run()
    assert [metric.value for metric in app.metric] == ["3", "2", "1", "3"]
    assert [card.label for card in app.expander[1:]] == [
        "ID control · Без замечаний",
        "ID shift · Замечаний: 1",
        "Архивист · ID office · Замечаний: 2",
    ]
    assert all(not card.proto.expanded for card in app.expander[1:])

    app.selectbox[0].set_value("Без замечаний").run()
    assert [card.label for card in app.expander[1:]] == ["ID control · Без замечаний"]
    assert app.caption[-1].value == "Найдено 1 из 3 вакансий"
    assert app.success

    app.selectbox[0].set_value("С замечаниями").run()
    assert len(app.expander[1:]) == 2
    assert [metric.value for metric in app.metric] == ["3", "2", "1", "3"]

    app.text_input[0].set_value("OFFICE").run()
    assert len(app.expander[1:]) == 1
    assert app.expander[1].label.startswith("Архивист")

    app.selectbox[0].set_value("С замечаниями")
    app.selectbox[1].set_value("location")
    app.selectbox[2].set_value("contradiction")
    app.text_input[0].set_value("АРХИВ").run()
    assert not app.exception
    assert len(app.expander[1:]) == 1
    assert app.get("link_button")[0].url == "https://example.org/vacancy"
    assert [item.value for item in app.text][1::3] == ["workPlaceType", "schedule"]
    assert [item.value for item in app.text][2::3] == [
        "Удалённый режим.",
        "Сменный график",
    ]
    assert [item.value for item in app.code] == [
        "Укажите один формат.",
        "Укажите время смен.",
    ]
    app.selectbox[1].set_value("schedule").run()
    assert app.info[0].value == "По этим фильтрам вакансий нет."
    assert app.caption[-1].value == "Найдено 0 из 3 вакансий"
    assert not app.expander[1:]


def test_editor_handles_missing_evidence_and_bad_uploads():
    app = AppTest.from_file(ROOT / "app.py").run()
    report = [
        {
            "id": "missing-salary",
            "url": "javascript:alert(1)",
            "issues": [
                {
                    "topic": "salary",
                    "type": "missing",
                    "evidence": [{"field": "salary", "quote": ""}],
                    "problem": "Оплата не указана.",
                    "recommendation": "Добавьте вилку оплаты.",
                }
            ],
        }
    ]
    app.file_uploader[0].upload("report.json", json.dumps(report).encode()).run()
    assert not app.exception
    assert not app.get("link_button")
    assert [item.value for item in app.text][-2:] == ["salary", "Сведения отсутствуют."]

    app.file_uploader[0].upload("empty.json", b"[]").run()
    assert app.info[0].value == "Отчёт пуст."
    for invalid in (
        b"{",
        b"[{}]",
        b'[{"id":"a","url":null,"issues":[],"job_title":7}]',
    ):
        app.file_uploader[0].upload("invalid.json", invalid).run()
        assert app.error
        assert not app.exception
