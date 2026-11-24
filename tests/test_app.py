import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ROOT = Path(__file__).resolve().parents[1]


def test_editor_loads_report_and_filters_issues():
    app = AppTest.from_file(ROOT / "app.py").run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["3", "2", "1", "2"]
    app.selectbox[0].set_value("Без замечаний").run()
    assert app.selectbox[-1].options == ["control-detailed"]
    assert app.success

    report = json.loads((ROOT / "examples" / "synthetic_report.json").read_bytes())
    report[2]["url"] = "https://example.org/vacancy"
    app.file_uploader[0].upload("report.json", json.dumps(report).encode()).run()
    app.selectbox[0].set_value("С замечаниями")
    app.selectbox[1].set_value("location")
    app.selectbox[2].set_value("contradiction")
    app.text_input[0].set_value("OFFICE").run()
    assert not app.exception
    assert app.selectbox[-1].options == ["office-conflict"]
    assert app.get("link_button")[0].url == "https://example.org/vacancy"
    assert [item.value for item in app.text][-4:] == [
        "workPlaceType",
        "Исключительно удалённый режим.",
        "conditions",
        "Очная работа в офисе каждый рабочий день.",
    ]
    assert app.code[0].value.startswith("Укажите один формат")
    app.selectbox[1].set_value("schedule").run()
    assert app.info[0].value == "По этим фильтрам вакансий нет."


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
    for invalid in (b"{", b"[{}]"):
        app.file_uploader[0].upload("invalid.json", invalid).run()
        assert app.error
        assert not app.exception
