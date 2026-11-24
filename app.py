from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st
from pydantic import TypeAdapter, ValidationError

from vacancy_audit.models import VacancyReport

TOPICS = {
    "job_title": "Должность",
    "duties": "Обязанности",
    "requirements": "Требования",
    "salary": "Оплата",
    "schedule": "График",
    "location": "Место работы",
    "employment": "Занятость",
}
TYPES = {
    "missing": "Нет сведений",
    "ambiguity": "Неоднозначность",
    "contradiction": "Противоречие",
}
EXAMPLE = Path(__file__).resolve().parent / "examples" / "synthetic_report.json"


def is_web_url(url: str) -> bool:
    if "\\" in url or any(
        character.isspace() or ord(character) < 32 for character in url
    ):
        return False
    try:
        parsed = urlsplit(url)
        parsed.port
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def main() -> None:
    st.set_page_config(page_title="Прозрачная вакансия", layout="wide")
    st.title("Прозрачная вакансия")
    uploaded = st.file_uploader("Загрузите JSON-отчёт", type="json")
    if uploaded is None:
        st.caption("Открыт демонстрационный отчёт.")
    try:
        raw = EXAMPLE.read_bytes() if uploaded is None else uploaded.getvalue()
        reports = TypeAdapter(list[VacancyReport]).validate_json(raw)
    except OSError:
        st.error("Не удалось прочитать демонстрационный отчёт.")
        return
    except ValidationError:
        st.error("Файл должен содержать корректный JSON-отчёт проверки вакансий.")
        return
    if not reports:
        st.info("Отчёт пуст.")
        return

    counts = Counter(issue.topic for report in reports for issue in report.issues)
    with_issues = sum(bool(report.issues) for report in reports)
    columns = st.columns(4)
    columns[0].metric("Вакансии", len(reports))
    columns[1].metric("С замечаниями", with_issues)
    columns[2].metric("Без замечаний", len(reports) - with_issues)
    columns[3].metric("Замечания", counts.total())
    with st.expander("Замечания по темам"):
        st.table(
            [
                {"Тема": label, "Замечания": counts[topic]}
                for topic, label in TOPICS.items()
            ]
        )

    filters = st.columns(4)
    search = filters[0].text_input("Поиск по ID").strip().casefold()
    status = filters[1].selectbox(
        "Наличие замечаний", ["Все вакансии", "С замечаниями", "Без замечаний"]
    )
    topic = filters[2].selectbox(
        "Тема", [None, *TOPICS], format_func=lambda value: TOPICS.get(value, "Все темы")
    )
    kind = filters[3].selectbox(
        "Тип", [None, *TYPES], format_func=lambda value: TYPES.get(value, "Все типы")
    )

    matches = []
    for report in reports:
        if search not in report.id.casefold():
            continue
        if (status == "С замечаниями" and not report.issues) or (
            status == "Без замечаний" and report.issues
        ):
            continue
        issues = [
            issue
            for issue in report.issues
            if (topic is None or issue.topic == topic)
            and (kind is None or issue.type == kind)
        ]
        if issues or (topic is None and kind is None):
            matches.append((report, issues))

    if not matches:
        st.info("По этим фильтрам вакансий нет.")
        return

    index = st.selectbox(
        "ID вакансии",
        range(len(matches)),
        format_func=lambda index: matches[index][0].id,
    )
    report, issues = matches[index]
    if report.url and is_web_url(report.url):
        st.link_button("Открыть вакансию", report.url)
    if not issues:
        st.success("В отчёте по этой вакансии замечаний нет.")
        return
    for issue in issues:
        with st.container(border=True):
            st.subheader(f"{TOPICS[issue.topic]} · {TYPES[issue.type]}")
            st.text(issue.problem)
            for evidence in issue.evidence:
                st.text(evidence.field)
                st.text(evidence.quote or "Сведения отсутствуют.")
            st.markdown("**Рекомендация редактору**")
            st.code(issue.recommendation, language=None, wrap_lines=True)


if __name__ == "__main__":
    main()
