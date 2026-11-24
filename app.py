import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st
from pydantic import TypeAdapter, ValidationError

from vacancy_audit.models import VacancyReport
from vacancy_audit.rules import FIELDS

TOPICS = {
    "job_title": "Должность",
    "duties": "Обязанности",
    "requirements": "Требования",
    "salary": "Оплата",
    "schedule": "График",
    "location": "Место работы",
    "employment": "Занятость",
}
FIELD_LABELS = {
    field: TOPICS[topic] for topic, fields in FIELDS.items() for field in fields
}
FIELD_LABELS.update(
    {
        "experience": "Опыт работы",
        "education": "Образование",
        "qualification": "Квалификационные требования",
        "skills": "Навыки",
        "requirement.experience": "Опыт работы",
        "requirement.education": "Образование",
        "requirement.qualification": "Квалификационные требования",
        "requirement.skills": "Навыки",
        "salary_min": "Минимальная оплата",
        "salary_max": "Максимальная оплата",
        "salary_currency": "Валюта оплаты",
        "currency": "Валюта оплаты",
        "schedule": "График работы",
        "work_schedule": "График работы",
        "shift": "Смены",
        "work_hours": "Часы работы",
        "addresses": "Адреса работы",
        "addresses.address": "Адрес работы",
        "addresses.address.address": "Адрес работы",
        "addresses.address.location": "Адрес работы",
        "addresses.address.city": "Город",
        "addresses.address.region": "Регион",
        "address": "Адрес работы",
        "city": "Город",
        "region": "Регион",
        "region.name": "Название региона",
        "region.region_code": "Код региона",
        "workPlaceType": "Формат работы",
        "workPlaceType.workPlaceForeign": "Рабочие места для иностранных работников",
        "workPlaceType.workPlaceOrdinary": "Обычные рабочие места",
        "workPlaceType.workPlaceQuota": "Рабочие места по квоте",
        "workPlaceType.workPlaceSpecial": "Специальные рабочие места",
        "remote": "Удалённая работа",
        "employment_type": "Тип занятости",
        "contract_type": "Вид договора",
        "term": "Срок работы",
        "term.text": "Срок работы",
        "description": "Описание вакансии",
        "conditions": "Условия работы",
        "additional_information": "Дополнительная информация",
        "benefits": "Льготы и преимущества",
    }
)
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


def show_vacancy(report: VacancyReport) -> None:
    vacancy_id = f"ID {report.id}"
    title = (report.job_title or "").strip()
    identity = f"{title} · {vacancy_id}" if title else vacancy_id
    identity = re.sub(r"([\\`*_{}\[\]<>&$!~])", r"\\\1", " ".join(identity.split()))
    status = f"Замечаний: {len(report.issues)}" if report.issues else "Без замечаний"
    with st.expander(f"{identity} · {status}"):
        if report.url and is_web_url(report.url):
            st.link_button("Открыть вакансию", report.url)
        if not report.issues:
            st.success("В отчёте по этой вакансии замечаний нет.")
        for issue in report.issues:
            with st.container(border=True):
                st.subheader(f"{TOPICS[issue.topic]} · {TYPES[issue.type]}")
                st.text(issue.problem)
                for evidence in issue.evidence:
                    path = evidence.field
                    label = FIELD_LABELS.get(re.sub(r"\[\d+\]", "", path))
                    st.text(f"{label} ({path})" if label else path)
                    st.text(evidence.quote or "Сведения отсутствуют.")
                st.markdown("**Рекомендация редактору**")
                st.code(issue.recommendation, language=None, wrap_lines=True)


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
    search = filters[0].text_input("Поиск по ID или названию").strip().casefold()
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
        if (
            search not in report.id.casefold()
            and search not in (report.job_title or "").casefold()
        ):
            continue
        if (status == "С замечаниями" and not report.issues) or (
            status == "Без замечаний" and report.issues
        ):
            continue
        if (topic is None and kind is None) or any(
            (topic is None or issue.topic == topic)
            and (kind is None or issue.type == kind)
            for issue in report.issues
        ):
            matches.append(report)

    st.caption(f"Найдено {len(matches)} из {len(reports)} вакансий")
    if not matches:
        st.info("По этим фильтрам вакансий нет.")
    for report in matches:
        show_vacancy(report)


if __name__ == "__main__":
    main()
