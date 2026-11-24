import json
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation

from .models import Evidence, Issue, Topic

FIELDS: dict[Topic, tuple[str, ...]] = {
    "job_title": ("job-name", "job_title", "name", "title"),
    "duties": ("duty", "duties", "responsibilities"),
    "requirements": (
        "requirement",
        "requirements",
        "experience",
        "education",
        "qualification",
        "skills",
    ),
    "salary": (
        "salary_min",
        "salary_max",
        "salary",
        "salary_currency",
        "pay",
        "compensation",
    ),
    "schedule": ("schedule", "shift", "work_schedule", "work_hours"),
    "location": (
        "addresses",
        "location",
        "address",
        "region",
        "workPlaceType",
        "remote",
        "city",
    ),
    "employment": ("employment", "employment_type", "contract_type", "term"),
}
SHARED = ("description", "conditions", "additional_information", "benefits")
MISSING: dict[Topic, tuple[str, str]] = {
    "job_title": (
        "Не указано название должности.",
        "Добавьте понятное название должности.",
    ),
    "duties": ("Не описаны обязанности.", "Перечислите основные задачи сотрудника."),
    "requirements": (
        "Не указаны требования к кандидату.",
        "Укажите обязательные навыки или явно укажите, что специальных требований нет.",
    ),
    "salary": (
        "Оплата не указана.",
        "Укажите сумму, диапазон или порядок расчёта оплаты.",
    ),
    "schedule": (
        "График работы не указан.",
        "Укажите дни и часы работы либо длительность и порядок смен.",
    ),
    "location": (
        "Не указано место или формат работы.",
        "Укажите населённый пункт или возможность работать удалённо.",
    ),
    "employment": (
        "Не указан тип занятости.",
        "Укажите тип занятости, например полную или частичную.",
    ),
}
PLACEHOLDERS = {"", "-", "—", "не указано", "не указан", "не указана", "н/д", "null"}
SALARY_BOUNDS = ("salary_min", "salary_max")


def fields(vacancy: dict) -> dict:
    data = {
        key: vacancy[key]
        for key in dict.fromkeys(
            (
                *SHARED,
                "currency",
                *(field for group in FIELDS.values() for field in group),
            )
        )
        if key in vacancy
    }
    addresses = data.get("addresses")
    if isinstance(addresses, dict) and isinstance(addresses.get("address"), list):
        data["addresses"] = {
            "address": [
                {
                    key: row[key]
                    for key in ("location", "address", "city", "region")
                    if key in row
                }
                if isinstance(row, dict)
                else row
                for row in addresses["address"]
            ]
        }
    return data


def leaves(value: object, path: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, f"{path}[{index}]")
    else:
        yield path, value


def money(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value).strip().replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def informative(path: str, value: object) -> bool:
    if (
        value is None
        or isinstance(value, str)
        and value.strip().casefold() in PLACEHOLDERS
    ):
        return False
    return not (path in (*SALARY_BOUNDS, "salary") and money(value) == 0)


def display(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def topic_present(
    topic: Topic, values: dict[str, object], *, shared: bool = True
) -> bool:
    candidates = (
        (*FIELDS[topic], *SHARED) if shared and topic != "job_title" else FIELDS[topic]
    )
    return any(
        path.split(".", 1)[0].split("[", 1)[0] in candidates
        and informative(path, value)
        for path, value in values.items()
    )


def rules(data: dict) -> list[Issue]:
    values = dict(leaves(data))
    issues = []
    for topic, (problem, recommendation) in MISSING.items():
        if not topic_present(topic, values):
            issues.append(
                Issue(
                    topic=topic,
                    type="missing",
                    evidence=[
                        Evidence(field=field, quote="") for field in FIELDS[topic][:2]
                    ]
                    if topic == "salary"
                    else [Evidence(field=FIELDS[topic][0], quote="")],
                    problem=problem,
                    recommendation=recommendation,
                )
            )
    minimum, maximum = (money(data.get(field)) for field in SALARY_BOUNDS)
    if minimum is not None and maximum is not None and minimum > maximum > 0:
        issues.append(
            Issue(
                topic="salary",
                type="contradiction",
                evidence=[
                    Evidence(field=field, quote=display(data[field]))
                    for field in SALARY_BOUNDS
                ],
                problem="Минимальная зарплата выше максимальной.",
                recommendation="Проверьте обе границы и укажите правильный диапазон зарплаты.",
            )
        )
    return issues


def validate_evidence(issue: Issue, data: dict) -> None:
    values = dict(leaves(data))
    if not issue.problem.strip() or not issue.recommendation.strip():
        raise ValueError("Пустое описание замечания")
    if issue.type == "missing" and topic_present(issue.topic, values, shared=False):
        raise ValueError("Для темы уже есть сведения")
    for evidence in issue.evidence:
        path, quote = evidence.field, evidence.quote
        if not quote and issue.type == "missing":
            known = (
                path in values
                or path in FIELDS[issue.topic]
                or path
                in {
                    "requirement.education",
                    "requirement.experience",
                    "requirement.qualification",
                    "addresses.address[0].location",
                }
            )
            if known and not informative(path, values.get(path)):
                continue
        value = values.get(path)
        if (
            path not in values
            or not quote.strip()
            or (
                quote not in value
                if isinstance(value, str)
                else quote != display(value)
            )
        ):
            raise ValueError("Цитата отсутствует в исходном поле")
    if issue.type == "contradiction":
        for first in issue.evidence:
            for second in issue.evidence:
                if first.quote.strip() == second.quote.strip():
                    continue
                if first.field != second.field:
                    return
                value = display(values[first.field])
                start = value.find(first.quote)
                if value.find(second.quote, start + len(first.quote)) >= 0:
                    return
        raise ValueError("Для противоречия нужны две разные цитаты")


def merge_issues(issues: list[Issue]) -> list[Issue]:
    result: list[Issue] = []
    for issue in issues:
        paths = {e.field for e in issue.evidence}
        message = " ".join(issue.problem.casefold().split())
        if any(
            other.topic == issue.topic
            and (
                other.type == "missing"
                and issue.type in {"missing", "ambiguity"}
                or other.type == issue.type
                and (
                    paths == {e.field for e in other.evidence}
                    or message == " ".join(other.problem.casefold().split())
                )
            )
            for other in result
        ):
            continue
        result.append(issue)
    return result
