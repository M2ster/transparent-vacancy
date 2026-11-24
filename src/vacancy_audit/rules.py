import json
import re
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
FREE_TEXT = (
    *SHARED,
    *FIELDS["duties"],
    "requirement",
    "requirements",
    "qualification",
    "requirement.qualification",
    "requirements.qualification",
)
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
YEARS = r"(?:от|не менее|минимум)\s+[1-9]\d*\s*(?:лет|год\w*|месяц\w*)\b"
EXPERIENCE = re.compile(
    rf"\b(?:опыт|стаж)\w*\b.*?\b{YEARS}|\b{YEARS}\s+(?:опыт|стаж)\w*\b", re.I
)
OPTIONAL = re.compile(
    r"\b(?:желател\w*|приветств\w*|преимуществ\w*|не\s+(?:обязател\w*|требу\w*))\b",
    re.I,
)
REMOTE = re.compile(
    r"\b(?:полностью|только|исключительно|100\s*%)\s+(?:удал[её]н\w*|дистанцион\w*)",
    re.I,
)
OFFICE = re.compile(r"\b(?:в\s+офис\w*|очн\w*)\b", re.I)
ALL_DAYS = re.compile(
    r"\b(?:ежеднев\w*|кажд\w*\s+рабоч\w*\s+день|все\s+рабочие\s+дни)\b", re.I
)
OFFICE_ONLY = re.compile(
    r"\b(?:только|исключительно)\s+(?:работа\s+)?в\s+офис\w*\b", re.I
)
TIMETABLE = re.compile(r"\b(?:\d+/\d+|\d{1,2}:\d{2}\s*(?:[–—-]|до)\s*\d{1,2}:\d{2})\b")
OPEN_SCHEDULE = re.compile(
    r"\b(?:гибк\w*|плавающ\w*|иногда|или|по договор[её]нности|по согласованию)\b", re.I
)
GENERIC_SCHEDULE = re.compile(r"(?:гибкий|сменный|плавающий) график[.!]?", re.I)
START_WINDOW = re.compile(
    r"\b(?:начал\w*|старт\w*)\b[^.!?]*?\b(?:между|с)\s+\d{1,2}:\d{2}"
    r"\s+(?:и|до)\s+\d{1,2}:\d{2}\b",
    re.I,
)
DEFERRED_LOCATION = re.compile(
    r"\b(?:формат|адрес|место)(?:\s+и\s+адрес)?\s+(?:работы|офиса|рабочего места)\b"
    r".*?\b(?:уточн\w*|сообщ\w*|обсужд\w*)\b"
    r".*?\b(?:позже|отдельно|на\s+собеседовани\w*|при\s+собеседовани\w*)\b",
    re.I,
)


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


def topic_present(topic: Topic, values: dict[str, object]) -> bool:
    return any(
        path.split(".", 1)[0].split("[", 1)[0] in FIELDS[topic]
        and path != "salary_currency"
        and informative(path, value)
        for path, value in values.items()
    )


def has_free_text(values: dict[str, object]) -> bool:
    return any(
        re.sub(r"\[\d+\]", "", path) in FREE_TEXT
        and isinstance(value, str)
        and informative(path, value)
        for path, value in values.items()
    )


def rules(data: dict) -> list[Issue]:
    values = dict(leaves(data))
    issues = []
    free_text = has_free_text(values)
    for topic in MISSING:
        # Structured blanks are conclusive only when prose cannot supply the condition.
        if not topic_present(topic, values) and (topic == "job_title" or not free_text):
            issues.append(missing_issue(topic))
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
    schedule = data.get("schedule")
    if isinstance(schedule, str) and GENERIC_SCHEDULE.fullmatch(schedule.strip()):
        details = " ".join(value for value in values.values() if isinstance(value, str))
        if not TIMETABLE.search(details) and not re.search(
            r"\b(?:можно\s+выбрать|согласовыва\w*|начало\s+смен\w*|начало\s+работ\w*)\b",
            details,
            re.I,
        ):
            issues.append(
                Issue(
                    topic="schedule",
                    type="ambiguity",
                    evidence=[Evidence(field="schedule", quote=schedule)],
                    problem="Указан тип графика, но его практические условия не описаны.",
                    recommendation="Уточните, как выбираются часы или как чередуются смены.",
                )
            )
    return issues + text_issues(values)


def missing_issue(topic: Topic) -> Issue:
    problem, recommendation = MISSING[topic]
    paths = FIELDS[topic][:2] if topic == "salary" else FIELDS[topic][:1]
    return Issue(
        topic=topic,
        type="missing",
        evidence=[Evidence(field=path, quote="") for path in paths],
        problem=problem,
        recommendation=recommendation,
    )


def text_issues(values: dict[str, object]) -> list[Issue]:
    clauses = [
        Evidence(field=path, quote=clause.strip())
        for path, value in values.items()
        if isinstance(value, str)
        for clause in re.split(r"(?<=[.!?;])\s+|\n", value)
        if clause.strip()
    ]
    issues = []
    experience = next(
        (
            item
            for item in clauses
            if (
                EXPERIENCE.search(item.quote)
                and not OPTIONAL.search(item.quote)
                and item.field.split(".", 1)[0].split("[", 1)[0]
                in (*FIELDS["requirements"], *SHARED)
            )
        ),
        None,
    )
    zero = next(
        (
            path
            for path in ("requirement.experience", "experience")
            if money(values.get(path)) == 0
        ),
        None,
    )
    if zero and experience:
        issues.append(
            Issue(
                topic="requirements",
                type="contradiction",
                evidence=[
                    Evidence(field=zero, quote=display(values[zero])),
                    experience,
                ],
                problem="В поле стажа указан ноль, а в тексте задан обязательный минимум.",
                recommendation="Согласуйте минимальный стаж в структурированном поле и тексте.",
            )
        )
    remote = next(
        (
            item
            for item in clauses
            if REMOTE.search(item.quote)
            and not re.search(
                r"\bне\s+(?:полностью|только|исключительно)\b", item.quote, re.I
            )
        ),
        None,
    )
    office = next(
        (
            item
            for item in clauses
            if OFFICE.search(item.quote)
            and (ALL_DAYS.search(item.quote) or OFFICE_ONLY.search(item.quote))
            and not OPTIONAL.search(item.quote)
            and not re.search(
                r"\b(?:без|обучени\w*|стажиров\w*)\b|\bне\s+(?:кажд|ежеднев|только)",
                item.quote,
                re.I,
            )
        ),
        None,
    )
    if remote and office and remote != office:
        issues.append(
            Issue(
                topic="location",
                type="contradiction",
                evidence=[remote, office],
                problem="Полностью удалённый формат расходится с требованием работать в офисе.",
                recommendation="Укажите один формат и согласуйте с ним правила присутствия в офисе.",
            )
        )
    deferred = next(
        (item for item in clauses if DEFERRED_LOCATION.search(item.quote)), None
    )
    if deferred:
        issues.append(
            Issue(
                topic="location",
                type="ambiguity",
                evidence=[deferred],
                problem="Формат или рабочий адрес предлагается узнать позже.",
                recommendation="Укажите известный формат работы и место, где сотрудник будет работать очно.",
            )
        )
    return issues


def combine_issues(data: dict, inferred: list[Issue]) -> list[Issue]:
    verified = rules(data)
    scheduled = data.get("schedule", "")
    minimum, maximum = (money(data.get(field)) for field in SALARY_BOUNDS)
    determinate = [
        {item.field for item in issue.evidence}
        for issue in verified
        if issue.type == "contradiction"
    ]
    for issue in inferred:
        paths = {item.field for item in issue.evidence}
        if issue.type in {"ambiguity", "contradiction"} and paths in determinate:
            continue
        if any(DEFERRED_LOCATION.search(item.quote) for item in issue.evidence):
            continue
        if (
            issue.topic == "salary"
            and (
                issue.type == "ambiguity"
                and paths <= {*SALARY_BOUNDS, "currency", "salary_currency"}
                or issue.type == "contradiction"
                and paths == set(SALARY_BOUNDS)
            )
            and minimum is not None
            and minimum > 0
            and (maximum is None or maximum == 0 or maximum >= minimum)
        ):
            continue
        if issue.topic == "schedule" and isinstance(scheduled, str):
            timetable = TIMETABLE.findall(scheduled)
            if (
                issue.type == "ambiguity"
                and paths == {"schedule"}
                and (
                    timetable
                    and len({"/" in part for part in timetable}) == len(timetable)
                    and not OPEN_SCHEDULE.search(scheduled)
                    or GENERIC_SCHEDULE.fullmatch(scheduled.strip())
                    and any(
                        START_WINDOW.search(value)
                        for _, value in leaves(data)
                        if isinstance(value, str)
                    )
                )
            ):
                continue
            if (
                paths == {"schedule", "employment"}
                and re.fullmatch(r"гибкий график[.!]?", scheduled.strip(), re.I)
                and isinstance(data.get("employment"), str)
                and re.fullmatch(
                    r"(?:полная|частичная) занятость[.!]?",
                    data["employment"].strip(),
                    re.I,
                )
            ):
                continue
        verified.append(issue)
    return merge_issues(verified)


def missing_field(topic: Topic, path: str) -> bool:
    if path in FIELDS[topic]:
        return True
    if topic == "requirements":
        return bool(
            re.fullmatch(
                r"requirements?\.(education|experience|qualification|skills)", path
            )
        )
    return topic == "location" and bool(
        re.fullmatch(r"addresses\.address\[\d+\]\.(location|address|city|region)", path)
    )


def validate_evidence(issue: Issue, data: dict) -> None:
    values = dict(leaves(data))
    if not issue.problem.strip() or not issue.recommendation.strip():
        raise ValueError("Пустое описание замечания")
    if issue.type == "missing":
        if topic_present(issue.topic, values):
            raise ValueError("Для темы уже есть сведения")
        if any(
            evidence.quote
            or not missing_field(issue.topic, evidence.field)
            or informative(evidence.field, values.get(evidence.field))
            for evidence in issue.evidence
        ):
            raise ValueError(
                "Для пропуска укажите отсутствующее поле темы и пустую цитату"
            )
        return
    for evidence in issue.evidence:
        path, quote = evidence.field, evidence.quote
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
