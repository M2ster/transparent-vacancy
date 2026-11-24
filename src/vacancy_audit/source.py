import json
from collections.abc import Iterable
from pathlib import Path

import httpx

API_URL = "http://opendata.trudvsem.ru/api/v1/vacancies/region/39"


def integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Ожидалось целое число")
    return int(value)


def unpack(payload: object, *, api: bool = False) -> list[dict]:
    if isinstance(payload, dict):
        if ("status" in payload or api) and str(payload.get("status")) != "200":
            raise ValueError("Портал вернул статус ошибки или пропустил status")
        results = payload.get("results")
        rows = results.get("vacancies") if isinstance(results, dict) else None
    else:
        rows = payload if not api else None
    if not isinstance(rows, list):
        raise ValueError("Ожидался массив вакансий или results.vacancies")

    vacancies = []
    for index, row in enumerate(rows, 1):
        vacancy = (
            row.get("vacancy", None if api else row) if isinstance(row, dict) else None
        )
        vacancy_id = vacancy.get("id") if isinstance(vacancy, dict) else None
        if (
            isinstance(vacancy_id, bool)
            or not isinstance(vacancy_id, (str, int))
            or not str(vacancy_id).strip()
        ):
            raise ValueError(f"Запись {index}: отсутствует корректный идентификатор")
        vacancies.append(vacancy)
    return vacancies


def unique(vacancies: Iterable[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for vacancy in vacancies:
        by_id.setdefault(str(vacancy["id"]).strip(), vacancy)
    return list(by_id.values())


def read_file(path: Path) -> list[dict]:
    return unique(unpack(json.loads(path.read_text(encoding="utf-8-sig"))))


def fetch_vacancies(
    client: httpx.Client, url: str, *, maximum: int, page_size: int
) -> list[dict]:
    if maximum < 1 or not 1 <= page_size <= 100:
        raise ValueError(
            "Лимит должен быть положительным, размер страницы — от 1 до 100"
        )
    limit = min(maximum, page_size)
    by_id: dict[str, dict] = {}
    seen_pages: set[tuple[str, ...]] = set()
    page = 0
    available = None
    while len(by_id) < maximum:
        response = client.get(url, params={"offset": page, "limit": limit})
        response.raise_for_status()
        payload = response.json()
        vacancies = unpack(payload, api=True)
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("В ответе портала отсутствует meta")
        try:
            total = integer(meta.get("total"))
            actual_limit = integer(meta.get("limit", limit))
            actual_page = integer(meta.get("offset", page))
        except ValueError:
            raise ValueError("Некорректные параметры пагинации в meta") from None
        if (
            total < 0
            or actual_limit != limit
            or actual_page != page
            or len(vacancies) > limit
            or (page == 0 and bool(total) != bool(vacancies))
            or (len(vacancies) < limit and page * limit + len(vacancies) < total)
        ):
            raise ValueError("Портал вернул противоречивые параметры страницы")
        available = total if available is None else min(available, total)
        fingerprint = tuple(sorted(str(row["id"]).strip() for row in vacancies))
        if vacancies and fingerprint in seen_pages:
            raise ValueError("Портал вернул повтор страницы")
        seen_pages.add(fingerprint)
        for vacancy in vacancies:
            by_id.setdefault(str(vacancy["id"]).strip(), vacancy)
        page += 1
        if len(vacancies) < limit or page * limit >= available:
            break
    return list(by_id.values())[:maximum]
