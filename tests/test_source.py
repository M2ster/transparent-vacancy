import json

import httpx
import pytest

from vacancy_audit.source import fetch_vacancies, read_file

URL = "https://portal.test/vacancies"


def envelope(ids, *, total=5, offset=0, limit=2, status="200"):
    return {
        "status": status,
        "meta": {"total": str(total), "limit": str(limit), "offset": str(offset)},
        "results": {"vacancies": [{"vacancy": {"id": value}} for value in ids]},
    }


def test_api_pagination_deduplicates_and_applies_limit():
    calls = []
    pages = [["a", "b"], ["b", "c"], ["d"]]

    def handler(request):
        page = int(request.url.params["offset"])
        calls.append(page)
        assert request.url.params["limit"] == "2"
        return httpx.Response(200, json=envelope(pages[page], offset=page))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_vacancies(client, URL, maximum=4, page_size=2)
    assert calls == [0, 1, 2]
    assert [row["id"] for row in result] == ["a", "b", "c", "d"]


def test_api_limit_below_page_size():
    def handler(request):
        assert request.url.params["limit"] == "1"
        return httpx.Response(200, json=envelope(["a"], limit=1))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert fetch_vacancies(client, URL, maximum=1, page_size=100) == [{"id": "a"}]


@pytest.mark.parametrize(
    ("status", "payload", "error"),
    [
        (503, {}, httpx.HTTPStatusError),
        (200, envelope([], total=0, status="500"), ValueError),
        (200, {"status": 200, "results": {}}, ValueError),
        (200, envelope(["a"], offset=1), ValueError),
        (200, envelope([None]), ValueError),
        (200, envelope([], total=5), ValueError),
    ],
)
def test_api_rejects_errors(status, payload, error):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json=payload))
    with httpx.Client(transport=transport) as client, pytest.raises(error):
        fetch_vacancies(client, URL, maximum=3, page_size=2)


def test_api_rejects_repeated_page_even_without_meta_offset():
    payload = envelope(["a", "b"])
    del payload["meta"]["offset"]
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(ValueError, match="повтор страницы"),
    ):
        fetch_vacancies(client, URL, maximum=3, page_size=2)


def test_local_unpacks_nested_and_preserves_order(tmp_path):
    path = tmp_path / "vacancies.json"
    path.write_text(json.dumps(envelope(["a", 2, "a", "b"])), encoding="utf-8")
    assert [row["id"] for row in read_file(path)] == ["a", 2, "b"]
