import argparse
import json
import logging
import math
from pathlib import Path

import httpx

from .llm import MAX_AUTO_CONTEXT, ReviewError, analyze
from .models import VacancyReport
from .report import write_report
from .rules import combine_issues, fields
from .source import API_URL, fetch_vacancies, read_file

log = logging.getLogger("vacancy_audit")


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("требуется положительное целое число")
    return result


def seconds(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("требуется положительный таймаут в секундах")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Редакторская проверка вакансий")
    parser.add_argument(
        "--input", type=Path, help="Локальный JSON вместо запроса к порталу"
    )
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument(
        "--limit", type=positive_int, default=200, help="Максимум вакансий из API"
    )
    parser.add_argument(
        "--page-size", type=positive_int, default=100, help="От 1 до 100"
    )
    parser.add_argument("--api-timeout", type=seconds, default=15)
    parser.add_argument("--output", type=Path, default=Path("reports/latest.json"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument(
        "--num-ctx", type=positive_int, help="Фиксированный контекст вместо автовыбора"
    )
    parser.add_argument("--num-predict", type=positive_int, default=900)
    parser.add_argument("--ollama-timeout", type=seconds, default=60)
    args = parser.parse_args(argv)
    if args.page_size > 100:
        parser.error("--page-size должен быть от 1 до 100")
    if args.num_predict >= (args.num_ctx or MAX_AUTO_CONTEXT):
        parser.error("--num-predict должен быть меньше максимального контекста запуска")

    try:
        if args.input is not None:
            vacancies = read_file(args.input)
        else:
            with httpx.Client(
                timeout=args.api_timeout, follow_redirects=True
            ) as client:
                vacancies = fetch_vacancies(
                    client,
                    args.api_url,
                    maximum=args.limit,
                    page_size=args.page_size,
                )
        results = []
        with httpx.Client(timeout=args.ollama_timeout, trust_env=False) as client:
            for index, vacancy in enumerate(vacancies, 1):
                log.info("Вакансия %d из %d", index, len(vacancies))
                data = fields(vacancy)
                try:
                    model_issues = analyze(
                        client,
                        data,
                        url=args.ollama_url,
                        model=args.model,
                        context=args.num_ctx,
                        predict=args.num_predict,
                    )
                except ReviewError as exc:
                    raise ReviewError(f"Вакансия {index}: {exc}") from None
                url = next(
                    (
                        vacancy[key].strip()
                        for key in ("vac_url", "url", "link")
                        if isinstance(vacancy.get(key), str) and vacancy[key].strip()
                    ),
                    None,
                )
                results.append(
                    VacancyReport(
                        id=str(vacancy["id"]).strip(),
                        url=url,
                        issues=combine_issues(data, model_issues),
                    )
                )
        write_report(args.output, results)
    except httpx.HTTPStatusError as exc:
        log.error("Портал вернул HTTP %d", exc.response.status_code)
    except httpx.HTTPError:
        log.error("Портал недоступен или не ответил вовремя")
    except (json.JSONDecodeError, UnicodeError):
        log.error("Не удалось прочитать корректный JSON в UTF-8")
    except OSError as exc:
        log.error("Ошибка чтения или записи файла (%s)", type(exc).__name__)
    except (ReviewError, ValueError) as exc:
        log.error("%s", exc)
    else:
        log.info("Отчёт записан. Вакансий: %d", len(results))
        return 0
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())
