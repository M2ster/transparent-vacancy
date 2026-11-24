import json
import os
import tempfile
from pathlib import Path

from .models import VacancyReport


def write_report(path: Path, reports: list[VacancyReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                [report.model_dump() for report in reports],
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
