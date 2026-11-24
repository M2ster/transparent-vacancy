from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Topic = Literal[
    "job_title",
    "duties",
    "requirements",
    "salary",
    "schedule",
    "location",
    "employment",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False, strict=True)


class Evidence(Contract):
    field: str = Field(min_length=1)
    quote: str


class Issue(Contract):
    topic: Topic
    type: Literal["missing", "ambiguity", "contradiction"]
    evidence: list[Evidence] = Field(min_length=1)
    problem: str = Field(min_length=1, max_length=220)
    recommendation: str = Field(min_length=1, max_length=260)


class Review(Contract):
    issues: list[Issue]


class VacancyReport(Review):
    id: str
    url: str | None
