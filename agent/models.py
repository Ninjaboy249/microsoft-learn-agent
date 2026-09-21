"""Validated data contracts used by the agent and UI."""

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Source(BaseModel):
    """A Microsoft Learn page used to ground a response."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: HttpUrl


class Note(BaseModel):
    """A titled section of generated learning material."""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)


class LearnResponse(BaseModel):
    """Structured, grounded output returned by the agent."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    definition: str = ""
    summary: str = Field(min_length=1)
    key_concepts: list[str] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)