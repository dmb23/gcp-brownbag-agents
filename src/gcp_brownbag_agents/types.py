from dataclasses import dataclass
from typing import TypedDict

import httpx
from pydantic import BaseModel


@dataclass
class RunDeps:
    client: httpx.AsyncClient


class ReferenceLink(BaseModel):
    """Web address of a helpful resource"""

    description: str
    url: str


class ImageLink(BaseModel):
    """Web address of an image / figure explaining some context"""

    description: str
    url: str


class ResearchResult(BaseModel):
    """Results of some research towards a suitable topic for a presentation."""

    full_text: str
    references: list[ReferenceLink]
    images: list[ImageLink]


class Story(BaseModel):
    """Story from Hackernews"""

    title: str
    url: str


class DuckDuckGoResult(TypedDict):
    """A DuckDuckGo search result."""

    title: str
    """The title of the search result."""
    href: str
    """The URL of the search result."""
    body: str
    """The body of the search result."""
