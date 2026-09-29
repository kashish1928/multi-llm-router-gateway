"""Router interface. v1 is heuristic; a learned router can implement the same protocol."""

from __future__ import annotations

from typing import Protocol

from app.models import ChatCompletionRequest, RouteDecision


class Router(Protocol):
    def classify(self, request: ChatCompletionRequest) -> RouteDecision: ...
