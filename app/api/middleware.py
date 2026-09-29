"""Pure-ASGI middleware: request IDs and request body size limits."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def error_body(code: str, message: str, request_id: str, err_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "code": code, "request_id": request_id}}


class GatewayMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                raw.append((b"x-request-id", request_id.encode()))
                message["headers"] = raw
            await send(message)

        if scope["method"] in {"POST", "PUT", "PATCH"}:
            declared = headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
                await self._reject(send_with_id, request_id)
                return
            chunks: list[bytes] = []
            total = 0
            more = True
            while more:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.max_body_bytes:
                    await self._reject(send_with_id, request_id)
                    return
                chunks.append(chunk)
                more = message.get("more_body", False)
            body = b"".join(chunks)
            replayed = False

            async def replay() -> Message:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, send_with_id)
            return

        await self.app(scope, receive, send_with_id)

    async def _reject(self, send: Send, request_id: str) -> None:
        payload = json.dumps(
            error_body("request_too_large", f"Request body exceeds {self.max_body_bytes} bytes.", request_id)
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})
