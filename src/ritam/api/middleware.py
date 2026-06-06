import json
import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ritam.observability.context import request_id_context_var


class LoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.logger = logging.getLogger(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start_time = time.perf_counter()
        request_token = request_id_context_var.set(str(uuid.uuid4()))
        result: dict[str, object] = {}

        async def wrapped_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                result["status_code"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append(
                    (
                        b"x-request-id",
                        request_id_context_var.get().encode("utf-8"),
                    )
                )
                message["headers"] = headers
            elif message["type"] == "http.response.body":
                result["body"] = message.get("body")
            await send(message)

        error_type: str | None = None
        try:
            await self.app(scope, receive, wrapped_send)
        except Exception as exc:
            # Capture the class name so the summary log records the crash, then
            # re-raise so the ASGI server can still produce its 500.
            error_type = exc.__class__.__name__
            raise
        finally:
            # Default to 500 — if response never started, the request crashed
            # before a status was set, and we want the summary log to reflect that.
            raw_status = result.get("status_code")
            status_code = raw_status if isinstance(raw_status, int) else 500
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_extra_fields: dict[str, object] = {
                "path": scope["path"],
                "method": scope["method"],
                "status_code": status_code,
                "duration_ms": round(duration_ms, 3),
            }
            if error_type:
                log_extra_fields["error_type"] = error_type
            if status_code >= 400:
                body = result.get("body")
                if isinstance(body, bytes):
                    try:
                        log_extra_fields["error"] = json.loads(
                            body.decode("utf-8")
                        ).get("error")
                    except Exception:
                        log_extra_fields["error"] = "Unparseable response body"
            log_level = logging.ERROR if status_code >= 400 else logging.INFO
            self.logger.log(
                log_level, "request_complete", extra={"fields": log_extra_fields}
            )
            request_id_context_var.reset(request_token)
