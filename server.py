"""Local OpenAI-compatible proxy that securely exposes OpenRouter free models."""

from __future__ import annotations

import hmac
import json
import os
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import httpx
from dotenv import load_dotenv

from main import DEFAULT_MODEL, OPENROUTER_BASE_URL, ensure_free_model, get_api_key

UPSTREAM_TIMEOUT = 120.0


def _is_free_model_id(model_id: str) -> bool:
    """Return True if model_id is OpenRouter's free router or a :free model."""
    cleaned = model_id.strip()
    return cleaned == "openrouter/free" or cleaned.endswith(":free")


def _json_error(status: int, message: str, error_type: str = "invalid_request_error") -> bytes:
    """Serialize an OpenAI-style JSON error payload."""
    return json.dumps({"error": {"message": message, "type": error_type}}).encode("utf-8")


def _send_macos_notification(title: str, message: str) -> None:
    """Display a native macOS desktop notification when running on Darwin."""
    if sys.platform != "darwin":
        return
    script = (
        f'display notification {json.dumps(message)} '
        f'with title {json.dumps(title)} sound name "Sosumi"'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception as exc:
        sys.stderr.write(f"[alert] Failed to send macOS notification: {exc}\n")


def _send_webhook(webhook_url: str, message: str, client: httpx.Client) -> None:
    """POST an alert message to an optional webhook (Discord/Slack/ntfy/generic)."""
    if not webhook_url:
        return
    try:
        client.post(
            webhook_url,
            json={"text": message, "content": message, "message": message},
            timeout=10.0,
        )
    except Exception as exc:
        sys.stderr.write(f"[alert] Failed to send webhook notification: {exc}\n")


class ChargeMonitor:
    """Tracks OpenRouter spend and fires alerts immediately on any non-zero charge."""

    def __init__(
        self,
        alert_webhook_url: str = "",
        notify_macos: bool = True,
        notifier: Callable[[str, str], None] | None = None,
    ) -> None:
        self.alert_webhook_url = alert_webhook_url.strip()
        self.notify_macos = notify_macos
        self._notifier = notifier
        self._lock = threading.Lock()
        self.last_known_usage: float = 0.0

    def alert(self, title: str, message: str, http_client: httpx.Client) -> None:
        """Log and dispatch desktop/webhook notifications."""
        sys.stderr.write(f"[ALERT] {title}: {message}\n")
        if self._notifier is not None:
            self._notifier(title, message)
            return
        if self.notify_macos:
            _send_macos_notification(title, message)
        if self.alert_webhook_url:
            _send_webhook(self.alert_webhook_url, f"{title}: {message}", http_client)

    def check_response_cost(
        self,
        payload: dict[str, Any],
        model: str,
        http_client: httpx.Client,
    ) -> None:
        """Inspect a chat completion response/chunk for non-zero usage.cost."""
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        for field in ("cost", "total_cost"):
            val = usage.get(field)
            if isinstance(val, (int, float)) and val > 0:
                self.alert(
                    "OpenRouter Charge Alert",
                    f"Non-zero cost (${val:.6f}) reported for model '{model}'.",
                    http_client,
                )
                break

    def check_key_usage(
        self,
        base_url: str,
        openrouter_api_key: str,
        http_client: httpx.Client,
        is_startup: bool = False,
    ) -> dict[str, Any] | None:
        """Query OpenRouter /key endpoint and alert if usage increased or is > 0 on startup."""
        try:
            resp = http_client.get(
                f"{base_url}/key",
                headers={"Authorization": f"Bearer {openrouter_api_key}"},
                timeout=10.0,
            )
            if resp.status_code != HTTPStatus.OK:
                return None
            data = resp.json().get("data", {})
        except Exception:
            return None

        usage = float(data.get("usage") or 0.0)
        with self._lock:
            if is_startup:
                self.last_known_usage = usage
                if usage > 0:
                    self.alert(
                        "OpenRouter Charge Alert",
                        f"Existing key spend detected at startup: ${usage:.6f}.",
                        http_client,
                    )
            elif usage > self.last_known_usage:
                delta = usage - self.last_known_usage
                self.last_known_usage = usage
                self.alert(
                    "OpenRouter Charge Alert",
                    f"Key spend increased by ${delta:.6f} (total: ${usage:.6f}).",
                    http_client,
                )
        return data


class ProxyConfig:
    """Runtime configuration for the local OpenRouter proxy."""

    def __init__(
        self,
        openrouter_api_key: str,
        proxy_api_key: str = "",
        default_model: str = DEFAULT_MODEL,
        cors_origins: tuple[str, ...] = (),
        base_url: str = OPENROUTER_BASE_URL,
        alert_webhook_url: str = "",
        notify_macos: bool = True,
    ) -> None:
        self.openrouter_api_key = openrouter_api_key
        self.proxy_api_key = proxy_api_key.strip()
        self.default_model = ensure_free_model(default_model)
        self.cors_origins = cors_origins
        self.base_url = base_url.rstrip("/")
        self.alert_webhook_url = alert_webhook_url.strip()
        self.notify_macos = notify_macos

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """Load configuration from environment variables / .env."""
        load_dotenv()
        openrouter_key = get_api_key()
        proxy_key = os.getenv("PROXY_API_KEY", "").strip()
        default_model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        raw_origins = os.getenv("PROXY_CORS_ORIGINS", "")
        cors_origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        webhook = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        notify_macos = os.getenv("NOTIFY_MACOS", "true").strip().lower() != "false"
        return cls(
            openrouter_api_key=openrouter_key,
            proxy_api_key=proxy_key,
            default_model=default_model,
            cors_origins=cors_origins,
            alert_webhook_url=webhook,
            notify_macos=notify_macos,
        )


class OpenRouterProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler exposing /health, /v1/models, /v1/usage, and /v1/chat/completions."""

    protocol_version = "HTTP/1.1"
    config: ProxyConfig
    http_client: httpx.Client
    monitor: ChargeMonitor

    def log_message(self, format: str, *args: Any) -> None:
        """Log requests to stderr without leaking sensitive headers."""
        sys.stderr.write(f"[proxy] {self.address_string()} - {format % args}\n")

    def _cors_headers(self) -> dict[str, str]:
        origin = self.headers.get("Origin", "")
        allowed = self.config.cors_origins
        if not allowed or not origin:
            return {}
        if "*" in allowed or origin in allowed:
            return {
                "Access-Control-Allow-Origin": origin if "*" not in allowed else "*",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            }
        return {}

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _require_auth(self) -> bool:
        """Verify Bearer token against PROXY_API_KEY when configured."""
        expected = self.config.proxy_api_key
        if not expected:
            return True

        auth_header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not auth_header.startswith(prefix):
            self._send_bytes(
                HTTPStatus.UNAUTHORIZED,
                _json_error(HTTPStatus.UNAUTHORIZED, "Missing Bearer token.", "authentication_error"),
            )
            return False

        provided = auth_header[len(prefix):].strip()
        if not hmac.compare_digest(provided, expected):
            self._send_bytes(
                HTTPStatus.UNAUTHORIZED,
                _json_error(HTTPStatus.UNAUTHORIZED, "Invalid proxy API key.", "authentication_error"),
            )
            return False

        return True

    def _upstream_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Personal OpenRouter Proxy",
        }

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_bytes(HTTPStatus.OK, b'{"status":"ok"}')
            return

        if path == "/v1/models":
            if not self._require_auth():
                return
            self._handle_list_models()
            return

        if path == "/v1/usage":
            if not self._require_auth():
                return
            self._handle_usage()
            return

        self._send_bytes(
            HTTPStatus.NOT_FOUND,
            _json_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}", "not_found_error"),
        )

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            if not self._require_auth():
                return
            self._handle_chat_completions()
            return

        self._send_bytes(
            HTTPStatus.NOT_FOUND,
            _json_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}", "not_found_error"),
        )

    def _handle_usage(self) -> None:
        data = self.monitor.check_key_usage(
            base_url=self.config.base_url,
            openrouter_api_key=self.config.openrouter_api_key,
            http_client=self.http_client,
        )
        if data is None:
            self._send_bytes(
                HTTPStatus.BAD_GATEWAY,
                _json_error(HTTPStatus.BAD_GATEWAY, "Could not fetch OpenRouter key usage.", "api_error"),
            )
            return
        summary = {
            "usage_usd": data.get("usage", 0),
            "usage_daily_usd": data.get("usage_daily", 0),
            "limit_usd": data.get("limit"),
            "limit_remaining_usd": data.get("limit_remaining"),
            "is_free_tier": data.get("is_free_tier"),
            "free_model_daily_requests": data.get("free_model_daily_requests"),
        }
        self._send_bytes(HTTPStatus.OK, json.dumps(summary).encode("utf-8"))

    def _handle_list_models(self) -> None:
        try:
            resp = self.http_client.get(
                f"{self.config.base_url}/models",
                headers={"Authorization": f"Bearer {self.config.openrouter_api_key}"},
            )
        except httpx.HTTPError as exc:
            self._send_bytes(
                HTTPStatus.BAD_GATEWAY,
                _json_error(HTTPStatus.BAD_GATEWAY, f"Upstream connection error: {exc}", "api_error"),
            )
            return

        if resp.status_code != HTTPStatus.OK:
            self._send_bytes(resp.status_code, resp.content)
            return

        data = resp.json()
        free_models = [
            m for m in data.get("data", []) if _is_free_model_id(str(m.get("id", "")))
        ]
        body = json.dumps({"object": "list", "data": free_models}).encode("utf-8")
        self._send_bytes(HTTPStatus.OK, body)

    def _schedule_key_usage_check(self) -> None:
        threading.Thread(
            target=self.monitor.check_key_usage,
            args=(self.config.base_url, self.config.openrouter_api_key, self.http_client),
            daemon=True,
        ).start()

    def _handle_chat_completions(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_bytes(
                HTTPStatus.BAD_REQUEST,
                _json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header."),
            )
            return

        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_bytes(
                HTTPStatus.BAD_REQUEST,
                _json_error(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON."),
            )
            return

        if not isinstance(payload, dict):
            self._send_bytes(
                HTTPStatus.BAD_REQUEST,
                _json_error(HTTPStatus.BAD_REQUEST, "JSON body must be an object."),
            )
            return

        requested_model = str(payload.get("model") or self.config.default_model)
        try:
            safe_model = ensure_free_model(requested_model)
        except RuntimeError as exc:
            self._send_bytes(
                HTTPStatus.BAD_REQUEST,
                _json_error(HTTPStatus.BAD_REQUEST, str(exc)),
            )
            return

        payload["model"] = safe_model
        url = f"{self.config.base_url}/chat/completions"

        if payload.get("stream"):
            self._forward_streaming(url, payload, safe_model)
            return

        try:
            resp = self.http_client.post(url, headers=self._upstream_headers(), json=payload)
        except httpx.HTTPError as exc:
            self._send_bytes(
                HTTPStatus.BAD_GATEWAY,
                _json_error(HTTPStatus.BAD_GATEWAY, f"Upstream connection error: {exc}", "api_error"),
            )
            return

        if resp.status_code == HTTPStatus.OK:
            try:
                self.monitor.check_response_cost(resp.json(), safe_model, self.http_client)
            except Exception:
                pass
            self._schedule_key_usage_check()

        content_type = resp.headers.get("content-type", "application/json")
        self._send_bytes(resp.status_code, resp.content, content_type=content_type)

    def _forward_streaming(self, url: str, payload: dict[str, Any], model: str) -> None:
        try:
            with self.http_client.stream(
                "POST",
                url,
                headers=self._upstream_headers(),
                json=payload,
            ) as resp:
                self.send_response(resp.status_code)
                self.send_header(
                    "Content-Type",
                    resp.headers.get("content-type", "text/event-stream"),
                )
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                for k, v in self._cors_headers().items():
                    self.send_header(k, v)
                self.end_headers()
                for chunk in resp.iter_bytes():
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        self._inspect_stream_chunk(chunk, model)
                self.close_connection = True
            self._schedule_key_usage_check()
        except httpx.HTTPError as exc:
            self._send_bytes(
                HTTPStatus.BAD_GATEWAY,
                _json_error(HTTPStatus.BAD_GATEWAY, f"Upstream stream error: {exc}", "api_error"),
            )

    def _inspect_stream_chunk(self, chunk: bytes, model: str) -> None:
        for line in chunk.decode("utf-8", errors="ignore").splitlines():
            if not line.startswith("data: "):
                continue
            raw_json = line[6:].strip()
            if not raw_json or raw_json == "[DONE]":
                continue
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                self.monitor.check_response_cost(parsed, model, self.http_client)


def create_server(
    config: ProxyConfig,
    host: str = "127.0.0.1",
    port: int = 8000,
    http_client: httpx.Client | None = None,
    monitor: ChargeMonitor | None = None,
) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer instance."""
    client = http_client or httpx.Client(timeout=UPSTREAM_TIMEOUT)
    active_monitor = monitor or ChargeMonitor(
        alert_webhook_url=config.alert_webhook_url,
        notify_macos=config.notify_macos,
    )

    class ConfiguredHandler(OpenRouterProxyHandler):
        pass

    ConfiguredHandler.config = config
    ConfiguredHandler.http_client = client
    ConfiguredHandler.monitor = active_monitor
    return ThreadingHTTPServer((host, port), ConfiguredHandler)


def main() -> int:
    config = ProxyConfig.from_env()
    host = os.getenv("PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PROXY_PORT", "8000"))

    if not config.proxy_api_key:
        print(
            "Warning: PROXY_API_KEY is not set. Local requests will not require a Bearer token.",
            file=sys.stderr,
        )

    client = httpx.Client(timeout=UPSTREAM_TIMEOUT)
    monitor = ChargeMonitor(
        alert_webhook_url=config.alert_webhook_url,
        notify_macos=config.notify_macos,
    )
    monitor.check_key_usage(
        base_url=config.base_url,
        openrouter_api_key=config.openrouter_api_key,
        http_client=client,
        is_startup=True,
    )

    server = create_server(
        config=config,
        host=host,
        port=port,
        http_client=client,
        monitor=monitor,
    )
    print(f"Personal OpenRouter Proxy listening on http://{host}:{port}/v1 (free models only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy.")
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
