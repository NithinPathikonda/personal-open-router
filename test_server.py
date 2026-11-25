"""Unit tests for the local OpenRouter free-model proxy handler and charge alerts."""

from __future__ import annotations

import io
import json
import unittest
from email.parser import BytesParser
from typing import Any

import httpx

from server import ChargeMonitor, OpenRouterProxyHandler, ProxyConfig


class _FakeSocket:
    """Minimal socket-like object backed by BytesIO for in-memory handler tests."""

    def __init__(self, raw_request: bytes) -> None:
        self._rfile = io.BytesIO(raw_request)
        self._wfile = io.BytesIO()

    def makefile(self, mode: str, *args: Any, **kwargs: Any) -> io.BytesIO:
        if "r" in mode:
            return self._rfile
        return self._wfile

    def sendall(self, data: bytes) -> None:
        self._wfile.write(data)

    def close(self) -> None:
        pass


class ProxyHandlerTests(unittest.TestCase):
    """Verify auth, free-model enforcement, model filtering, usage, and charge alerts."""

    def setUp(self) -> None:
        self.captured_requests: list[httpx.Request] = []
        self.alerts: list[tuple[str, str]] = []
        self.mock_key_usage: float = 0.0
        self.mock_response_cost: float = 0.0

        def handler(request: httpx.Request) -> httpx.Response:
            self.captured_requests.append(request)
            if request.url.path.endswith("/key"):
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "usage": self.mock_key_usage,
                            "usage_daily": self.mock_key_usage,
                            "limit": 0,
                            "limit_remaining": 0,
                            "is_free_tier": True,
                            "free_model_daily_requests": {"used": 2, "limit": 50, "remaining": 48},
                        }
                    },
                )
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "openrouter/free"},
                            {"id": "meta-llama/llama-3.3-70b-instruct:free"},
                            {"id": "openai/gpt-4o"},
                        ]
                    },
                )
            if request.url.path.endswith("/chat/completions"):
                payload = json.loads(request.content.decode("utf-8"))
                if payload.get("stream"):
                    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
                    return httpx.Response(
                        200,
                        content=sse,
                        headers={"content-type": "text/event-stream"},
                    )
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-test",
                        "model": payload["model"],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": self.mock_response_cost},
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    },
                )
            return httpx.Response(404, json={"error": "not found"})

        self.upstream_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.config = ProxyConfig(
            openrouter_api_key="sk-or-real-secret",
            proxy_api_key="local-proxy-token",
            default_model="openrouter/free",
        )
        self.monitor = ChargeMonitor(
            notifier=lambda title, msg: self.alerts.append((title, msg)),
        )

    def tearDown(self) -> None:
        self.upstream_client.close()

    def _dispatch(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        body_bytes = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
        req_headers = {"Host": "127.0.0.1:8000", "Connection": "close"}
        if json_body is not None:
            req_headers["Content-Type"] = "application/json"
            req_headers["Content-Length"] = str(len(body_bytes))
        if headers:
            req_headers.update(headers)

        header_lines = "".join(f"{k}: {v}\r\n" for k, v in req_headers.items())
        raw_req = f"{method} {path} HTTP/1.1\r\n{header_lines}\r\n".encode("utf-8") + body_bytes
        sock = _FakeSocket(raw_req)

        class ConfiguredHandler(OpenRouterProxyHandler):
            config = self.config
            http_client = self.upstream_client
            monitor = self.monitor

            def log_message(self, format: str, *args: Any) -> None:
                pass

        ConfiguredHandler(sock, ("127.0.0.1", 54321), None)  # type: ignore[arg-type]
        raw_resp = sock._wfile.getvalue()
        head, _, body = raw_resp.partition(b"\r\n\r\n")
        status_line, _, header_block = head.partition(b"\r\n")
        status_code = int(status_line.split()[1])
        parsed_headers = dict(BytesParser().parsebytes(header_block).items())
        return status_code, parsed_headers, body

    def test_health_is_public(self) -> None:
        status, _, body = self._dispatch("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})

    def test_missing_and_invalid_auth_rejected(self) -> None:
        status_missing, _, _ = self._dispatch(
            "POST",
            "/v1/chat/completions",
            json_body={"model": "openrouter/free", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status_missing, 401)

        status_wrong, _, _ = self._dispatch(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-token"},
            json_body={"model": "openrouter/free", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status_wrong, 401)
        self.assertEqual(len(self.captured_requests), 0)

    def test_paid_model_rejected_before_upstream(self) -> None:
        status, _, body = self._dispatch(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-proxy-token"},
            json_body={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 400)
        self.assertIn("non-free model", json.loads(body)["error"]["message"])
        self.assertEqual(len(self.captured_requests), 0)

    def test_free_model_forwarded_with_real_key_and_no_alert_when_zero_cost(self) -> None:
        status, _, body = self._dispatch(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-proxy-token"},
            json_body={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "ok")
        self.assertEqual(
            self.captured_requests[0].headers["Authorization"],
            "Bearer sk-or-real-secret",
        )
        self.assertEqual(self.alerts, [])

    def test_nonzero_response_cost_triggers_alert(self) -> None:
        self.mock_response_cost = 0.0025
        status, _, _ = self._dispatch(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-proxy-token"},
            json_body={
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(self.alerts), 1)
        self.assertIn("0.002500", self.alerts[0][1])

    def test_key_usage_increase_triggers_alert_and_usage_endpoint(self) -> None:
        self.mock_key_usage = 0.015
        status, _, body = self._dispatch(
            "GET",
            "/v1/usage",
            headers={"Authorization": "Bearer local-proxy-token"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["usage_usd"], 0.015)
        self.assertEqual(len(self.alerts), 1)
        self.assertIn("0.015000", self.alerts[0][1])

    def test_streaming_forwarded(self) -> None:
        with self.subTest("stream"):
            status, headers, body = self._dispatch(
                "POST",
                "/v1/chat/completions",
                headers={"Authorization": "Bearer local-proxy-token"},
                json_body={
                    "model": "openrouter/free",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(headers.get("Content-Type"), "text/event-stream")
            self.assertIn(b"data: [DONE]", body)

    def test_models_filtered_to_free_only(self) -> None:
        status, _, body = self._dispatch(
            "GET",
            "/v1/models",
            headers={"Authorization": "Bearer local-proxy-token"},
        )
        self.assertEqual(status, 200)
        ids = [item["id"] for item in json.loads(body)["data"]]
        self.assertEqual(
            ids,
            ["openrouter/free", "meta-llama/llama-3.3-70b-instruct:free"],
        )


if __name__ == "__main__":
    unittest.main()
