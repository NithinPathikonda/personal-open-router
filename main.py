"""End-to-end OpenRouter chat completion example using the OpenAI SDK."""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"


def ensure_free_model(model: str) -> str:
    """Reject any model slug that is not explicitly free on OpenRouter."""
    cleaned = model.strip()
    if cleaned != "openrouter/free" and not cleaned.endswith(":free"):
        raise RuntimeError(
            f"Refusing to use non-free model '{cleaned}'. "
            "Only 'openrouter/free' or slugs ending with ':free' are allowed."
        )
    return cleaned


def get_api_key() -> str:
    """Return OPENROUTER_API_KEY from the environment or exit with instructions."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "Create a key at https://openrouter.ai/keys and set it in your environment:\n"
            "  export OPENROUTER_API_KEY='sk-or-v1-...'\n"
            "or copy .env.example to .env and add it there (never commit .env).",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def send_chat_request(model: str, prompt: str) -> tuple[str, str]:
    """Send a free-tier chat completion request to OpenRouter and return (used_model, reply)."""
    safe_model = ensure_free_model(model)
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
        default_headers={
            "X-OpenRouter-Title": "OpenRouter Starter",
        },
    )

    try:
        response = client.chat.completions.create(
            model=safe_model,
            messages=[{"role": "user", "content": prompt}],
        )
    except AuthenticationError as exc:
        raise RuntimeError(
            "Authentication failed. Verify your OPENROUTER_API_KEY at https://openrouter.ai/keys."
        ) from exc
    except RateLimitError as exc:
        raise RuntimeError(
            "Rate limit exceeded on free tier. Wait a moment and try again."
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to OpenRouter ({OPENROUTER_BASE_URL}): {exc}"
        ) from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"OpenRouter API error (HTTP {exc.status_code}): {exc.message}"
        ) from exc

    if not response.choices:
        raise RuntimeError("OpenRouter returned a response with no choices.")

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenRouter returned an empty message.")

    used_model = getattr(response, "model", None) or safe_model
    return used_model, content


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a request to OpenRouter.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="In one sentence, what is OpenRouter?",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        help="Model slug from https://openrouter.ai/models",
    )
    args = parser.parse_args()

    try:
        used_model, reply = send_chat_request(model=args.model, prompt=args.prompt)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"[{used_model}]\n{reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
