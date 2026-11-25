# OpenRouter Starter & Local Free-Model Proxy

Minimal Python integration and a **local OpenAI-compatible proxy** (`server.py`) so all your other projects can use OpenRouter's **free models** without duplicating your real `OPENROUTER_API_KEY`.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` (`.env` is gitignored) and fill in:
   - `OPENROUTER_API_KEY` from [openrouter.ai/keys](https://openrouter.ai/keys)
   - `PROXY_API_KEY` (a local secret token for your other projects, e.g. `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`)

## Option A: Run as a Local Proxy for All Projects (Recommended)

Start the proxy on `127.0.0.1:8000`:

```bash
python3 server.py
```

What this guarantees:
- **Single secret store:** Only this repo's `.env` holds your real `OPENROUTER_API_KEY`.
- **Local auth:** Other projects authenticate using `PROXY_API_KEY` (`Authorization: Bearer <PROXY_API_KEY>`).
- **Free models only:** Requests for any model other than `openrouter/free` or `*:free` are rejected with HTTP `400` before reaching OpenRouter.
- **Instant charge alerts:** If OpenRouter ever reports any non-zero cost (`usage.cost > 0`) or key spend increases on `/api/v1/key`, the proxy logs an `[ALERT]`, triggers a macOS desktop notification (`NOTIFY_MACOS=true`), and optionally posts to `ALERT_WEBHOOK_URL` (Discord/Slack/ntfy.sh).
- **Localhost-only:** Binds to `127.0.0.1` by default.

> **Tip:** In the OpenRouter dashboard under **Keys** / **Guardrails**, also set your API key's credit limit to **`$0`** (right now it may default to a higher limit like `$100`) as a hard server-side safety cap.

### Using it from any other project

**Python (`openai` SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-PROXY_API_KEY",
)

resp = client.chat.completions.create(
    model="openrouter/free",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

**JavaScript / TypeScript (`openai` SDK):**

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8000/v1",
  apiKey: process.env.PROXY_API_KEY,
});
```

**`curl`:**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer your-PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"openrouter/free","messages":[{"role":"user","content":"Hi"}]}'
```

## Option B: Run the Standalone CLI

`main.py` also enforces **free models only**:

```bash
python3 main.py
python3 main.py "Explain async/await in two sentences." --model openrouter/free
```
