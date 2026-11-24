# OpenRouter Starter

Minimal Python integration using the OpenAI SDK pointed at `https://openrouter.ai/api/v1`.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys) and set `OPENROUTER_API_KEY`:

   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

   Or copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` (`.env` is gitignored).

## Run

`main.py` enforces **free models only**: it only allows `openrouter/free` (OpenRouter's free-model router, the default) or model slugs ending with `:free` from [openrouter.ai/models](https://openrouter.ai/models). Any paid model slug is rejected before a request is sent.

```bash
python3 main.py
```

With a custom prompt and specific free model slug:

```bash
python3 main.py "Explain async/await in two sentences." --model openrouter/free
```
