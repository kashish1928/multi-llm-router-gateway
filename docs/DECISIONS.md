# Design Decisions & Assumptions

This log records what was verified, against which source, and every assumption made
where verification was not possible. Date of build: 2026-09-29.

## 1. Sources consulted

`openrouter.ai` and `langfuse.com` were **blocked by the build environment's egress
proxy**, so their pages could not be fetched directly. Verification therefore used (a)
web-search excerpts of the official pages, and (b) the **installed package source code**
of each library, which is authoritative for the exact version pinned in `uv.lock`.

| Topic | Source | Used for |
|---|---|---|
| OpenRouter model fallbacks | https://openrouter.ai/docs/guides/routing/model-fallbacks (search excerpt) | `models` array: ordered fallbacks tried on rate-limit, downtime, moderation errors |
| OpenRouter limits | https://openrouter.ai/docs/api/reference/limits (search excerpt) | Free models: 20 req/min; 50 req/day unfunded, 1000 req/day after ≥$10 credits |
| OpenRouter `GET /api/v1/key` | https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key (search excerpt) | `data.is_free_tier`, `data.free_model_daily_requests{used,limit,remaining}` |
| OpenRouter `GET /api/v1/models` | https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties (search excerpt) | `{data: Model[]}` with `id`; `context_length`, `pricing.prompt/completion` (USD per token) |
| OpenRouter reasoning | https://openrouter.ai/docs/api_reference/parameters, https://openrouter.ai/openai/gpt-oss-20b/api (search excerpts) | `reasoning: {"effort": ...}`; gpt-oss supports low/medium/high |
| Langfuse Python SDK v4 | installed `langfuse==4.15.6` source (`langfuse/__init__.py`, `_client/client.py`, `_client/propagation.py`, `_client/environment_variables.py`); https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4 (search excerpt) | `Langfuse(public_key, secret_key, base_url)`, `start_as_current_observation(as_type=...)`, `propagate_attributes(...)`, `flush()`, `shutdown()`; env `LANGFUSE_BASE_URL` (legacy `LANGFUSE_HOST` fallback) |
| OpenAI Python SDK | installed `openai==3.22.0` source (`_client.py`, `_httpx2.py`, `_exceptions.py`, `_constants.py`) | `AsyncOpenAI(base_url, max_retries, timeout, default_headers, http_client)`; defaults `max_retries=2`, `timeout=600s/connect 5s` (overridden); exception hierarchy |
| uv | `uv 0.8.17` CLI; https://docs.astral.sh/uv/guides/integration/docker/ | `uv init --app`, `uv python pin`, `uv add [--dev]`, `uv sync --locked [--no-dev]`, Docker layer pattern |

### Pinned versions (from `uv.lock`)
fastapi 0.142.0 · starlette 1.7.0 · uvicorn 0.54.0 · pydantic 2.13.5 · pydantic-settings 2.15.0 ·
openai 3.22.0 · httpx 0.28.1 · httpx2 2.13.1 · loguru 0.7.3 · langfuse 4.15.6 · tenacity 9.1.4 ·
pytest 9.1.1 · pytest-asyncio 1.4.0 · respx 0.23.1 · pytest-cov 7.1.0 · ruff 0.16.9 · mypy 2.3.1 · Python 3.12

## 2. Notable findings during verification

1. **openai 3.x uses `httpx2`, not `httpx`.** respx only patches `httpx`. The SDK still
   accepts a legacy `httpx.AsyncClient` as `http_client` (see `openai/_httpx2.py`), so the
   gateway builds one explicitly (`app/llm/client.py`). This keeps explicit connect/read
   timeouts and makes every upstream call mockable. Probed and confirmed before writing code.
   The SDK's own `timeout` is typed as `httpx2.Timeout`, so `httpx2` is a declared dependency.
2. **`ANTHROPIC_BASE_URL` collision.** Many environments (including the one this was built
   in) set `ANTHROPIC_BASE_URL=https://api.anthropic.com` for the Anthropic SDK. An earlier
   field name `anthropic_base_url` silently picked it up and dropped the `/v1` path. The
   break-glass settings are namespaced (`BREAK_GLASS_ANTHROPIC_BASE_URL`, etc.) and tests
   clear all Settings env vars (`tests/conftest.py::_isolate_env`).
3. **`.env` inline comments.** python-dotenv reads `KEY=   # note` as the value `# note`.
   With the spec's inline-comment `.env.example`, `GATEWAY_API_KEYS` would have accepted the
   comment text as a valid bearer key. `.env.example` uses own-line comments, and `Settings`
   drops any value starting with `#` (regression-tested).
4. **tenacity + streaming.** A `StopAsyncIteration` raised inside tenacity's `async for`
   silently ends the retry loop. Empty upstream streams are converted to
   `EmptyStreamError` before they can escape (regression-tested).

## 3. Assumptions (unverifiable here, config-driven so they are easy to change)

| # | Assumption | Mitigation |
|---|---|---|
| A1 | Model IDs `openai/gpt-oss-20b:free`, `openai/gpt-oss-20b`, `meta-llama/llama-3.1-8b-instruct`, `google/gemini-3.1-pro-preview`, `anthropic/claude-sonnet-5.5` exist on OpenRouter. | Startup `GET /models` fails fast with the list of missing IDs. IDs live only in `app/model_registry.json` (or `MODEL_REGISTRY_PATH`). |
| A2 | Registry prices/context windows are placeholders. | With `SYNC_REGISTRY_FROM_OPENROUTER=true` (default) they are overwritten from `/models` at startup (`context_length`, `pricing.*` × 1e6). Free models always stay at $0. |
| A3 | The `models` array carries only the *fallbacks* after `model` (primary). | Matches the documented `model` + `models` fallback example. If OpenRouter expects the primary repeated, only `_build_kwargs` changes. |
| A4 | `response.model` is the model that actually served the request. Free and paid gpt-oss share a base ID; if OpenRouter reports the free variant without `:free`, the gateway would report a (false) fallback. | Documented limitation; the attempt log shows which hop succeeded. |
| A5 | Lowest reasoning effort for gpt-oss-20b is `low` (`reasoning.effort`). Some docs list `minimal`/`none` for other models; gpt-oss documents low/medium/high. | Per-model `reasoning_effort` field in the registry. |
| A6 | `X-Degraded` is set only for COMPLEX → `openai/gpt-oss-20b` (per spec). | Per-tier `degraded_models` list in the registry. |
| A7 | Break-glass providers use their OpenAI-compatible endpoints (`https://api.anthropic.com/v1/`, `https://generativelanguage.googleapis.com/v1beta/openai/`). Model IDs are not guessed. | Disabled by default; requires `BREAK_GLASS_ENABLED=true` **and** an explicit model ID per provider. |
| A8 | Llama Prompt Guard 2 (86M/22M) is licence-gated on Hugging Face and has a 512-token window; last label = malicious. | Optional (`GUARDRAIL_CLASSIFIER_ENABLED`), lazy-imports `transformers`/`torch` (not dependencies). Scores 500-token windows with 125-token overlap, takes the max. |
| A9 | No embedding endpoint is assumed for the semantic cache. | Default embedder is a local hashed char-trigram vectorizer; `Embedder` protocol allows a real model. |

## 4. Design decisions

- **Two-layer fallback.** Each hop sends `model=<hop>` + `models=<remaining chain>` (native
  fallback). Only when OpenRouter itself returns 429/5xx/timeout does the gateway retry the
  hop (full-jitter exponential backoff, `Retry-After`/`retry-after-ms` honoured, capped by
  `RETRY_AFTER_CAP_S`; beyond the cap it walks immediately) and then walk the chain.
  Non-429 4xx are never retried or walked. SDK retries are set to 0 so they don't stack.
- **Free-tier quota.** Slots are reserved *before* each free call (failed attempts count
  upstream). Free-model 429s are not retried; the gateway walks straight to paid
  `openai/gpt-oss-20b`. Exhausted caps remove the free model both as a hop and from
  native `models` lists. Counters are in-process (see limitations).
- **Circuit breaker** per model: opens after `CIRCUIT_FAILURE_THRESHOLD` consecutive
  retryable failures, half-opens after `CIRCUIT_COOLDOWN_S` with a single trial request.
- **Context fit** is enforced twice: the router escalates the tier, and the dispatcher
  filters every candidate (including native fallbacks) by `window >= est_tokens + max_tokens`.
  An explicit override that fits nowhere returns 413 without any upstream call.
- **Token estimate** = `ceil(chars / 4)`, the common rule of thumb for English with
  BPE tokenizers. Used only for routing and context fit; billing uses upstream `usage`.
- **Intent keywords** are matched on the last user turn's head+tail (2k chars each) so
  words inside a large pasted document don't dominate routing.
- **Overrides** accept only allowlisted aliases (`gpt-oss|gemini|claude`) via
  `X-Model-Override`, `model_override`, or the OpenAI `model` field (`auto` = route).
  Any other string is 400. Client-supplied OpenRouter routing params (`models`, `provider`,
  `route`, …) are dropped; only an allowlist of OpenAI params is forwarded.
- **Free-tier privacy.** The SIMPLE tier gets a generic system prompt; the gateway system
  prompt is withheld unless `FREE_TIER_SEND_GATEWAY_SYSTEM_PROMPT=true`. Gateway
  instructions are always a separate `system` message, never concatenated with user text.
- **Guardrails** normalize (NFKC, strip zero-width/bidi/tag chars, casefold) and then
  run: regex heuristics → base64 payload decoding/size → optional ML classifier. Only
  `user`/`tool` content is scanned. Clients get a generic `input_rejected`; the rule is
  logged and traced.
- **Output guard** withholds responses that contain configured secrets or 60-char windows
  of the gateway system prompt (502 `output_rejected`). For streams it checks a rolling
  4 KB window and terminates the stream with an error event.
- **Tracing** records one `TraceRecord` per request and exports it after completion as a
  Langfuse trace (root span + guardrail + route + one generation per upstream attempt).
  Exporter exceptions are swallowed (`SafeTracer`); flush/shutdown run in the lifespan.
  Keys/users are HMAC-hashed; raw content only with `TRACE_RAW_CONTENT=true`.
- **Streaming**: fallback applies only until the first chunk is received. After that, an
  upstream error becomes an SSE `error` event (`stream_interrupted`).
- **Cascade mode** (off by default, non-streaming, never with an override): tiers run
  cheapest-first and escalate on empty / refusal / malformed-JSON answers or upstream 5xx.

## 5. Known limitations

- Rate limits, free-tier counters, breakers and the semantic cache are **per process**.
  Multi-replica deployments need a shared store (e.g. Redis) behind the same interfaces.
- Langfuse child observation timestamps reflect export time; true per-attempt latency is
  in each observation's metadata.
- Heuristic routing eval accuracy is 97.2% (35/36) on a set authored alongside the
  heuristics. It is **not** a held-out benchmark. Known miss: definitional questions that
  mention strong keywords ("what is an algorithm?") route to INTERMEDIATE.
- Token estimates ignore images/tool schemas.
- The Docker image was not built in the authoring environment (no Docker daemon); CI builds it.
