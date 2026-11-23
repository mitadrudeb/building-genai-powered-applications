# Project: Multi-provider LLM chatbot

Gradio chatbot that talks to a local open-source model (Qwen3.5 via Hugging Face
transformers) or to frontier APIs (OpenAI, Anthropic, Gemini). Portfolio project.

## Decisions
- UI: Gradio ChatInterface with streaming. No Flask, Docker or LangChain.
- Providers share one interface (adapter pattern): messages in -> streamed text out.
- Internal message format: {"role": "user"|"assistant", "content": str}.
- Local model loads lazily, once. Device is auto-detected (CUDA or CPU).
- API keys come from .env, Colab Secrets, or a session-only UI field. Never commit keys.
- Model IDs live in config, never hard-coded in logic.
- Colab is the developer's only GPU runtime; the laptop runs tests and API-only mode.
- Lives in folder 02-conversational-ai-chatbot of the monorepo; developed on branch rebuild-02-chatbot.

## Rules
- Work one stage at a time. Explain concepts simply (reader is a beginner).
- Never commit, push or delete files without asking first.
- Tests must mock models and APIs (no network, no GPU).
- Do not invent results or metrics in docs.

## Status
Stage 0 done. Stage 1 in progress (skeleton).