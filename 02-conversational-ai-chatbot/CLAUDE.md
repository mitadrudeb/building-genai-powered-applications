# Project: Multi-provider LLM chatbot (Project 02)

A Gradio chatbot that talks to a local open-source model (Qwen3.5 with Hugging Face
transformers) or, later, to frontier APIs (OpenAI, Anthropic, Gemini). Portfolio project.
It lives in the folder 02-conversational-ai-chatbot of the monorepo.
Work branch: rebuild-02-chatbot.

## Decisions
- UI: Gradio ChatInterface with streaming. No Flask, Docker or LangChain.
- Message format: {"role": "user" or "assistant", "content": "plain text"}.
- Only local_model.py changes messages into the format Qwen needs.
- Keep the code simple. Add abstractions (config file, provider interface) only when
  they are really needed (Stage 4).
- API keys come from .env, Colab Secrets or a session-only UI field. Never commit keys.
- The model only runs in Colab (no GPU on the laptop). The laptop is for editing, git
  and tests.

## Rules
- Work one stage at a time. The reader is a beginner: explain simply.
- Never commit, push or delete files without asking first.
- Tests must use fakes for the model and the APIs (no network, no GPU).
- Never write results or numbers in docs that were not measured.

## Writing style for project files
- Comments, docstrings and markdown read like a careful beginner's own notes:
  simple words, short sentences, explain why (not every obvious line).
- Semi-professional: clear and accurate, no hype, no jokes.

## Status
Stage 2 done (terminal chat works in Colab). Next: Stage 3 (Gradio UI with streaming).