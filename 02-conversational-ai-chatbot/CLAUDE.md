# Project 02: Conversational AI Chatbot (open-source vs frontier)

A learning project, shown as three Jupyter notebooks in folder
02-conversational-ai-chatbot of the monorepo
github.com/mitadrudeb/building-genai-powered-applications.
Work branch: rebuild-02-chatbot. Project 01 must not be touched.

## Notebooks
1. 01_open_source_chatbot.ipynb - Qwen/Qwen3.5-2B with Hugging Face transformers,
   Colab T4 GPU. Tokens, chat template, generate(), memory, system prompt,
   streaming with a thread, Gradio UI.
2. 02_frontier_chatbot.ipynb - OpenAI API (no GPU). Key from Colab Secrets,
   messages, system prompt, streaming, token usage and cost, errors, Gradio UI.
3. 03_final_chatbot_app.ipynb - one Gradio app with a dropdown (open-source or
   OpenAI) and a side-by-side comparison.

## Decisions
- Notebooks are the project. No Python package, no adapter classes.
- Keep code simple: one idea per cell, explained in a markdown cell before it.
- Frontier side: OpenAI only (tested live). Anthropic and Gemini are only
  mentioned in the README as possible next steps.
- Gradio 6 ChatInterface; history items are {"role", "content"}; content can be
  a string or a list of parts.
- Notebooks are built and run in Google Colab, then saved to the branch.
- Keep cell outputs in saved notebooks, but never put an API key in a cell.
  Keys only in Colab Secrets or a local .env file (git-ignored).
- A public Gradio share link must not let strangers use my OpenAI key.

## Writing style for project files
- Markdown and comments read like a careful beginner's own notes: simple words,
  short sentences, explain why. Semi-professional: clear, accurate, no hype.
- Never write results or numbers that I did not measure myself.

## Rules
- One step at a time. Never commit, push or delete files without asking me.
- Do not touch files outside 02-conversational-ai-chatbot unless I ask.
- Ignore the untracked file signal-desktop-keyring.gpg. Never add it.

## Status
Restructured to notebooks. Next: build notebook 01 in Colab.
