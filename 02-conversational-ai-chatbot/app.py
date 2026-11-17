"""
🤖 Project 02: Conversational AI Chatbot
========================================
An open-source, empathetic dialogue assistant powered by `facebook/blenderbot-400M-distill`.
Features multi-turn sliding window memory, tunable generation parameters, and both
Gradio Web UI and interactive CLI modes.

Author: Mitadru Deb
Repository: building-genai-powered-applications
License: MIT
"""

import sys
import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------
# 1. Model Configuration & Initialization
# ---------------------------------------------------------
MODEL_NAME = "facebook/blenderbot-400M-distill"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Initializing Conversational AI Assistant...")
print(f"📦 Model:  {MODEL_NAME}")
print(f"💻 Device: {DEVICE.upper()}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    print("✅ Model & Tokenizer loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


# ---------------------------------------------------------
# 2. Text Extraction Helper (Gradio 4 / 5 / 6 Compatibility)
# ---------------------------------------------------------
def extract_text(item) -> str:
    """
    Safely extract plain text string from str, list, or dict.
    Gradio 5/6 can wrap messages as dicts {'text': ...} or lists ['...'].
    """
    if not item:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        if "text" in item:
            return extract_text(item["text"])
        if "content" in item:
            return extract_text(item["content"])
    if isinstance(item, (list, tuple)):
        parts = [extract_text(sub) for sub in item]
        return " ".join(p for p in parts if p).strip()
    return str(item).strip()


# ---------------------------------------------------------
# 3. Conversational Generation Engine
# ---------------------------------------------------------
def generate_bot_response(
    message,
    history,
    temperature: float = 0.7,
    top_p: float = 0.85,
    repetition_penalty: float = 1.3,
    max_tokens: int = 60,
    history_turns: int = 6,
) -> str:
    """
    Generate a coherent, context-aware response using multi-turn conversational memory.

    Parameters:
    - message: Latest user input (string or dict).
    - history: Conversation history (list of message dicts or legacy tuples).
    - temperature: Sampling randomness (lower = focused, higher = creative).
    - top_p: Nucleus sampling probability threshold.
    - repetition_penalty: Penalizes repetitive phrases and token loops.
    - max_tokens: Maximum new tokens to generate per response.
    - history_turns: Number of preceding dialogue turns retained in context.
    """
    user_text = extract_text(message)
    if not user_text:
        return "Please say something so we can chat!"

    # 1. Parse history into individual conversational utterances
    utterances = []
    if history:
        for turn in history:
            if isinstance(turn, dict):
                # Gradio messages format: {'role': 'user'/'assistant', 'content': ...}
                text = extract_text(turn.get("content"))
                if text:
                    utterances.append(text)
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                # Gradio legacy tuples format: (user_msg, bot_msg)
                user_msg = extract_text(turn[0])
                bot_msg = extract_text(turn[1])
                if user_msg:
                    utterances.append(user_msg)
                if bot_msg:
                    utterances.append(bot_msg)

    # 2. Sliding window memory: keep only recent dialogue turns (N turns = 2N utterances)
    max_history_count = int(history_turns) * 2
    recent_utterances = utterances[-max_history_count:]
    recent_utterances.append(user_text)

    # 3. BlenderBot native delimiter: double space ("  ") between utterances
    prompt = "  ".join(recent_utterances)

    # 4. Tokenize with safe truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    # 5. Generate conversational response with sampling controls
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        no_repeat_ngram_size=3,
        repetition_penalty=float(repetition_penalty),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
    )

    # 6. Decode output tokens back into natural language
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response


# ---------------------------------------------------------
# 4. Interactive CLI Mode (Beginner-Friendly Terminal Chat)
# ---------------------------------------------------------
def run_cli_chat():
    """Run an interactive terminal session with sliding-window memory."""
    print("=" * 60)
    print("🤖 Conversational AI Assistant (CLI Mode)")
    print("Type your message and press Enter.")
    print("Commands: 'clear' to reset memory, 'exit' or 'quit' to quit.")
    print("=" * 60 + "\n")

    history = []

    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Session ended. Happy coding!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("[Memory cleared]\n")
            continue

        bot_response = generate_bot_response(
            message=user_input,
            history=history,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.3,
            max_tokens=60,
            history_turns=6,
        )

        print(f"Bot  > {bot_response}\n")

        # Save to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": bot_response})


# ---------------------------------------------------------
# 5. Bonus: Frontier Model Integration (OpenAI API)
# ---------------------------------------------------------
# Anyone with their own OpenAI API key can test this section with frontier
# models such as GPT-4o-mini or GPT-4o.
#
# Prerequisites:
#   pip install openai
#
# Setup:
#   export OPENAI_API_KEY="your-api-key-here"   (Linux / macOS)
#   $env:OPENAI_API_KEY="your-api-key-here"     (Windows PowerShell)
#
# import os
# from openai import OpenAI
#
# def generate_openai_response(
#     message,
#     history,
#     model_name: str = "gpt-4o-mini",
#     temperature: float = 0.7,
#     max_tokens: int = 150,
#     history_turns: int = 6,
# ) -> str:
#     """
#     Generate a response using OpenAI's Chat Completions API with conversational memory.
#     Showcases frontier model API integration and structured message formatting.
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         return (
#             "⚠️ OPENAI_API_KEY environment variable not found. "
#             "Please set your OpenAI API key to test frontier model completions."
#         )
#
#     client = OpenAI(api_key=api_key)
#     user_text = extract_text(message)
#     if not user_text:
#         return "Please enter a message!"
#
#     # Construct the messages payload with a system prompt and sliding window history
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful, empathetic, and engaging AI assistant.",
#         }
#     ]
#
#     past_messages = []
#     if history:
#         for turn in history:
#             if isinstance(turn, dict):
#                 role = turn.get("role", "user")
#                 text = extract_text(turn.get("content"))
#                 if text:
#                     past_messages.append({"role": role, "content": text})
#             elif isinstance(turn, (list, tuple)) and len(turn) == 2:
#                 u_text = extract_text(turn[0])
#                 b_text = extract_text(turn[1])
#                 if u_text:
#                     past_messages.append({"role": "user", "content": u_text})
#                 if b_text:
#                     past_messages.append({"role": "assistant", "content": b_text})
#
#     # Apply sliding window (keep last N turns = 2N messages)
#     max_history_count = int(history_turns) * 2
#     messages.extend(past_messages[-max_history_count:])
#     messages.append({"role": "user", "content": user_text})
#
#     try:
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=messages,
#             temperature=float(temperature),
#             max_tokens=int(max_tokens),
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"OpenAI API Error: {e}"


# ---------------------------------------------------------
# 6. Gradio Web Interface
# ---------------------------------------------------------
def create_gradio_app():
    """Build and configure the Gradio ChatInterface."""
    custom_description = (
        "### 💬 Open-Source Conversational Dialogue Agent\n"
        "Powered by **Facebook BlenderBot-400M-distill** — a lightweight (~400M parameter) Seq2Seq model "
        "trained specifically on multi-turn empathetic and engaging dialogue. "
        "Runs 100% locally on CPU/GPU without external API keys!"
    )

    examples = [
        ["Hello! What is your favorite book?"],
        ["If you could visit any planet, where would you go?"],
        ["Can you explain how neural networks learn in simple terms?"],
        ["What are some fun hobbies to pick up on the weekend?"],
        ["Tell me an interesting fact about space exploration."],
    ]

    chat_interface = gr.ChatInterface(
        fn=generate_bot_response,
        title="🤖 Conversational AI Chatbot",
        description=custom_description,
        examples=examples,
        additional_inputs=[
            gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.05,
                label="🌡️ Temperature (Creativity)",
                info="Higher = creative & diverse; Lower = focused & deterministic.",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.85,
                step=0.05,
                label="🎯 Top-P (Nucleus Sampling)",
                info="Cumulative probability threshold for candidate token selection.",
            ),
            gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.3,
                step=0.05,
                label="🔄 Repetition Penalty",
                info="Penalizes repetitive words or token cycles.",
            ),
            gr.Slider(
                minimum=20,
                maximum=150,
                value=60,
                step=10,
                label="📏 Max New Tokens",
                info="Maximum number of tokens to generate per response.",
            ),
            gr.Slider(
                minimum=2,
                maximum=12,
                value=6,
                step=2,
                label="🧠 Context Memory Window (Turns)",
                info="Number of preceding user-bot dialogue turns retained in prompt context.",
            ),
        ],
        additional_inputs_accordion=gr.Accordion(
            "⚙️ Generation & Memory Settings", open=False
        ),
    )

    return chat_interface


# ---------------------------------------------------------
# 7. Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational AI Chatbot")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run chatbot directly in interactive terminal CLI mode",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public shareable Gradio link (ideal for Google Colab)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio web server on (default: 7860)",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli_chat()
    else:
        app = create_gradio_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
        )
