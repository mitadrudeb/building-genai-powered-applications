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
print(f"📦 Model: {MODEL_NAME}")
print(f"💻 Device: {DEVICE.upper()}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    print("✅ Model & Tokenizer loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


# ---------------------------------------------------------
# 2. Conversational Generation Engine
# ---------------------------------------------------------
def generate_bot_response(
    message: str,
    history: list,
    temperature: float = 0.7,
    top_p: float = 0.85,
    repetition_penalty: float = 1.3,
    max_tokens: int = 60,
    history_turns: int = 6,
) -> str:
    """
    Generate a coherent, context-aware response using multi-turn conversational memory.

    Parameters:
    - message (str): The latest user input.
    - history (list): Chat history (supports both Gradio 5/6 message dicts and legacy tuples).
    - temperature (float): Sampling randomness (lower = deterministic, higher = creative).
    - top_p (float): Nucleus sampling probability threshold.
    - repetition_penalty (float): Penalizes repeated phrases and token loops.
    - max_tokens (int): Maximum new tokens to generate per response.
    - history_turns (int): Maximum number of previous dialogue turns to retain.
    """
    if not message or not message.strip():
        return "Please say something so we can chat!"

    # 1. Parse conversational history into individual utterances
    utterances = []
    if history:
        for turn in history:
            if isinstance(turn, dict):
                # Modern Gradio format: [{'role': 'user', 'content': '...'}, ...]
                content = turn.get("content", "").strip()
                if content:
                    utterances.append(content)
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                # Legacy Gradio format: [(user_msg, bot_msg), ...]
                user_msg, bot_msg = turn
                if user_msg:
                    utterances.append(str(user_msg).strip())
                if bot_msg:
                    utterances.append(str(bot_msg).strip())

    # 2. Apply sliding window memory (keep last N dialogue turns)
    max_history_count = int(history_turns) * 2
    recent_utterances = utterances[-max_history_count:]
    recent_utterances.append(message.strip())

    # 3. Format prompt using BlenderBot's native dialogue delimiter (double space: "  ")
    prompt = "  ".join(recent_utterances)

    # 4. Tokenize with safe truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    # 5. Generate conversational response with sampling controls
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        no_repeat_ngram_size=3,
        repetition_penalty=float(repetition_penalty),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p)
    )

    # 6. Decode output tokens to natural language
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response


# ---------------------------------------------------------
# 3. Interactive CLI Mode (Beginner-Friendly Terminal Chat)
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

        # Generate response using sliding window history
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

        # Record exchange in history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": bot_response})


# ---------------------------------------------------------
# 4. Gradio Web Interface
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
                info="Higher values = more creative & diverse; Lower = focused & deterministic.",
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
# 5. Application Entry Point
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
