"""Gradio website for the chatbot."""
import gradio as gr

from chatbot.local_model import load_model, stream_reply

# I keep the model in global variables so it is loaded only one time.
_processor = None
_model = None

def get_model():
    """Load the model on the first call. Later calls reuse it."""
    global _processor, _model
    if _model is None:
        _processor, _model = load_model()
    return _processor, _model

def to_text(content):
    """Gradio can send the content as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )


def chat(message, history, system_prompt, temperature, max_new_tokens):
    """Gradio calls this function every time the user sends a message."""
    processor, model = get_model()

    # Build plain-text messages from the history that Gradio gives me.
    # The new message is not in the history yet, so I add it at the end.
    messages = [
        {"role": m["role"], "content": to_text(m["content"])} for m in history
    ]
    messages.append({"role": "user", "content": message})

    yield from stream_reply(
        processor, model, messages, system_prompt, temperature, int(max_new_tokens)
    )


demo = gr.ChatInterface(
    fn=chat,
    title="Chatbot with a local open-source model",
    description="The model runs on this machine. The first message can take longer.",
    additional_inputs=[
        gr.Textbox(
            value="You are a helpful assistant. Answer briefly and clearly.",
            label="System prompt",
        ),
        # Minimum 0.1: a temperature of 0 can cause an error when sampling
        gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(64, 1024, value=512, step=64, label="Max new tokens"),
    ],
)

if __name__ == "__main__":
    demo.launch()