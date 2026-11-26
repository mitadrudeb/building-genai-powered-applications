"""Helpers to load a small open-source chat model and get replies from it."""

from threading import Thread
from transformers import AutoModelForMultimodalLM, AutoProcessor, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen3.5-2B"

def load_model(model_id=MODEL_ID):
    """Download the model (only the first time) and load it into memory."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    return processor, model

def _prepare_inputs(processor, model, history, system_prompt=""):
    """Turn the chat history into tokens that the model can read.

    history is a list like [{"role": "user", "content": "Hi"}, ...].
    """

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.extend(history)

    messasges = [
        {
            "role": m["role"],
            "content": [
                {"type": "text", 
                 "text": m["content"]}
            ]
        } for m in messages
    ]

    return processor.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)


def get_reply(processor, model, history, system_prompt="",
              temperature=1.0, max_new_tokens=512):
    """Wait until the whole answer is ready, then return it as one string."""
    inputs = _prepare_inputs(processor, model, history, system_prompt)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=1.0,
    )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()

def stream_reply(processor, model, history, system_prompt="",
                 temperature=1.0, max_new_tokens=512):
    """Give back the answer piece by piece. Every yield is the full text so far."""
    inputs = _prepare_inputs(processor, model, history, system_prompt)

    # The streamer collects the new text while the model is still writing.
    # timeout: if the generation thread crashes, I get an error after 60 s
    # and the page does not wait forever.
    
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0,
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=20,
        top_p=1.0,
    )

    # Run the model in a separate thread so that we can yield partial outputs.
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield the text as it comes in.
    text= ""
    for new_text in streamer:
        text += new_text
        yield text.strip()

