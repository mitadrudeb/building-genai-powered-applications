
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-2B"

def load_model(model_id=MODEL_ID):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    return processor, model

def get_reply(processor, model, history):

    
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in history
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=20,
    )
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]  # keep only the new text
    return processor.decode(new_ids, skip_special_tokens=True).strip()