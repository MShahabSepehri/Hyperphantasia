import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def load_model(model_name, device='cuda', cache_dir=None):
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map=device,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map=device,
        cache_dir=cache_dir
    )
    return model, processor

def create_content(prompt):
    text = None
    image_list = []
    for c in prompt:
        if c[0] == "image":
            image_list.append(Image.open(c[1]).convert("RGB"))
        else:
            text = c[1]
    return image_list, text

@torch.no_grad()
def ask_question(model, prompt, processor, num_beams, max_new_tokens, top_p, temperature):
    image, text = create_content(prompt)

    inputs = processor.process(images=image, text=text)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": (temperature > 0),
        "top_p": top_p,        
        "num_beams": num_beams,
    }
    if temperature <= 0:
        params["temperature"] = None
    gc = GenerationConfig(
            **params,
            stop_strings="<|endoftext|>",
    )
    output = model.generate_from_batch(inputs, gc, tokenizer=processor.tokenizer)
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text
