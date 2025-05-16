import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter.*",
    category=UserWarning,
)


def load_model(model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", device='cuda', cache_dir=None):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    return model, processor


def creat_message(prompt):
    content = []
    for c in prompt:
        if c[0] == "text":
            content.append({"type": c[0], "text": c[1]})
        else:
            content.append({"type": c[0], "url": c[1]})
    message = [{"role": "user", "content": content}]
    return message

@torch.no_grad()
def ask_question(model, processor, prompt, temperature, max_new_tokens=2048):
    message = creat_message(prompt)

    inputs = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device, torch.float16)

    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": (temperature > 0),
    }
    if temperature <= 0:
        del params["temperature"]
    output = model.generate(**inputs, **params)
    response = processor.decode(output[0], skip_special_tokens=True)
    response = response.split('assistant\n')[-1]
    return response