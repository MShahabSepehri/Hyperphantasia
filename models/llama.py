import torch
import os
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import MllamaForConditionalGeneration, AutoProcessor

def load_model(model_id, device='cuda', cache_dir=None):
    load_dotenv(override=True)
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token is not None:
        login(token)
    else:
        login()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir,)
    # model.to(device)
    return model, processor


def create_content(prompt):
    content = []
    image_list = []
    for c in prompt:
        if c[0] == "image":
            content.append({"type": "image"})
            image_list.append(Image.open(c[1]))
        else:
            content.append({"type": "text", "text": c[1]})
    messages = [{"role": "user", "content": content}]
    return messages, image_list

@torch.no_grad()
def ask_question(model, processor, prompt, temperature=0.2, top_p=None, num_beams=1, max_new_tokens=100):

    messages, image_list = create_content(prompt)
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image_list, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": (temperature > 0),
        "top_p": top_p,        
        "num_beams": num_beams,
    }
    if temperature <= 0:
        params["temperature"] = None

    output = model.generate(**inputs, **params)
    tmp = processor.decode(inputs.get('input_ids')[0], skip_special_tokens=True)
    txt = processor.decode(output[0], skip_special_tokens=True).replace(tmp, '')
    return txt
