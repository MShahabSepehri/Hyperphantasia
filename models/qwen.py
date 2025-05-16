import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct", device='auto', cache_dir=None):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def creat_message(prompt):
    content = []
    for c in prompt:
        content.append({"type": c[0], c[0]: c[1]})
    messages = [{"role": "user", "content": content}]
    image_inputs, video_inputs = process_vision_info(messages)
    return messages, image_inputs, video_inputs

@torch.no_grad()
def ask_question(model, processor, prompt, temperature, max_new_tokens=4096):
    messages, image_inputs, video_inputs = creat_message(prompt)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        )
    inputs = inputs.to("cuda")
    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": (temperature > 0),
    }
    if temperature <= 0:
        params["temperature"] = None

    generated_ids = model.generate(**inputs, **params)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return response[-1]