import os
import base64
from dotenv import load_dotenv
from anthropic import Anthropic

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
        return base64_string


def get_client():
    load_dotenv(override=True)
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client

def create_content(prompt):
    content = []
    for c in prompt:
        if c[0] == "image":
            content.append({"type": "image", 
                            "source": {"type": "base64", 
                                       "media_type": "image/jpeg", 
                                       "data": get_base64_encoded_image(c[1])}})
        else:
            content.append({"type": "text", "text": c[1]})
    message_list = [{"role": 'user', "content": content}]
    return message_list

def ask_question(client, prompt, init_prompt, temperature, deployment_name, max_tokens=2048):
    message_list = create_content(prompt)
    response = client.messages.create(
        model=deployment_name,
        max_tokens=max_tokens,
        messages=message_list,
        temperature=temperature,
        system=init_prompt,
    )
    return response.content[0].text