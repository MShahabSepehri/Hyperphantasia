import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

def get_client(max_retries=2, timeout=30):

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=max_retries, timeout=timeout)

    return client

def get_response(client, deployment_name, init_prompt, prompt, temperature, max_retry=3, print_error=True, max_tokens=4096):
    counter = max_retry
    response = None
    while counter > 0:
        try:
            if 'o3' in deployment_name or 'o4' in deployment_name:
                response = client.chat.completions.create(model=deployment_name,
                                                        messages=[
                                                            {"role": "user", "content": prompt},
                                                            ],
                                                        reasoning_effort="medium",
                                                        )
            else:
                response = client.chat.completions.create(model=deployment_name,
                                                        messages=[
                                                            {"role": "system", "content": init_prompt},
                                                            {"role": "user", "content": prompt},
                                                            ],
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,
                                                        )
            response = response.choices[0].message.content
            break
        except Exception as e:
            if print_error:
                print(e)
            counter -= 1
    return response

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
 
def creat_message(prompt):
    content = []
    for c in prompt:
        if c[0] == "text":
            content.append({"type": "text", "text": c[1]})
        elif c[0] == "image":
            image = encode_image(c[1])
            content.append({
                    "type": "image_url","image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                        }})
    return content

def ask_question(client, prompt, init_prompt, deployment_name, temperature, max_retry=3, print_error=False, max_tokens=4096):
    content = creat_message(prompt)
    response = get_response(client=client,
                            deployment_name=deployment_name, 
                            init_prompt=init_prompt, 
                            prompt=content, 
                            temperature=temperature,
                            max_retry=max_retry,
                            print_error=print_error,
                            max_tokens=max_tokens)

    return response