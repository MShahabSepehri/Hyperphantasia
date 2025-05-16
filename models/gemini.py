import os
from dotenv import load_dotenv
from IPython.display import Image
import google.generativeai as genai

def configure_client():
    # Load environment variables from .env file
    load_dotenv(override=True)

    # Access the API key
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


def load_model(init_prompt, temperature, deployment_name):
    configure_client() 
    model = genai.GenerativeModel(deployment_name, 
                                  generation_config=genai.GenerationConfig(temperature=temperature),
                                  system_instruction=init_prompt,
                                  )
    return model

def create_content(prompt):
    content = []
    for mode, c in prompt:
        if mode == 'text':
            content.append(c)
        else:
            content.append(Image(c))
    return content

# Based on https://github.com/google-gemini/cookbook/blob/main/quickstarts/Prompting.ipynb
def ask_question(model, prompt):
    # img = Image(image_path)
    response = model.generate_content(create_content(prompt))
    return response.text