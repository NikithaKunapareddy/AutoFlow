import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env if present
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)
try:
    # List available models
    models = list(genai.list_models())
    print("Available models:")
    for m in models:
        print(m.name)
    # Try to use the first model that supports content generation
    model_name = None
    for m in models:
        if hasattr(m, 'supported_generation_methods') and 'generateContent' in getattr(m, 'supported_generation_methods', []):
            model_name = m.name
            break
    if not model_name and models:
        model_name = models[0].name  # fallback to first model
    if not model_name:
        raise Exception("No available Gemini model found for content generation.")
    print(f"\nUsing model: {model_name}\n")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Hello, are you working?")
    print("API key is working! Response:")
    print(response.text)
except Exception as e:
    print("API key is NOT working or there is another issue:")
    print(e)