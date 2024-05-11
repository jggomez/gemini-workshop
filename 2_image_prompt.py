import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv
import PIL.Image

load_dotenv()

console = Console()

if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    img = PIL.Image.open('data/images/animales_mexico.jpeg')

    model = genai.GenerativeModel('gemini-pro-vision')
    num_tokens = model.count_tokens(img)
    console.print(num_tokens)

    prompt = [
        "Write a nature blog with images and references with the following image:", img]
    num_tokens = model.count_tokens(prompt)
    console.print(num_tokens)
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        console.print(chunk.text)
