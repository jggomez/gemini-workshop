import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()


def init_genai(google_api_key: str):
    genai.configure(api_key=google_api_key)


def list_models():
    models = genai.list_models()
    for model in models:
        console.print(model)


def create_ai_model(model_name: str) -> genai.GenerativeModel:
    return genai.GenerativeModel(model_name)


def generate_song(model: genai.GenerativeModel):
    prompt = "Create a beautiful song for my mother",

    generation_config = genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        # stop_sequences=['x'],
        #max_output_tokens=4000,
        temperature=1.0)

    response = model.generate_content(
        prompt, generation_config=generation_config)
    tokens = model.count_tokens(prompt)
    console.print(tokens)
    # Gemini can generate multiple possible responses for a single prompt
    console.print(response.candidates)
    # If the API failed to return a result
    console.print(response.prompt_feedback)
    # In simple cases
    console.print(response.text)


def generate_song_stream(model: genai.GenerativeModel):
    prompt = "Create a beautiful song for my mother",
    chunks = model.generate_content(prompt, stream=True)
    for chunk in chunks:
        console.print(chunk.text)


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    console.print(google_api_key)
    init_genai(google_api_key)
    list_models()
    model = create_ai_model("gemini-1.5-pro-latest")
    generate_song(model=model)
    # generate_song_stream(model=model)
