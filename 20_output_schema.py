import os
import google.generativeai as genai
from rich.console import Console
from dotenv import load_dotenv
import typing_extensions as typing

load_dotenv()

console = Console()


class Recipe(typing.TypedDict):
    name: str
    description: str
    calories: str
    ingredients: typing.List[str]
    instructions: typing.List[str]


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    model = genai.GenerativeModel(
        'gemini-1.5-flash-001',
        generation_config={"response_mime_type": "application/json",
                           "response_schema": list[Recipe]})

    prompt = "List 3 popular mexican recipes with their names, descriptions, calories, ingredients and instructions"

    response = model.generate_content(prompt)
    console.print(response.text)
