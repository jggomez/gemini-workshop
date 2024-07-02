import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    chat = model.start_chat(history=[])

    prompt_user = "In one sentence, explain how a computer works to a young child."
    console.print(f"User: {prompt_user}")

    response = chat.send_message(prompt_user, stream=True)

    console.print("LLM: ")
    for chunk in response:
        console.print(chunk.text)

    prompt_user = "Okay, how about a more detailed explanation to a high schooler?"
    console.print(f"User: {prompt_user}")

    response = chat.send_message(prompt_user, stream=True)

    console.print("LLM: ")
    for chunk in response:
        console.print(chunk.text)

    for message in chat.history:
        console.print(
            f'**{message.role}**: {message.parts[0].text}')
