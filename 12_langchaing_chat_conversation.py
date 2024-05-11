from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rich.console import Console
import os
from dotenv import load_dotenv

load_dotenv()

console = Console()

if __name__ == "__main__":
    api_key = str(os.getenv("GOOGLE_API_KEY"))

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=api_key)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What's in this image?",
            },  # You can optionally provide text parts
            {
                "type": "image_url",
                "image_url": "https://picsum.photos/seed/picsum/200/300"
            },
        ]
    )

    result = llm.invoke([message])
    console.print(result.content)
