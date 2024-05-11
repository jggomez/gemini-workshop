from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from rich.console import Console
import os
import getpass
from dotenv import load_dotenv

load_dotenv()

console = Console()

if __name__ == "__main__":
    api_key = str(os.getenv("GOOGLE_API_KEY"))

    # Basic example with GoogleGenerativeAI
    llm = GoogleGenerativeAI(
        model='models/gemini-1.5-pro-latest',
        google_api_key=api_key)

    result = llm.invoke("Create a beautiful song for my mother")
    console.print(result)

    # Stream
    for chunk in llm.stream("Create a beautiful song for my mother"):
        console.print(chunk)

    # With PromptTemplate
    template = """Question: {question}"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    question = "How much is 2+2?"
    console.print(chain.invoke({"question": question}))
