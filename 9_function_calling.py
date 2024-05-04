import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    def calculate_salary_bonus(age: float,
                               salary: float,
                               years_of_experience: float):
        return salary + age + years_of_experience

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest',
                                  tools=[calculate_salary_bonus])

    console.print(model)

    console.print(model._tools.to_proto())

    chat = model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(
        'This employee Juan is 35 years old and his salary is 4000 USD and he has been working in this company for 4 years. What is the value of the bonus for this employee?')
    console.print(response.text)

    for content in chat.history:
        part = content.parts[0]
        console.print(content.role, "->", type(part).to_dict(part))
