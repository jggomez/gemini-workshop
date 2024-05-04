import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    calculate_salary_bonus = {'function_declarations': [
        {'name': 'calculate_salary_bonus',
         'description': 'Returns the salary bonus for a given employee',
         'parameters': {'type_': 'OBJECT',
                        'properties': {
                            'age': {'type_': 'NUMBER'},
                            'salary': {'type_': 'NUMBER'},
                            'years_of_experience': {'type_': 'NUMBER'}
                        },
                        'required': ['age', 'salary', 'years_of_experience']}}]}

    model = genai.GenerativeModel(
        'models/gemini-1.5-pro-latest', tools=calculate_salary_bonus)
    chat = model.start_chat()

    response = chat.send_message(
        "This employee Juan is 35 years old and his salary is 4000 USD and he has been working in this company for 4 years. What is the value of the bonus for this employee?",
    )

    console.print(response.candidates)

    for content in chat.history:
        part = content.parts[0]
        console.print(content.role, "->", type(part).to_dict(part))

    def multiply(a: float, b: float):
        """returns a * b."""
        return a*b

    genai.GenerativeModel(model_name='gemini-1.0-pro',
                          tools=[multiply])

    calculator = {'function_declarations': [
        {'name': 'multiply',
         'description': 'Returns the product of two numbers.',
         'parameters': {'type_': 'OBJECT',
                        'properties': {
                            'a': {'type_': 'NUMBER'},
                            'b': {'type_': 'NUMBER'}},
                        'required': ['a', 'b']}}]}
    
    calculator_sum = {'function_declarations': [
        {'name': 'sum',
         'description': 'Returns the sum of two numbers.',
         'parameters': {'type_': 'OBJECT',
                        'properties': {
                            'a': {'type_': 'NUMBER'},
                            'b': {'type_': 'NUMBER'}},
                        'required': ['a', 'b']}}]}

    model = genai.GenerativeModel('gemini-pro', tools=[calculator, calculator_sum])
    chat = model.start_chat()

    response = chat.send_message(
        "What's 234551 + 325552 ?",
    )

    console.print(response.candidates)

    for content in chat.history:
        part = content.parts[0]
        console.print(content.role, "->", type(part).to_dict(part))

    fc = response.candidates[0].content.parts[0].function_call
    console.print(fc)

    result = fc.args['a'] * fc.args['b']

    response = chat.send_message(
        glm.Content(
            parts=[glm.Part(
                function_response=glm.FunctionResponse(
                    name='multiply',
                    response={'result': result}))]))
    
    console.print(response)
