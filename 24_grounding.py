from rich.console import Console

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    grounding,
)

console = Console()


def grounding_llm():
    model = GenerativeModel("gemini-1.5-flash-001")
    tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
    prompt = "Quien gano el premio nobel de fisica en el 2024"
    response = model.generate_content(
        prompt,
        tools=[tool],
        generation_config=GenerationConfig(
            temperature=0.0,
        ),
    )
    console.print(response.text)
    console.print(response)


if __name__ == "__main__":
    grounding_llm()
