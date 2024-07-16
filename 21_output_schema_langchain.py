import os
from rich.console import Console
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser
from langchain_google_genai import HarmCategory
from langchain_google_genai import HarmBlockThreshold

load_dotenv()

console = Console()


class Translations(BaseModel):
    translations: list[str] = Field(description="list of translations")

    @field_validator('translations')
    def validate_empty(cls, field):
        if len(field) == 1:
            raise ValueError(
                "The field translations must have at least two results")
        return field


if __name__ == "__main__":
    api_key = str(os.getenv("GOOGLE_API_KEY"))

    source_lang = "EN"
    target_lang = "IT"
    source_text = "when you’re out of bullets and staring down the barrel of a Kalashnikov,"

    # Basic example with GoogleGenerativeAI
    model = GoogleGenerativeAI(
        model='models/gemini-1.5-flash-001',
        google_api_key=api_key,
        temperature=0.1,
        system_message=f"""
            You are an expert linguist, specializing in translation editing from \
            {source_lang} to {target_lang}.
            You will be provided at least one translation.
            Output as a comma separated list
        """,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        })

    translations_prompt = f"""
        This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text.
        Provide two translations. Do not provide any explanations or text apart from the translation.
        text: {source_text}
        """

    parser = PydanticOutputParser(
        pydantic_object=Translations, return_exceptions=False)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    retry_parser = RetryOutputParser.from_llm(parser=parser, llm=model, max_retries=3)

    main_chain = RunnableParallel(
        completion=chain, prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    console.print(chain.invoke({"query": translations_prompt}))
