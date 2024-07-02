# Your Playground for Gemini

This repository is your one-stop shop for exploring the power of Large Language Models (LLMs) on Google Cloud. We've curated a collection of practical examples that showcase how to leverage Google's cutting-edge AI services for a variety of tasks.

### Dive into the World of LLMs:

* **Gemini Examples:** Get hands-on with Google's Gemini family of LLMs, known for their advanced capabilities in text, code, and multimodal understanding.
* **Text Generation:** Learn how to generate creative text formats like poems, code, scripts, musical pieces, emails, and more.
* **Multimodal Prompts:** Explore the exciting world of multimodal prompts, combining images and text to unlock new creative possibilities.
* **Langchain Integration:** Discover how to seamlessly integrate Google's Generative AI API with LangChain, a powerful framework for building complex AI applications.
* **Embeddings and Search:** Master the art of creating embeddings, vector representations of text, for tasks like similarity search and information retrieval.
* **Firestore Integration:** Learn how to store and retrieve embeddings efficiently using Google Cloud Firestore, a NoSQL database.
* **LLM Reflection:** Delve into the fascinating concept of LLM reflection, where LLMs can analyze and improve their own outputs.

## Getting Started

1. **Install dependencies:**
```bash
poetry install
```

2. **Set up environment variables:**

Create a .env file in the root directory of the project.
Add the following environment variables to the .env file:
```
GOOGLE_APPLICATION_CREDENTIALS=<path/to/your/credentials.json>
```

Use a `GOOGLE_API_KEY` as a environment variable. This can be generated in AI STUDIO

3. **Run the examples:**

Navigate to the directory containing the example you want to run.
Run the Python script using 
```
poetry run python <script_name.py>.
```
  
### Explore, Learn, and Build:

This repository is designed to be a learning resource and a springboard for your own LLM projects. Feel free to experiment, modify the examples, and build your own innovative applications using Google Cloud's powerful AI tools.