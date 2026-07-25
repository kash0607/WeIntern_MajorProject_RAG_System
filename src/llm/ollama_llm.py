import ollama


class OllamaLLM:

    def __init__(self):

        self.model = "llama3.2:3b"

    def generate(self, question, context):

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:
"I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]