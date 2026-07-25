from src.llm.ollama_llm import OllamaLLM

llm = OllamaLLM()

answer = llm.generate(
    question="What is Deep Learning?",
    context="""
Deep learning is a subset of machine learning
that uses many-layer neural networks to learn
representations from data.
"""
)

print(answer)