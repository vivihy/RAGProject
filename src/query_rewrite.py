from src.llm_config import BaseLLM

def hyde_rewrite(query: str, llm: BaseLLM, temperature: float = 0.0) -> str:
    prompt = f"Write a detailed paragraph that answers the following question. Only output the paragraph.\nQuestion: {query}\nParagraph:"
    return llm.generate(prompt, temperature=temperature)