from src.llm_config import BaseLLM

def generate_answer(query: str, retrieved_docs: list, config: dict, llm: BaseLLM) -> str:
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    if config["answer_style"] == "concise":
        prompt = f"Answer the question concisely based on the context.\nContext: {context}\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Answer the question and cite the relevant parts from the context.\nContext: {context}\nQuestion: {query}\nAnswer:"
    return llm.generate(prompt, temperature=config["temperature"])