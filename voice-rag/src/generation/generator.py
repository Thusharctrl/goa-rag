from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings


class AnswerGenerator:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for answer generation")
        self._client = Groq(api_key=settings.groq_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, query: str, contexts: list[str], language: str) -> str:
        if not contexts:
            return "I do not have enough information to answer that question."

        context_block = "\n\n".join(f"[{idx + 1}] {text}" for idx, text in enumerate(contexts))
        language_name = "Hindi" if language == "hi" else "English"

        system_prompt = (
            "You are a retrieval-grounded assistant. Answer ONLY using the provided context. "
            "If the context is insufficient, say you do not have enough information. "
            f"Respond in {language_name}."
        )
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Provide a concise, grounded answer with no unsupported claims."
        )

        response = self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
