from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sample_size: int = 100
    chroma_path: str = "data/chroma"
    collection_name: str = "msmarco_xi"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    sarvam_api_key: str = ""
    retrieval_top_k: int = 5
    min_retrieval_score: float = 0.55
    min_grounding_overlap: float = 0.08
    analytics_log_path: str = "data/analytics/latency.jsonl"
    hf_parquet_url: str = (
        "https://huggingface.co/datasets/"
        "ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
    )

    @property
    def chroma_dir(self) -> Path:
        return Path(self.chroma_path)

    @property
    def analytics_log_file(self) -> Path:
        return Path(self.analytics_log_path)


settings = Settings()
