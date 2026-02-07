"""
Model Registry - Curated list of AI models for LocalWrite.

Each model is carefully selected and tested for writing enhancement tasks.
Models are downloaded from Hugging Face and stored locally.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an available AI model."""
    id: str
    name: str
    recommended: bool
    best_for: str
    description: str
    size_gb: float
    size_display: str
    speed: str
    min_ram: int
    filename: str
    hf_repo: str
    download_url: str
    chat_template: str = "chatml"  # chatml, llama, mistral, gemma


# Curated models optimized for writing tasks
AVAILABLE_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="qwen-2.5-7b",
        name="Qwen 2.5 7B",
        recommended=True,
        best_for="General writing, essays, articles, professional documents",
        description="Best overall for writing tasks. Excellent instruction-following and natural language generation.",
        size_gb=4.68,
        size_display="4.7 GB",
        speed="Fast on Apple Silicon",
        min_ram=8,
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        hf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        download_url="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        chat_template="chatml",
    ),
    ModelInfo(
        id="gemma-2-9b",
        name="Gemma 2 9B",
        recommended=False,
        best_for="Creative writing, storytelling, human-sounding prose",
        description="Produces more creative, natural-sounding text. Great for fiction and expressive writing.",
        size_gb=5.76,
        size_display="5.8 GB",
        speed="Fast",
        min_ram=10,
        filename="gemma-2-9b-it-Q4_K_M.gguf",
        hf_repo="bartowski/gemma-2-9b-it-GGUF",
        download_url="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
        chat_template="gemma",
    ),
    ModelInfo(
        id="llama-3.2-3b",
        name="Llama 3.2 3B",
        recommended=False,
        best_for="Quick edits, short texts, low memory devices",
        description="Lightweight and fast. Perfect for quick improvements on machines with limited RAM.",
        size_gb=2.02,
        size_display="2.0 GB",
        speed="Very fast",
        min_ram=4,
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        download_url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        chat_template="llama",
    ),
    ModelInfo(
        id="llama-3.1-8b",
        name="Llama 3.1 8B",
        recommended=False,
        best_for="Professional writing, clean controllable prose",
        description="Meta's powerful 8B model. Excellent for professional and business content.",
        size_gb=4.92,
        size_display="4.9 GB",
        speed="Fast",
        min_ram=8,
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        hf_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        download_url="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        chat_template="llama",
    ),
    ModelInfo(
        id="mistral-7b",
        name="Mistral 7B v0.3",
        recommended=False,
        best_for="Balanced writing, real-time content generation",
        description="Great balance of speed and quality. Strong reasoning capabilities.",
        size_gb=4.37,
        size_display="4.4 GB",
        speed="Fast",
        min_ram=8,
        filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        hf_repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        download_url="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        chat_template="mistral",
    ),
]

# Model recommendations by writing task type
TASK_RECOMMENDATIONS: Dict[str, str] = {
    "general": "qwen-2.5-7b",
    "creative": "gemma-2-9b",
    "professional": "llama-3.1-8b",
    "quick": "llama-3.2-3b",
    "balanced": "mistral-7b",
}


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None


def get_recommended_model() -> ModelInfo:
    """Get the recommended model."""
    for model in AVAILABLE_MODELS:
        if model.recommended:
            return model
    return AVAILABLE_MODELS[0]


def get_model_for_task(task: str) -> Optional[ModelInfo]:
    """Get the best model for a specific task type."""
    model_id = TASK_RECOMMENDATIONS.get(task)
    if model_id:
        return get_model_by_id(model_id)
    return get_recommended_model()


def get_all_models() -> List[ModelInfo]:
    """Get all available models."""
    return AVAILABLE_MODELS.copy()


def get_models_by_min_ram(max_ram: int) -> List[ModelInfo]:
    """Get models that can run with specified RAM."""
    return [m for m in AVAILABLE_MODELS if m.min_ram <= max_ram]
