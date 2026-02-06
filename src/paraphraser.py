"""
Paraphraser - LLM-based text paraphrasing using Mistral 7B via llama-cpp-python.
Optimized for generating human-like text that bypasses AI detection.
"""

from typing import Optional, Callable, List
from dataclasses import dataclass
import os

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


@dataclass
class ParaphraserConfig:
    """Configuration for the paraphraser."""
    model_path: str = ""
    n_ctx: int = 4096  # Context window
    n_threads: int = 4  # CPU threads
    n_gpu_layers: int = 0  # GPU layers (Metal on Mac)
    temperature: float = 0.9  # Higher for more human-like variation
    top_p: float = 0.92
    top_k: int = 50
    repeat_penalty: float = 1.15  # Slightly higher to avoid repetition
    max_tokens: int = 1024


# System prompts optimized to bypass AI detection (Copyleaks, ZeroGPT, etc.)
SYSTEM_PROMPTS = {
    "default": """Rewrite this text as a real human would write it. You must:

1. VARY sentence length dramatically - use some very short sentences. Then use longer ones. Mix it up unpredictably.
2. Use contractions naturally: don't, won't, it's, that's, wouldn't, couldn't
3. Start some sentences with "And", "But", "So", "Well", "Now" - humans do this
4. Use casual transitions: "thing is", "basically", "honestly", "actually", "anyway"
5. Add slight redundancy occasionally - humans repeat ideas differently
6. Use simpler words: "use" not "utilize", "help" not "assist", "show" not "demonstrate"
7. Break grammar rules slightly: fragment sentences sometimes. Like this.
8. Vary paragraph length - some short, some longer
9. Keep all facts, names, numbers exactly the same

DO NOT:
- Use overly formal or perfect language
- Make every sentence the same length
- Use fancy vocabulary unnecessarily
- Add phrases like "Here is" or "The following"

Output ONLY the rewritten text, nothing else.""",

    "academic": """Rewrite this academic text naturally while keeping scholarly tone. You must:

1. Preserve all citations, references, and technical terms exactly
2. Vary sentence structures - mix complex and simple sentences
3. Use some passive voice but also active voice naturally
4. Occasionally start with transitions: "However,", "Moreover,", "Additionally,"
5. Keep discipline-specific terminology unchanged
6. Maintain formal register but avoid robotic perfection
7. Slightly vary how you connect ideas between sentences
8. Keep all factual claims and data exactly as written

Output ONLY the rewritten text, nothing else.""",

    "casual": """Rewrite this like you're explaining it to a friend. You must:

1. Use lots of contractions: don't, it's, won't, that's, I'd, we're
2. Start sentences with And, But, So, Or sometimes
3. Use casual phrases: "pretty much", "kind of", "you know", "the thing is"
4. Break up long sentences into shorter punchy ones
5. Use simple everyday words
6. Add occasional filler: "basically", "actually", "honestly"
7. Keep paragraphs short and readable
8. Maintain all the original facts and information

Output ONLY the rewritten text, nothing else.""",

    "technical": """Rewrite this technical content clearly while keeping accuracy. You must:

1. Keep ALL code, commands, file paths, and technical terms exactly unchanged
2. Use active voice: "Run the command" not "The command should be run"
3. Vary explanation styles within the text
4. Use simpler connecting words between technical points
5. Break complex explanations into shorter sentences
6. Keep numbered lists and structure if present
7. Preserve all technical accuracy completely

Output ONLY the rewritten text, nothing else."""
}


class Paraphraser:
    """
    Paraphrases text using a local LLM to bypass AI detection.
    """

    def __init__(self, config: Optional[ParaphraserConfig] = None):
        """Initialize the paraphraser with optional config."""
        self.config = config or ParaphraserConfig()
        self.model: Optional[Llama] = None
        self.is_loaded = False
        self.current_style = "default"

    def load_model(
        self,
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Load the LLM model.

        Args:
            model_path: Path to the GGUF model file
            progress_callback: Called with status messages

        Returns:
            True if loaded successfully
        """
        if Llama is None:
            if progress_callback:
                progress_callback("Error: llama-cpp-python not installed")
            return False

        path = model_path or self.config.model_path

        if not path or not os.path.exists(path):
            if progress_callback:
                progress_callback(f"Error: Model file not found: {path}")
            return False

        try:
            if progress_callback:
                progress_callback("Loading model... This may take a moment.")

            # Detect if Metal (GPU) is available on macOS
            n_gpu = self.config.n_gpu_layers
            if n_gpu == 0:
                # Try to use Metal on macOS
                import platform
                if platform.system() == "Darwin":
                    n_gpu = -1  # Use all layers on GPU

            self.model = Llama(
                model_path=path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=n_gpu,
                verbose=False
            )

            self.config.model_path = path
            self.is_loaded = True

            if progress_callback:
                progress_callback("Model loaded successfully!")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error loading model: {e}")
            return False

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False

    def set_style(self, style: str):
        """Set the paraphrasing style."""
        if style in SYSTEM_PROMPTS:
            self.current_style = style

    def paraphrase(
        self,
        text: str,
        style: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Paraphrase a single piece of text.

        Args:
            text: Text to paraphrase
            style: Paraphrasing style (default, academic, casual, technical)
            progress_callback: Called with status updates

        Returns:
            Paraphrased text
        """
        if not self.is_loaded or not self.model:
            return text  # Return original if model not loaded

        if not text or not text.strip():
            return text

        style = style or self.current_style
        system_prompt = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["default"])

        # Construct the prompt (without <s> as llama.cpp adds it automatically)
        prompt = f"""[INST] {system_prompt}

Text to paraphrase:
{text.strip()}

[/INST]"""

        try:
            if progress_callback:
                progress_callback("Generating paraphrase...")

            response = self.model(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=["</s>", "[INST]"],
                echo=False
            )

            result = response["choices"][0]["text"].strip()

            # Clean up any remaining artifacts
            result = self._clean_output(result)

            return result if result else text

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error during paraphrasing: {e}")
            return text

    def paraphrase_batch(
        self,
        texts: List[str],
        style: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[str]:
        """
        Paraphrase multiple texts with progress tracking.

        Args:
            texts: List of texts to paraphrase
            style: Paraphrasing style
            progress_callback: Called with (current, total, status)

        Returns:
            List of paraphrased texts
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing {i + 1}/{total}...")

            result = self.paraphrase(text, style)
            results.append(result)

        return results

    def _clean_output(self, text: str) -> str:
        """Clean up model output artifacts."""
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Here's the paraphrased text:",
            "Here is the paraphrased text:",
            "Paraphrased version:",
            "Rewritten text:",
            "Here's the rewritten version:",
        ]

        result = text.strip()

        for prefix in prefixes_to_remove:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()

        # Remove surrounding quotes if present
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]

        return result

    def get_available_styles(self) -> List[str]:
        """Return list of available paraphrasing styles."""
        return list(SYSTEM_PROMPTS.keys())

    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


class ParaphraserManager:
    """
    Manages the paraphrasing workflow for a complete document.
    """

    def __init__(self, paraphraser: Paraphraser):
        self.paraphraser = paraphraser
        self._cancelled = False

    def cancel(self):
        """Cancel the current processing."""
        self._cancelled = True

    def reset(self):
        """Reset cancellation flag."""
        self._cancelled = False

    def process_text_blocks(
        self,
        text_blocks,  # List[TextBlock] from pdf_processor
        style: str = "default",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        """
        Process a list of text blocks, paraphrasing each one.
        Updates the paraphrased_text attribute of each block in-place.

        Returns:
            Number of successfully processed blocks
        """
        self._cancelled = False
        total = len(text_blocks)
        processed = 0

        for i, block in enumerate(text_blocks):
            if self._cancelled:
                if progress_callback:
                    progress_callback(i, total, "Cancelled")
                break

            if progress_callback:
                preview = block.text[:50] + "..." if len(block.text) > 50 else block.text
                progress_callback(i + 1, total, f"Processing: {preview}")

            paraphrased = self.paraphraser.paraphrase(block.text, style)
            block.paraphrased_text = paraphrased
            processed += 1

        return processed
