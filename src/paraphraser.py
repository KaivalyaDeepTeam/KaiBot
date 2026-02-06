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
    temperature: float = 0.85  # Higher for more human-like variation
    top_p: float = 0.92
    top_k: int = 50
    repeat_penalty: float = 1.15  # Penalize repetition more
    max_tokens: int = 1024  # Allow longer outputs


# System prompts optimized to bypass AI detection (Copyleaks, ZeroGPT, etc.)
SYSTEM_PROMPTS = {
    "default": """Rewrite this text like a human blogger would write it. Make it sound natural and conversational.

You MUST:
- Change sentence structures completely - don't keep the same pattern
- Use contractions everywhere: don't, won't, it's, that's, you've, they're, can't, isn't
- Start some sentences with And, But, So, Or, Well, Now, Look, See, Thing is
- Mix very short sentences with longer ones. Like this. Then write a longer one that flows naturally.
- Use casual phrases: pretty much, kind of, a lot of, tons of, bunch of, really, actually, basically, honestly
- Replace formal words: use "get" not "obtain", "help" not "assist", "show" not "demonstrate", "use" not "utilize", "about" not "approximately", "big" not "substantial"
- Add filler words occasionally: just, really, actually, probably, maybe, quite, pretty
- Keep all facts, numbers, and names exactly the same
- Keep similar length to original

DO NOT:
- Add citations, references, or sources
- Add new information not in original
- Use overly formal or academic language
- Write "Here is" or any meta-commentary

Write the rewritten text directly:""",

    "academic": """Rewrite this academic text naturally while keeping scholarly tone.

You MUST:
- Restructure sentences - don't keep same patterns
- Vary between active and passive voice naturally
- Use transitions: However, Moreover, Additionally, Furthermore, In addition, That said
- Mix complex sentences with simpler ones for readability
- Keep all facts, data, citations, and technical terms exactly
- Maintain formal register but sound like a human researcher wrote it

DO NOT add fake citations or references. Keep similar length.

Write the rewritten text:""",

    "casual": """Rewrite this like you're chatting with a friend. Super casual and natural.

You MUST:
- Use tons of contractions: don't, it's, won't, that's, I'd, we're, they've, can't
- Start sentences with And, But, So, Or, Anyway, Look, Thing is
- Use casual phrases: pretty much, kind of, you know, I mean, honestly, basically
- Keep sentences punchy. Short works. Then mix in longer ones when needed.
- Use simple words everyone knows
- Keep all the facts and details the same

Write it like a real person talking:""",

    "technical": """Rewrite this technical content clearly and naturally.

You MUST:
- Keep ALL technical terms, code, commands, and specifications exactly unchanged
- Use active voice: "Run the command" not "The command should be run"
- Break complex explanations into digestible pieces
- Use simple connecting words: then, next, after that, once done
- Vary sentence structure naturally
- Keep all accuracy and details

Write the rewritten technical content:"""
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

            # Calculate max tokens based on input length
            input_words = len(text.split())
            # Estimate ~1.5 tokens per word, add 30% buffer for natural variation
            dynamic_max_tokens = min(int(input_words * 1.5 * 1.3), self.config.max_tokens)
            dynamic_max_tokens = max(dynamic_max_tokens, 100)  # Minimum 100 tokens

            response = self.model(
                prompt,
                max_tokens=dynamic_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=["</s>", "[INST]", "\n\nReferences:", "\n\nSources:", "\n\n["],
                echo=False
            )

            result = response["choices"][0]["text"].strip()

            # Clean up any remaining artifacts
            result = self._clean_output(result, text)

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

    def _clean_output(self, text: str, original: str) -> str:
        """Clean up model output and validate against original."""
        import re

        result = text.strip()

        # Remove common prefixes that models add
        prefixes_to_remove = [
            "here's the paraphrased text:",
            "here is the paraphrased text:",
            "paraphrased version:",
            "rewritten text:",
            "here's the rewritten version:",
            "here is the rewritten text:",
            "the rewritten text:",
            "rewritten version:",
            "output:",
        ]

        result_lower = result.lower()
        for prefix in prefixes_to_remove:
            if result_lower.startswith(prefix):
                result = result[len(prefix):].strip()
                result_lower = result.lower()

        # Remove surrounding quotes if present
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]

        # Remove fake citations/references that LLM might add
        # Pattern: (Author et al., YYYY) or [1], [2], etc.
        result = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.,?\s*\d{4}\)', '', result)
        result = re.sub(r'\[[0-9]+\]', '', result)
        result = re.sub(r'\([\w\s&]+,\s*\d{4}\)', '', result)

        # Remove "References:" sections and anything after
        ref_patterns = [
            r'\n\s*References:.*',
            r'\n\s*Sources:.*',
            r'\n\s*Citations:.*',
            r'\n\s*Bibliography:.*',
        ]
        for pattern in ref_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.DOTALL)

        # If output is way too long compared to input, truncate
        orig_words = len(original.split())
        result_words = len(result.split())

        if result_words > orig_words * 1.5:
            # Output is too long, truncate to similar length
            words = result.split()
            result = ' '.join(words[:int(orig_words * 1.2)])

        # If result is too short or empty, return original
        if not result.strip() or result_words < orig_words * 0.3:
            return original

        return result.strip()

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
