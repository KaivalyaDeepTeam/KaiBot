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
    n_ctx: int = 8192  # Larger context window for chat
    n_threads: int = 4  # CPU threads
    n_gpu_layers: int = 0  # GPU layers (Metal on Mac)
    temperature: float = 0.7  # Balanced creativity
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1  # Slight repetition penalty
    max_tokens: int = 4096  # Allow much longer outputs for generation


# System prompts - focused modes that transform vocabulary and style
SYSTEM_PROMPTS = {
    "professional": """Transform this text into professional, business-appropriate language.

VOCABULARY CHANGES (apply these transformations):
- "Hi" → "Hello" or "Greetings"
- "Hey" → "Hello"
- "ain't" → "is not" / "are not"
- "gonna" → "going to"
- "wanna" → "want to"
- "gotta" → "have to" / "need to"
- "kinda" → "somewhat" / "rather"
- "sorta" → "somewhat"
- "yeah" → "yes"
- "nope" → "no"
- "cool" → "excellent" / "acceptable"
- "awesome" → "impressive" / "excellent"
- "stuff" → "materials" / "items" / "matters"
- "things" → "items" / "aspects" / "elements"
- "get" → "obtain" / "receive" / "acquire"
- "a lot" → "significantly" / "considerably"
- "pretty much" → "essentially" / "largely"
- "kind of" → "somewhat" / "to some extent"
- Remove excessive exclamation marks

PRESERVE EXACTLY: All names, facts, numbers, dates, core information.

STYLE:
- Professional vocabulary throughout
- Clear, precise language
- Proper sentence structure
- Active voice preferred
- No slang, contractions, or casual expressions

Output the professional rewrite only:""",

    "conversational": """Transform this text into casual, friendly language like talking to a friend.

VOCABULARY CHANGES (apply these transformations):
- "Hello" → "Hi" or "Hey"
- "Greetings" → "Hi there"
- "is not" → "isn't"
- "are not" → "aren't"
- "will not" → "won't"
- "cannot" → "can't"
- "going to" → "gonna"
- "want to" → "wanna"
- "have to" → "gotta"
- "excellent" → "awesome" / "cool"
- "impressive" → "amazing"
- "obtain" → "get"
- "receive" → "get"
- "utilize" → "use"
- "therefore" → "so"
- "however" → "but"
- "additionally" → "also" / "plus"
- "regarding" → "about"
- "approximately" → "about" / "around"

PRESERVE EXACTLY: All names, facts, numbers, dates - core story stays the same.

STYLE:
- Use contractions throughout
- Start some sentences with And, But, So, Well
- Add casual phrases: "pretty much", "kind of", "basically", "honestly", "you know"
- Short punchy sentences
- Friendly, relaxed tone

Output the casual rewrite only:""",

    "scholarly": """Transform this text into formal academic writing style.

VOCABULARY CHANGES (apply these transformations):
- "show" → "demonstrate" / "illustrate"
- "use" → "utilize" / "employ"
- "get" → "obtain" / "acquire"
- "look at" → "examine" / "analyze"
- "find out" → "determine" / "ascertain"
- "think" → "posit" / "hypothesize" / "consider"
- "big" → "substantial" / "significant"
- "a lot" → "numerous" / "substantial"
- "good" → "favorable" / "positive" / "beneficial"
- "bad" → "adverse" / "negative" / "detrimental"
- Remove all contractions
- Remove casual expressions

PRESERVE EXACTLY: All names, data, citations, technical terms, dates, numbers.

STYLE:
- Formal academic tone
- Third person perspective when possible
- Proper scholarly structure
- Mix of active and passive voice
- No informal language or contractions

Output the scholarly rewrite only:""",

    "creative": """Transform this text into vivid, engaging creative writing.

VOCABULARY CHANGES (apply these transformations):
- Replace plain verbs with vivid ones: "walked" → "strolled/strode/ambled"
- Replace basic adjectives with evocative ones: "big" → "massive/towering/enormous"
- Add sensory details and imagery
- Use metaphors and comparisons where natural
- Replace "said" with varied dialogue tags when appropriate
- Transform passive descriptions into active scenes

PRESERVE EXACTLY: All names, places, dates, numbers, events, facts.

STYLE:
- Vivid, engaging word choices
- Varied sentence structures and lengths
- Create rhythm and flow
- Show don't tell
- Make it memorable and immersive

Output the creative rewrite only:""",

    "concise": """Make this text shorter and more direct while keeping all essential meaning.

TRANSFORMATIONS:
- Remove filler words: "really", "very", "quite", "just", "actually", "basically"
- Remove redundant phrases: "in order to" → "to", "at this point in time" → "now"
- Combine related sentences
- Remove unnecessary modifiers
- Cut wordy expressions: "due to the fact that" → "because"
- Remove excessive examples (keep only the best one)

PRESERVE: All important names, facts, numbers, dates.

TARGET: Roughly half the original length while keeping meaning intact.

Output the shortened text only:""",
}

# Chat/Generation prompt - optimized for direct, complete responses like ChatGPT
CHAT_SYSTEM_PROMPT = """You are a highly capable AI assistant. Your job is to IMMEDIATELY and FULLY complete whatever the user asks.

CRITICAL RULES:
1. ALWAYS provide complete, finished work - never outlines, summaries, or partial responses
2. NEVER ask clarifying questions unless absolutely necessary - just do the task
3. When asked to write something, WRITE IT IN FULL - not an outline
4. When asked for code, provide COMPLETE, WORKING code
5. Be thorough and detailed - users want comprehensive responses
6. Start working immediately - don't explain what you're going to do

For writing requests (articles, essays, stories):
- Write the COMPLETE text, not an outline
- Include all requested word count or content
- Use engaging, polished prose
- Structure with proper paragraphs and flow

For code requests:
- Provide complete, runnable code
- Include necessary imports and setup
- Add helpful comments
- Handle edge cases

For questions:
- Give direct, comprehensive answers
- Don't hedge unnecessarily
- Be confident and helpful

Remember: Users want RESULTS, not plans. Complete the task fully."""


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
        system_prompt = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["professional"])

        # Construct the prompt using Qwen ChatML format
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{text.strip()}<|im_end|>
<|im_start|>assistant
"""

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
                stop=["<|im_end|>", "<|im_start|>", "\n\nReferences:", "\n\nSources:", "\n\n["],
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

    def chat(
        self,
        message: str,
        conversation_history: Optional[List[dict]] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Direct chat/generation with conversation memory.

        Args:
            message: User's current request or question
            conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts
            progress_callback: Called with status updates

        Returns:
            Generated response
        """
        if not self.is_loaded or not self.model:
            return "Error: Model not loaded"

        if not message or not message.strip():
            return ""

        # Build prompt with conversation history
        prompt = f"<|im_start|>system\n{CHAT_SYSTEM_PROMPT}<|im_end|>\n"

        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # Add current message
        prompt += f"<|im_start|>user\n{message.strip()}<|im_end|>\n<|im_start|>assistant\n"

        try:
            if progress_callback:
                progress_callback("Generating response...")

            response = self.model(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False
            )

            result = response["choices"][0]["text"].strip()
            return result

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}")
            return f"Error: {str(e)}"

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
