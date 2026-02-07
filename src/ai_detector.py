"""
AI Detection Module - LLM-based text analysis for AI probability scoring.
Uses local Qwen model to analyze text and detect AI-generated content.
"""

import re
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from PyQt6.QtCore import QThread, pyqtSignal


@dataclass
class SentenceAnalysis:
    """Analysis result for a single sentence."""
    text: str
    start_idx: int
    end_idx: int
    ai_probability: float  # 0.0 to 1.0
    flags: List[str]  # Reasons for AI detection


@dataclass
class TextAnalysisResult:
    """Complete analysis result for text."""
    overall_score: float  # 0-100 AI probability
    sentence_analyses: List[SentenceAnalysis]
    summary: str
    flags: List[str]


# Common AI-generated phrases that signal machine writing
AI_INDICATOR_PHRASES = [
    # Opening phrases
    r"\bin today'?s (?:world|society|age|era)\b",
    r"\bit(?:'s| is) (?:important|crucial|essential|vital) to (?:note|understand|recognize)\b",
    r"\bas we (?:navigate|delve|explore|embark)\b",
    r"\bin (?:the realm|this context|light of)\b",
    r"\blet'?s (?:dive|delve|explore)\b",

    # Transition phrases
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\badditionally\b",
    r"\bconsequently\b",
    r"\bnevertheless\b",
    r"\bnonetheless\b",
    r"\bin conclusion\b",
    r"\bto summarize\b",
    r"\bin summary\b",
    r"\ball in all\b",
    r"\bat the end of the day\b",

    # Filler phrases
    r"\bit(?:'s| is) worth (?:noting|mentioning)\b",
    r"\bneedless to say\b",
    r"\bwithout a doubt\b",
    r"\bby and large\b",
    r"\bfor the most part\b",
    r"\bin other words\b",
    r"\bthat being said\b",
    r"\bhaving said that\b",

    # Formal vocabulary
    r"\butilize[sd]?\b",
    r"\bfacilitate[sd]?\b",
    r"\bleverage[sd]?\b",
    r"\boptimize[sd]?\b",
    r"\bstreamline[sd]?\b",
    r"\bsynergy\b",
    r"\bparadigm\b",
    r"\bholistic\b",
    r"\brobust\b",
    r"\bseamless(?:ly)?\b",
    r"\bpivotal\b",
    r"\bcrucial\b",
    r"\bprofound(?:ly)?\b",
    r"\bintricate\b",
    r"\bmultifaceted\b",

    # AI-specific patterns
    r"\bplays a (?:crucial|vital|key|important|significant) role\b",
    r"\bserves as a (?:testament|reminder)\b",
    r"\bsheds light on\b",
    r"\bpaves the way\b",
    r"\bbreaks new ground\b",
    r"\bpushes the (?:boundaries|envelope)\b",
    r"\bstands as (?:a testament|proof)\b",
    r"\bin the (?:grand scheme|bigger picture)\b",
    r"\bwhen it comes to\b",
    r"\bin terms of\b",
    r"\bthe fact that\b",
    r"\bdue to the fact\b",
    r"\bin order to\b",

    # List introductions
    r"\bthere are (?:several|many|numerous|various) (?:ways|reasons|factors)\b",
    r"\b(?:first|second|third)ly\b",
    r"\bon one hand.*on the other hand\b",

    # Hedging language
    r"\bit can be (?:argued|said|noted)\b",
    r"\bone might (?:argue|say|think)\b",
    r"\bsome (?:experts|researchers|people) (?:believe|argue|suggest)\b",
]

# Compile patterns for efficiency
AI_PATTERNS = [re.compile(p, re.IGNORECASE) for p in AI_INDICATOR_PHRASES]


class AIDetector:
    """Detects AI-generated content using LLM analysis and heuristics."""

    def __init__(self):
        self.model = None
        self.is_loaded = False

    def set_model(self, model):
        """Set the LLM model for analysis."""
        self.model = model
        self.is_loaded = model is not None

    def analyze_text(self, text: str, use_llm: bool = True,
                     progress_callback: Optional[Callable] = None) -> TextAnalysisResult:
        """
        Analyze text for AI probability.

        Args:
            text: Text to analyze
            use_llm: Whether to use LLM for deeper analysis
            progress_callback: Optional callback for progress updates

        Returns:
            TextAnalysisResult with overall score and per-sentence analysis
        """
        if not text.strip():
            return TextAnalysisResult(
                overall_score=0,
                sentence_analyses=[],
                summary="No text to analyze",
                flags=[]
            )

        sentences = self._split_sentences(text)
        sentence_analyses = []
        all_flags = []

        total = len(sentences)
        for i, (sentence, start, end) in enumerate(sentences):
            if progress_callback:
                progress_callback(f"Analyzing sentence {i+1}/{total}...")

            # Heuristic analysis
            heuristic_score, flags = self._analyze_sentence_heuristics(sentence)

            # LLM analysis if available and requested
            llm_score = 0.0
            if use_llm and self.is_loaded and self.model:
                llm_score = self._analyze_sentence_llm(sentence)

            # Combine scores (weighted average)
            if use_llm and self.is_loaded:
                final_score = (heuristic_score * 0.4) + (llm_score * 0.6)
            else:
                final_score = heuristic_score

            sentence_analyses.append(SentenceAnalysis(
                text=sentence,
                start_idx=start,
                end_idx=end,
                ai_probability=min(1.0, final_score),
                flags=flags
            ))
            all_flags.extend(flags)

        # Calculate overall score
        if sentence_analyses:
            # Weight by sentence length
            total_weight = sum(len(s.text) for s in sentence_analyses)
            if total_weight > 0:
                overall = sum(s.ai_probability * len(s.text) for s in sentence_analyses) / total_weight
            else:
                overall = sum(s.ai_probability for s in sentence_analyses) / len(sentence_analyses)
        else:
            overall = 0.0

        # Additional global analysis
        global_score, global_flags = self._analyze_global_patterns(text)
        overall = (overall * 0.7) + (global_score * 0.3)
        all_flags.extend(global_flags)

        # Generate summary
        summary = self._generate_summary(overall, all_flags)

        return TextAnalysisResult(
            overall_score=min(100, overall * 100),
            sentence_analyses=sentence_analyses,
            summary=summary,
            flags=list(set(all_flags))  # Remove duplicates
        )

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with their positions."""
        # Pattern to split on sentence endings
        pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'

        sentences = []
        last_end = 0

        for match in re.finditer(pattern, text):
            end = match.start()
            sentence = text[last_end:end + 1].strip()
            if sentence:
                sentences.append((sentence, last_end, end + 1))
            last_end = match.end()

        # Add remaining text
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append((remaining, last_end, len(text)))

        # If no sentences found, treat whole text as one
        if not sentences and text.strip():
            sentences = [(text.strip(), 0, len(text))]

        return sentences

    def _analyze_sentence_heuristics(self, sentence: str) -> Tuple[float, List[str]]:
        """Analyze a sentence using heuristic patterns."""
        score = 0.0
        flags = []

        # Check for AI indicator phrases
        phrase_matches = 0
        for pattern in AI_PATTERNS:
            if pattern.search(sentence):
                phrase_matches += 1
                flags.append(f"AI phrase: {pattern.pattern[:30]}...")

        if phrase_matches > 0:
            score += min(0.4, phrase_matches * 0.1)

        # Check sentence structure
        words = sentence.split()
        word_count = len(words)

        # Very uniform sentence length is suspicious
        if 15 <= word_count <= 25:
            score += 0.05

        # Check for lack of contractions (AI often doesn't use them)
        contraction_pattern = r"\b\w+'\w+\b"
        if word_count > 10 and not re.search(contraction_pattern, sentence):
            score += 0.1
            flags.append("No contractions")

        # Check for overly formal structure
        if sentence.startswith(("Furthermore,", "Moreover,", "Additionally,", "Consequently,")):
            score += 0.15
            flags.append("Formal transition")

        # Check for passive voice (AI tends to use more passive)
        passive_pattern = r"\b(?:is|are|was|were|been|being)\s+\w+ed\b"
        if re.search(passive_pattern, sentence):
            score += 0.05

        # Check for hedging language
        hedging = r"\b(?:might|could|may|perhaps|possibly|potentially)\b"
        hedging_matches = len(re.findall(hedging, sentence, re.IGNORECASE))
        if hedging_matches >= 2:
            score += 0.1
            flags.append("Excessive hedging")

        # Check for em-dashes (AI uses these frequently)
        if "—" in sentence or " - " in sentence:
            score += 0.1
            flags.append("Em-dash usage")

        # Perfect grammar with complex structure
        semicolons = sentence.count(";")
        colons = sentence.count(":")
        if semicolons >= 1 or colons >= 1:
            score += 0.05

        return min(1.0, score), flags

    def _analyze_sentence_llm(self, sentence: str) -> float:
        """Use LLM to analyze if sentence is AI-generated."""
        if not self.model or not sentence.strip():
            return 0.0

        prompt = f"""<|im_start|>system
You are an AI detection expert. Analyze if the following sentence was likely written by AI or a human.
Consider: word choice, sentence structure, naturalness, and typical AI patterns.
Respond with ONLY a number from 0 to 100, where:
- 0 = Definitely human-written
- 100 = Definitely AI-generated
Just the number, nothing else.<|im_end|>
<|im_start|>user
Sentence: "{sentence}"<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.model(
                prompt,
                max_tokens=10,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>", "\n"]
            )

            result = response["choices"][0]["text"].strip()
            # Extract number from response
            numbers = re.findall(r'\d+', result)
            if numbers:
                score = int(numbers[0])
                return min(100, max(0, score)) / 100.0
        except Exception:
            pass

        return 0.5  # Default to uncertain

    def _analyze_global_patterns(self, text: str) -> Tuple[float, List[str]]:
        """Analyze global text patterns."""
        score = 0.0
        flags = []

        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return score, flags

        # Check burstiness (sentence length variation)
        lengths = [len(s[0].split()) for s in sentences]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            std_dev = math.sqrt(variance) if variance > 0 else 0

            # Low variance = more AI-like
            if std_dev < 3 and len(sentences) > 3:
                score += 0.15
                flags.append("Uniform sentence lengths")

            # All sentences in similar range
            if all(12 <= l <= 28 for l in lengths) and len(lengths) > 3:
                score += 0.1
                flags.append("Suspiciously consistent structure")

        # Check for repetitive patterns
        words = text.lower().split()
        if len(words) > 50:
            # Check word diversity
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:
                score += 0.1
                flags.append("Low vocabulary diversity")

        # Check paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            if para_lengths:
                avg_para = sum(para_lengths) / len(para_lengths)
                # Very uniform paragraph lengths
                if all(abs(l - avg_para) < 20 for l in para_lengths):
                    score += 0.05

        # Check for lack of personal pronouns (I, me, my)
        personal = len(re.findall(r'\b(?:I|me|my|myself)\b', text))
        if len(words) > 100 and personal == 0:
            score += 0.05
            flags.append("No first-person pronouns")

        return min(1.0, score), flags

    def _generate_summary(self, score: float, flags: List[str]) -> str:
        """Generate a human-readable summary."""
        if score < 0.2:
            return "Text appears to be human-written with high confidence."
        elif score < 0.4:
            return "Text shows some AI characteristics but is mostly human-like."
        elif score < 0.6:
            return "Text shows moderate AI patterns. Consider humanizing."
        elif score < 0.8:
            return "Text likely contains significant AI-generated content."
        else:
            return "Text strongly appears to be AI-generated."

    def get_score_label(self, score: float) -> str:
        """Get a label for the score."""
        if score < 20:
            return "Human"
        elif score < 40:
            return "Mostly Human"
        elif score < 60:
            return "Mixed"
        elif score < 80:
            return "Likely AI"
        else:
            return "AI Generated"

    def get_score_color(self, score: float) -> str:
        """Get a color for the score (for UI display)."""
        if score < 30:
            return "#10B981"  # Emerald (human)
        elif score < 60:
            return "#F59E0B"  # Amber (mixed)
        else:
            return "#EF4444"  # Red (AI)


class AIDetectorWorker(QThread):
    """Background worker for AI detection analysis."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # TextAnalysisResult

    def __init__(self, detector: AIDetector, text: str, use_llm: bool = True):
        super().__init__()
        self.detector = detector
        self.text = text
        self.use_llm = use_llm
        self._cancelled = False

    def run(self):
        try:
            result = self.detector.analyze_text(
                self.text,
                use_llm=self.use_llm,
                progress_callback=lambda msg: self.progress.emit(msg) if not self._cancelled else None
            )
            if not self._cancelled:
                self.finished.emit(result)
        except Exception as e:
            if not self._cancelled:
                self.finished.emit(TextAnalysisResult(
                    overall_score=0,
                    sentence_analyses=[],
                    summary=f"Analysis failed: {str(e)}",
                    flags=[]
                ))

    def cancel(self):
        self._cancelled = True
