"""
Text Analyzer Module - Writing statistics, readability, and tone detection.
Provides real-time analysis of text for the KaiBot UI.
"""

import re
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter


@dataclass
class TextStats:
    """Statistics about text."""
    char_count: int
    char_count_no_spaces: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_word_length: float
    avg_sentence_length: float
    reading_time_seconds: int
    speaking_time_seconds: int


@dataclass
class ReadabilityScores:
    """Readability analysis results."""
    flesch_reading_ease: float  # 0-100, higher = easier
    flesch_kincaid_grade: float  # Grade level
    grade_label: str  # "Grade 8", "College", etc.
    difficulty: str  # "Easy", "Moderate", "Difficult"


@dataclass
class ToneAnalysis:
    """Tone detection results."""
    primary_tone: str
    confidence: float
    tones: Dict[str, float]  # All detected tones with scores


@dataclass
class VocabularyAnalysis:
    """Vocabulary diversity analysis."""
    unique_words: int
    total_words: int
    diversity_ratio: float
    rare_words: List[str]
    repeated_words: List[Tuple[str, int]]


@dataclass
class FullTextAnalysis:
    """Complete text analysis result."""
    stats: TextStats
    readability: ReadabilityScores
    tone: ToneAnalysis
    vocabulary: VocabularyAnalysis


# Tone indicator words
TONE_INDICATORS = {
    "confident": [
        "definitely", "certainly", "absolutely", "clearly", "obviously",
        "undoubtedly", "surely", "must", "will", "proven", "guaranteed",
        "without doubt", "no question", "for sure", "unquestionably"
    ],
    "friendly": [
        "hey", "hi", "hello", "thanks", "thank you", "please", "appreciate",
        "glad", "happy", "excited", "love", "awesome", "great", "wonderful",
        "amazing", "cool", "nice", "sweet", "cheers", "buddy", "friend"
    ],
    "formal": [
        "hereby", "therefore", "thus", "hence", "whereas", "notwithstanding",
        "furthermore", "moreover", "consequently", "accordingly", "subsequently",
        "regarding", "concerning", "pursuant", "aforementioned", "hereafter"
    ],
    "casual": [
        "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep", "nope",
        "stuff", "things", "basically", "literally", "actually", "honestly",
        "like", "you know", "i mean", "right", "okay", "ok", "btw", "tbh"
    ],
    "academic": [
        "research", "study", "analysis", "hypothesis", "methodology", "findings",
        "evidence", "data", "significant", "correlation", "theoretical",
        "empirical", "literature", "framework", "paradigm", "discourse"
    ],
    "persuasive": [
        "should", "must", "need to", "have to", "important", "crucial", "vital",
        "essential", "critical", "urgent", "imagine", "consider", "think about",
        "remember", "don't forget", "keep in mind", "realize"
    ],
    "neutral": [
        "is", "are", "was", "were", "has", "have", "had", "will", "would",
        "can", "could", "may", "might", "this", "that", "these", "those"
    ]
}

# Common words to exclude from vocabulary analysis
COMMON_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "own", "same", "so", "than", "too", "very"
}


class TextAnalyzer:
    """Analyzes text for statistics, readability, and tone."""

    def __init__(self):
        # Compile tone patterns for efficiency
        self.tone_patterns = {}
        for tone, words in TONE_INDICATORS.items():
            pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
            self.tone_patterns[tone] = re.compile(pattern, re.IGNORECASE)

    def analyze(self, text: str) -> FullTextAnalysis:
        """Perform complete text analysis."""
        stats = self.get_stats(text)
        readability = self.get_readability(text)
        tone = self.get_tone(text)
        vocabulary = self.get_vocabulary(text)

        return FullTextAnalysis(
            stats=stats,
            readability=readability,
            tone=tone,
            vocabulary=vocabulary
        )

    def get_stats(self, text: str) -> TextStats:
        """Calculate basic text statistics."""
        if not text.strip():
            return TextStats(
                char_count=0,
                char_count_no_spaces=0,
                word_count=0,
                sentence_count=0,
                paragraph_count=0,
                avg_word_length=0,
                avg_sentence_length=0,
                reading_time_seconds=0,
                speaking_time_seconds=0
            )

        # Character counts
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

        # Word count
        words = self._get_words(text)
        word_count = len(words)

        # Sentence count
        sentences = self._get_sentences(text)
        sentence_count = len(sentences) if sentences else 1

        # Paragraph count
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs) if paragraphs else 1

        # Average word length
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0

        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Reading time (average 200-250 words per minute)
        reading_time_seconds = int((word_count / 200) * 60)

        # Speaking time (average 150 words per minute)
        speaking_time_seconds = int((word_count / 150) * 60)

        return TextStats(
            char_count=char_count,
            char_count_no_spaces=char_count_no_spaces,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_word_length=round(avg_word_length, 1),
            avg_sentence_length=round(avg_sentence_length, 1),
            reading_time_seconds=reading_time_seconds,
            speaking_time_seconds=speaking_time_seconds
        )

    def get_readability(self, text: str) -> ReadabilityScores:
        """Calculate readability scores."""
        if not text.strip():
            return ReadabilityScores(
                flesch_reading_ease=0,
                flesch_kincaid_grade=0,
                grade_label="N/A",
                difficulty="N/A"
            )

        words = self._get_words(text)
        sentences = self._get_sentences(text)
        syllables = sum(self._count_syllables(w) for w in words)

        word_count = len(words)
        sentence_count = len(sentences) if sentences else 1

        if word_count == 0 or sentence_count == 0:
            return ReadabilityScores(
                flesch_reading_ease=0,
                flesch_kincaid_grade=0,
                grade_label="N/A",
                difficulty="N/A"
            )

        # Flesch Reading Ease
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        asl = word_count / sentence_count  # Average sentence length
        asw = syllables / word_count  # Average syllables per word

        fre = 206.835 - (1.015 * asl) - (84.6 * asw)
        fre = max(0, min(100, fre))

        # Flesch-Kincaid Grade Level
        # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        fkg = (0.39 * asl) + (11.8 * asw) - 15.59
        fkg = max(0, min(18, fkg))

        # Grade label
        if fkg < 1:
            grade_label = "Kindergarten"
        elif fkg < 6:
            grade_label = f"Grade {int(fkg)}"
        elif fkg < 9:
            grade_label = f"Grade {int(fkg)}"
        elif fkg < 13:
            grade_label = f"Grade {int(fkg)}"
        elif fkg < 17:
            grade_label = "College"
        else:
            grade_label = "Graduate"

        # Difficulty
        if fre >= 80:
            difficulty = "Very Easy"
        elif fre >= 60:
            difficulty = "Easy"
        elif fre >= 40:
            difficulty = "Moderate"
        elif fre >= 20:
            difficulty = "Difficult"
        else:
            difficulty = "Very Difficult"

        return ReadabilityScores(
            flesch_reading_ease=round(fre, 1),
            flesch_kincaid_grade=round(fkg, 1),
            grade_label=grade_label,
            difficulty=difficulty
        )

    def get_tone(self, text: str) -> ToneAnalysis:
        """Detect the tone of the text."""
        if not text.strip():
            return ToneAnalysis(
                primary_tone="Neutral",
                confidence=0,
                tones={}
            )

        text_lower = text.lower()
        word_count = len(self._get_words(text))

        if word_count == 0:
            return ToneAnalysis(
                primary_tone="Neutral",
                confidence=0,
                tones={}
            )

        tone_scores = {}

        for tone, pattern in self.tone_patterns.items():
            matches = pattern.findall(text_lower)
            # Normalize by word count
            score = len(matches) / word_count * 100
            tone_scores[tone] = round(score, 2)

        # Find primary tone
        if tone_scores:
            primary_tone = max(tone_scores, key=tone_scores.get)
            confidence = tone_scores[primary_tone]

            # Adjust confidence based on score magnitude
            if confidence < 1:
                confidence = confidence * 50  # Scale up low scores
            else:
                confidence = min(100, confidence * 10)

            # If no strong tone detected, default to neutral
            if max(tone_scores.values()) < 0.5:
                primary_tone = "Neutral"
                confidence = 50
        else:
            primary_tone = "Neutral"
            confidence = 50

        # Format tone name
        primary_tone = primary_tone.title()

        return ToneAnalysis(
            primary_tone=primary_tone,
            confidence=round(confidence, 1),
            tones=tone_scores
        )

    def get_vocabulary(self, text: str) -> VocabularyAnalysis:
        """Analyze vocabulary diversity."""
        words = self._get_words(text)

        if not words:
            return VocabularyAnalysis(
                unique_words=0,
                total_words=0,
                diversity_ratio=0,
                rare_words=[],
                repeated_words=[]
            )

        # Normalize words
        words_lower = [w.lower() for w in words]
        word_counts = Counter(words_lower)

        # Unique words (excluding common words)
        content_words = [w for w in words_lower if w not in COMMON_WORDS]
        unique_content = set(content_words)

        # Diversity ratio
        unique_count = len(set(words_lower))
        total_count = len(words_lower)
        diversity_ratio = unique_count / total_count if total_count > 0 else 0

        # Rare words (appear once, longer than 6 chars, not common)
        rare_words = [
            w for w, count in word_counts.items()
            if count == 1 and len(w) > 6 and w not in COMMON_WORDS
        ][:10]  # Limit to 10

        # Most repeated non-common words
        repeated = [
            (w, count) for w, count in word_counts.most_common(20)
            if w not in COMMON_WORDS and count > 1
        ][:5]

        return VocabularyAnalysis(
            unique_words=len(unique_content),
            total_words=len(content_words),
            diversity_ratio=round(diversity_ratio, 2),
            rare_words=rare_words,
            repeated_words=repeated
        )

    def _get_words(self, text: str) -> List[str]:
        """Extract words from text."""
        # Match word characters and apostrophes
        return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)

    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        if len(word) <= 2:
            return 1

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e") and count > 1:
            count -= 1

        # Adjust for -le endings
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        return max(1, count)

    def format_reading_time(self, seconds: int) -> str:
        """Format reading time as human-readable string."""
        if seconds < 60:
            return f"{seconds}s read"
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds == 0:
            return f"{minutes}min read"
        return f"{minutes}m {remaining_seconds}s read"

    def format_stats_summary(self, stats: TextStats) -> str:
        """Format stats as a compact summary string."""
        time_str = self.format_reading_time(stats.reading_time_seconds)
        return f"{stats.word_count} words | {stats.sentence_count} sentences | {time_str}"


# Singleton instance for convenience
_analyzer = TextAnalyzer()


def analyze_text(text: str) -> FullTextAnalysis:
    """Convenience function for text analysis."""
    return _analyzer.analyze(text)


def get_stats(text: str) -> TextStats:
    """Convenience function for text stats."""
    return _analyzer.get_stats(text)


def get_readability(text: str) -> ReadabilityScores:
    """Convenience function for readability."""
    return _analyzer.get_readability(text)


def get_tone(text: str) -> ToneAnalysis:
    """Convenience function for tone detection."""
    return _analyzer.get_tone(text)
