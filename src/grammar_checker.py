"""
Grammar Checker Module - Basic grammar and spelling error detection.
Uses pattern-based rules for common errors without external dependencies.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class ErrorType(Enum):
    """Types of grammar/spelling errors."""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    STYLE = "style"
    CAPITALIZATION = "capitalization"


@dataclass
class GrammarError:
    """Represents a grammar or spelling error."""
    start: int
    end: int
    text: str
    error_type: ErrorType
    message: str
    suggestions: List[str]
    rule_id: str


# Common spelling mistakes and corrections
COMMON_MISSPELLINGS = {
    "teh": "the",
    "taht": "that",
    "recieve": "receive",
    "occured": "occurred",
    "occurence": "occurrence",
    "seperate": "separate",
    "definately": "definitely",
    "accomodate": "accommodate",
    "occassion": "occasion",
    "untill": "until",
    "alot": "a lot",
    "thier": "their",
    "beleive": "believe",
    "wierd": "weird",
    "freind": "friend",
    "goverment": "government",
    "enviroment": "environment",
    "recomend": "recommend",
    "begining": "beginning",
    "independant": "independent",
    "neccessary": "necessary",
    "tommorow": "tomorrow",
    "wich": "which",
    "becuase": "because",
    "probaly": "probably",
    "calender": "calendar",
    "commited": "committed",
    "concious": "conscious",
    "embarass": "embarrass",
    "existance": "existence",
    "familar": "familiar",
    "finaly": "finally",
    "grammer": "grammar",
    "harrass": "harass",
    "immediatly": "immediately",
    "knowlege": "knowledge",
    "libary": "library",
    "lisence": "license",
    "maintainance": "maintenance",
    "millenium": "millennium",
    "mispell": "misspell",
    "noticable": "noticeable",
    "paralell": "parallel",
    "percieve": "perceive",
    "persue": "pursue",
    "posession": "possession",
    "privelege": "privilege",
    "publically": "publicly",
    "rythm": "rhythm",
    "sieze": "seize",
    "similiar": "similar",
    "succesful": "successful",
    "suprise": "surprise",
    "truely": "truly",
    "visable": "visible",
    "writting": "writing",
}

# Grammar rules as (pattern, error_message, suggestions, rule_id)
GRAMMAR_RULES = [
    # Double words
    (r'\b(\w+)\s+\1\b', "Repeated word", lambda m: [m.group(1)], "REPEATED_WORD"),

    # a/an confusion
    (r'\ba\s+([aeiouAEIOU]\w+)', "Use 'an' before vowel sounds",
     lambda m: [f"an {m.group(1)}"], "A_AN_VOWEL"),
    (r'\ban\s+([^aeiouAEIOU\s]\w+)', "Use 'a' before consonant sounds",
     lambda m: [f"a {m.group(1)}"], "AN_A_CONSONANT"),

    # Subject-verb agreement
    (r'\b(he|she|it)\s+(are|were|have)\b', "Subject-verb disagreement",
     lambda m: [f"{m.group(1)} {'is' if m.group(2) == 'are' else 'was' if m.group(2) == 'were' else 'has'}"],
     "SUBJECT_VERB"),
    (r'\b(they|we|you)\s+(is|was|has)\b', "Subject-verb disagreement",
     lambda m: [f"{m.group(1)} {'are' if m.group(2) == 'is' else 'were' if m.group(2) == 'was' else 'have'}"],
     "SUBJECT_VERB"),

    # Common confusions
    (r'\b(your)\s+(welcome|right|wrong|the\s+best)\b', "Should be 'you're' (you are)",
     lambda m: [f"you're {m.group(2)}"], "YOUR_YOURE"),
    (r"\b(its)\s+(a|been|going|not|time)\b", "Should be 'it's' (it is/has)",
     lambda m: [f"it's {m.group(2)}"], "ITS_ITS"),
    (r'\b(there)\s+(going|coming|leaving|doing)\b', "Should be 'they're' (they are)",
     lambda m: [f"they're {m.group(2)}"], "THERE_THEYRE"),
    (r'\b(their)\s+(is|are|was|were)\b', "Should be 'there'",
     lambda m: [f"there {m.group(2)}"], "THEIR_THERE"),
    (r'\b(then)\s+(I|you|he|she|it|we|they)\b', "Should be 'than' for comparisons",
     lambda m: [f"than {m.group(2)}"], "THEN_THAN"),
    (r'\b(could|would|should)\s+of\b', "Should be 'have' not 'of'",
     lambda m: [f"{m.group(1)} have"], "OF_HAVE"),
    (r'\b(affect)\b(?=.*\bnoun\b)', "Effect is typically the noun, affect is the verb",
     lambda m: ["effect"], "AFFECT_EFFECT"),
    (r'\b(loose)\s+(weight|control|grip)\b', "Should be 'lose'",
     lambda m: [f"lose {m.group(2)}"], "LOOSE_LOSE"),

    # Double negatives
    (r"\b(don'?t|can'?t|won'?t|isn'?t|aren'?t)\s+\w*\s*(no|nothing|nobody|nowhere|never)\b",
     "Double negative", lambda m: [], "DOUBLE_NEGATIVE"),

    # Missing comma after introductory phrases
    (r'^(However|Therefore|Furthermore|Moreover|Nevertheless|Additionally|Consequently)\s+([A-Z])',
     "Add comma after introductory word",
     lambda m: [f"{m.group(1)}, {m.group(2)}"], "INTRO_COMMA"),

    # Run-on sentences (simple detection)
    (r'([.!?])\s*([a-z])', "Capitalize after sentence-ending punctuation",
     lambda m: [f"{m.group(1)} {m.group(2).upper()}"], "CAPITALIZE_SENTENCE"),
]

# Punctuation rules
PUNCTUATION_RULES = [
    # Space before punctuation
    (r'\s+([,.:;!?])', "Remove space before punctuation",
     lambda m: [m.group(1)], "SPACE_BEFORE_PUNCT"),

    # Missing space after punctuation
    (r'([,.:;!?])([A-Za-z])', "Add space after punctuation",
     lambda m: [f"{m.group(1)} {m.group(2)}"], "SPACE_AFTER_PUNCT"),

    # Multiple punctuation
    (r'([.!?]){2,}', "Multiple punctuation marks",
     lambda m: [m.group(1)], "MULTIPLE_PUNCT"),

    # Multiple spaces
    (r'  +', "Multiple spaces",
     lambda m: [" "], "MULTIPLE_SPACES"),

    # Comma splice (basic)
    (r',\s*(however|therefore|moreover|furthermore|nevertheless)\s*,',
     "Use semicolon or period before conjunctive adverb",
     lambda m: [f"; {m.group(1)},"], "COMMA_SPLICE"),
]

# Style suggestions
STYLE_RULES = [
    # Passive voice (simple detection)
    (r'\b(is|are|was|were|been|being)\s+(\w+ed)\b',
     "Consider using active voice",
     lambda m: [], "PASSIVE_VOICE"),

    # Wordy phrases
    (r'\bin order to\b', "Simplify to 'to'",
     lambda m: ["to"], "WORDY_IN_ORDER_TO"),
    (r'\bdue to the fact that\b', "Simplify to 'because'",
     lambda m: ["because"], "WORDY_DUE_TO"),
    (r'\bat this point in time\b', "Simplify to 'now' or 'currently'",
     lambda m: ["now", "currently"], "WORDY_AT_THIS_TIME"),
    (r'\bin the event that\b', "Simplify to 'if'",
     lambda m: ["if"], "WORDY_IN_EVENT"),
    (r'\bfor the purpose of\b', "Simplify to 'to' or 'for'",
     lambda m: ["to", "for"], "WORDY_FOR_PURPOSE"),
    (r'\bwith regard to\b', "Simplify to 'about' or 'regarding'",
     lambda m: ["about", "regarding"], "WORDY_WITH_REGARD"),
    (r'\bin spite of the fact that\b', "Simplify to 'although' or 'despite'",
     lambda m: ["although", "despite"], "WORDY_IN_SPITE"),
]


class GrammarChecker:
    """Checks text for grammar, spelling, and style issues."""

    def __init__(self, check_spelling: bool = True, check_grammar: bool = True,
                 check_punctuation: bool = True, check_style: bool = True):
        self.check_spelling = check_spelling
        self.check_grammar = check_grammar
        self.check_punctuation = check_punctuation
        self.check_style = check_style

        # Compile patterns
        self.grammar_patterns = [
            (re.compile(p, re.IGNORECASE | re.MULTILINE), msg, sugg, rid)
            for p, msg, sugg, rid in GRAMMAR_RULES
        ]
        self.punctuation_patterns = [
            (re.compile(p), msg, sugg, rid)
            for p, msg, sugg, rid in PUNCTUATION_RULES
        ]
        self.style_patterns = [
            (re.compile(p, re.IGNORECASE), msg, sugg, rid)
            for p, msg, sugg, rid in STYLE_RULES
        ]
        self.spelling_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in COMMON_MISSPELLINGS.keys()) + r')\b',
            re.IGNORECASE
        )

    def check(self, text: str) -> List[GrammarError]:
        """
        Check text for errors.

        Args:
            text: Text to check

        Returns:
            List of GrammarError objects
        """
        errors = []

        if self.check_spelling:
            errors.extend(self._check_spelling(text))

        if self.check_grammar:
            errors.extend(self._check_patterns(text, self.grammar_patterns, ErrorType.GRAMMAR))

        if self.check_punctuation:
            errors.extend(self._check_patterns(text, self.punctuation_patterns, ErrorType.PUNCTUATION))

        if self.check_style:
            errors.extend(self._check_patterns(text, self.style_patterns, ErrorType.STYLE))

        # Sort by position
        errors.sort(key=lambda e: e.start)

        # Remove overlapping errors (keep first)
        filtered = []
        last_end = -1
        for error in errors:
            if error.start >= last_end:
                filtered.append(error)
                last_end = error.end

        return filtered

    def _check_spelling(self, text: str) -> List[GrammarError]:
        """Check for common spelling mistakes."""
        errors = []

        for match in self.spelling_pattern.finditer(text):
            word = match.group(1).lower()
            correction = COMMON_MISSPELLINGS.get(word)

            if correction:
                # Preserve original capitalization
                if match.group(1).isupper():
                    suggestion = correction.upper()
                elif match.group(1)[0].isupper():
                    suggestion = correction.capitalize()
                else:
                    suggestion = correction

                errors.append(GrammarError(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(1),
                    error_type=ErrorType.SPELLING,
                    message=f"Possible spelling mistake: '{match.group(1)}'",
                    suggestions=[suggestion],
                    rule_id="SPELLING"
                ))

        return errors

    def _check_patterns(self, text: str, patterns: List[Tuple], error_type: ErrorType) -> List[GrammarError]:
        """Check text against a list of patterns."""
        errors = []

        for pattern, message, suggestion_func, rule_id in patterns:
            for match in pattern.finditer(text):
                suggestions = suggestion_func(match) if callable(suggestion_func) else []

                errors.append(GrammarError(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    error_type=error_type,
                    message=message,
                    suggestions=suggestions,
                    rule_id=rule_id
                ))

        return errors

    def get_error_count(self, text: str) -> dict:
        """Get count of errors by type."""
        errors = self.check(text)
        counts = {
            ErrorType.SPELLING: 0,
            ErrorType.GRAMMAR: 0,
            ErrorType.PUNCTUATION: 0,
            ErrorType.STYLE: 0,
            ErrorType.CAPITALIZATION: 0
        }
        for error in errors:
            counts[error.error_type] += 1
        return counts

    def get_total_errors(self, text: str) -> int:
        """Get total number of errors."""
        return len(self.check(text))

    def apply_suggestion(self, text: str, error: GrammarError, suggestion_index: int = 0) -> str:
        """Apply a suggestion to fix an error."""
        if not error.suggestions or suggestion_index >= len(error.suggestions):
            return text

        suggestion = error.suggestions[suggestion_index]
        return text[:error.start] + suggestion + text[error.end:]

    def apply_all_suggestions(self, text: str) -> Tuple[str, int]:
        """
        Apply all first suggestions to fix errors.

        Returns:
            Tuple of (corrected_text, number_of_fixes)
        """
        errors = self.check(text)
        if not errors:
            return text, 0

        # Apply in reverse order to preserve positions
        fixed_count = 0
        for error in reversed(errors):
            if error.suggestions:
                text = self.apply_suggestion(text, error, 0)
                fixed_count += 1

        return text, fixed_count


# Global instance
_checker = GrammarChecker()


def check_grammar(text: str) -> List[GrammarError]:
    """Check text for grammar errors."""
    return _checker.check(text)


def get_error_count(text: str) -> dict:
    """Get error counts by type."""
    return _checker.get_error_count(text)


def auto_fix(text: str) -> Tuple[str, int]:
    """Auto-fix all detected errors."""
    return _checker.apply_all_suggestions(text)
