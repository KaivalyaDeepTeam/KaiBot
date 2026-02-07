"""
Diff Viewer Module - Side-by-side comparison of original and humanized text.
Highlights additions, deletions, and modifications between texts.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
from difflib import SequenceMatcher, ndiff


class ChangeType(Enum):
    """Types of changes between texts."""
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class DiffSegment:
    """A segment of text with change information."""
    text: str
    change_type: ChangeType
    original_text: Optional[str] = None  # For modified segments


@dataclass
class DiffResult:
    """Result of comparing two texts."""
    original_segments: List[DiffSegment]
    modified_segments: List[DiffSegment]
    similarity_ratio: float
    stats: dict


class DiffViewer:
    """Compares and highlights differences between original and humanized text."""

    def __init__(self):
        self.color_unchanged = "#F9FAFB"  # Light gray
        self.color_added = "#D1FAE5"      # Green tint
        self.color_removed = "#FEE2E2"    # Red tint
        self.color_modified = "#FEF3C7"   # Yellow tint

    def compare(self, original: str, modified: str) -> DiffResult:
        """
        Compare original and modified text.

        Args:
            original: Original text
            modified: Modified/humanized text

        Returns:
            DiffResult with segments and statistics
        """
        if not original and not modified:
            return DiffResult(
                original_segments=[],
                modified_segments=[],
                similarity_ratio=1.0,
                stats={"additions": 0, "deletions": 0, "modifications": 0}
            )

        # Split into words for comparison
        original_words = self._tokenize(original)
        modified_words = self._tokenize(modified)

        # Use SequenceMatcher for comparison
        matcher = SequenceMatcher(None, original_words, modified_words)
        similarity = matcher.ratio()

        original_segments = []
        modified_segments = []

        stats = {"additions": 0, "deletions": 0, "modifications": 0, "unchanged": 0}

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            original_chunk = ' '.join(original_words[i1:i2])
            modified_chunk = ' '.join(modified_words[j1:j2])

            if opcode == 'equal':
                if original_chunk:
                    original_segments.append(DiffSegment(
                        text=original_chunk,
                        change_type=ChangeType.UNCHANGED
                    ))
                    modified_segments.append(DiffSegment(
                        text=modified_chunk,
                        change_type=ChangeType.UNCHANGED
                    ))
                    stats["unchanged"] += len(original_words[i1:i2])

            elif opcode == 'replace':
                original_segments.append(DiffSegment(
                    text=original_chunk,
                    change_type=ChangeType.MODIFIED,
                    original_text=original_chunk
                ))
                modified_segments.append(DiffSegment(
                    text=modified_chunk,
                    change_type=ChangeType.MODIFIED,
                    original_text=original_chunk
                ))
                stats["modifications"] += max(len(original_words[i1:i2]), len(modified_words[j1:j2]))

            elif opcode == 'delete':
                original_segments.append(DiffSegment(
                    text=original_chunk,
                    change_type=ChangeType.REMOVED
                ))
                stats["deletions"] += len(original_words[i1:i2])

            elif opcode == 'insert':
                modified_segments.append(DiffSegment(
                    text=modified_chunk,
                    change_type=ChangeType.ADDED
                ))
                stats["additions"] += len(modified_words[j1:j2])

        return DiffResult(
            original_segments=original_segments,
            modified_segments=modified_segments,
            similarity_ratio=similarity,
            stats=stats
        )

    def compare_sentences(self, original: str, modified: str) -> DiffResult:
        """Compare texts at sentence level for broader view."""
        original_sentences = self._split_sentences(original)
        modified_sentences = self._split_sentences(modified)

        matcher = SequenceMatcher(None, original_sentences, modified_sentences)
        similarity = matcher.ratio()

        original_segments = []
        modified_segments = []
        stats = {"additions": 0, "deletions": 0, "modifications": 0, "unchanged": 0}

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            original_chunk = ' '.join(original_sentences[i1:i2])
            modified_chunk = ' '.join(modified_sentences[j1:j2])

            if opcode == 'equal':
                if original_chunk:
                    original_segments.append(DiffSegment(
                        text=original_chunk,
                        change_type=ChangeType.UNCHANGED
                    ))
                    modified_segments.append(DiffSegment(
                        text=modified_chunk,
                        change_type=ChangeType.UNCHANGED
                    ))
                    stats["unchanged"] += i2 - i1

            elif opcode == 'replace':
                original_segments.append(DiffSegment(
                    text=original_chunk,
                    change_type=ChangeType.MODIFIED
                ))
                modified_segments.append(DiffSegment(
                    text=modified_chunk,
                    change_type=ChangeType.MODIFIED
                ))
                stats["modifications"] += max(i2 - i1, j2 - j1)

            elif opcode == 'delete':
                original_segments.append(DiffSegment(
                    text=original_chunk,
                    change_type=ChangeType.REMOVED
                ))
                stats["deletions"] += i2 - i1

            elif opcode == 'insert':
                modified_segments.append(DiffSegment(
                    text=modified_chunk,
                    change_type=ChangeType.ADDED
                ))
                stats["additions"] += j2 - j1

        return DiffResult(
            original_segments=original_segments,
            modified_segments=modified_segments,
            similarity_ratio=similarity,
            stats=stats
        )

    def to_html(self, result: DiffResult, show_original: bool = True) -> Tuple[str, str]:
        """
        Convert diff result to HTML for display.

        Args:
            result: DiffResult from compare()
            show_original: Whether to include original text HTML

        Returns:
            Tuple of (original_html, modified_html)
        """
        original_html = self._segments_to_html(result.original_segments, is_original=True)
        modified_html = self._segments_to_html(result.modified_segments, is_original=False)

        return original_html, modified_html

    def _segments_to_html(self, segments: List[DiffSegment], is_original: bool) -> str:
        """Convert segments to HTML with highlighting."""
        html_parts = []

        for segment in segments:
            text = self._escape_html(segment.text)

            if segment.change_type == ChangeType.UNCHANGED:
                html_parts.append(f'<span style="background-color: transparent;">{text}</span>')

            elif segment.change_type == ChangeType.ADDED:
                html_parts.append(
                    f'<span style="background-color: {self.color_added}; '
                    f'border-radius: 3px; padding: 1px 2px;">{text}</span>'
                )

            elif segment.change_type == ChangeType.REMOVED:
                html_parts.append(
                    f'<span style="background-color: {self.color_removed}; '
                    f'text-decoration: line-through; border-radius: 3px; padding: 1px 2px;">{text}</span>'
                )

            elif segment.change_type == ChangeType.MODIFIED:
                html_parts.append(
                    f'<span style="background-color: {self.color_modified}; '
                    f'border-radius: 3px; padding: 1px 2px;">{text}</span>'
                )

        return ' '.join(html_parts)

    def get_change_summary(self, result: DiffResult) -> str:
        """Get a human-readable summary of changes."""
        stats = result.stats
        total_changes = stats["additions"] + stats["deletions"] + stats["modifications"]

        if total_changes == 0:
            return "No changes detected"

        parts = []
        if stats["additions"] > 0:
            parts.append(f"{stats['additions']} added")
        if stats["deletions"] > 0:
            parts.append(f"{stats['deletions']} removed")
        if stats["modifications"] > 0:
            parts.append(f"{stats['modifications']} modified")

        similarity_pct = int(result.similarity_ratio * 100)
        return f"{', '.join(parts)} ({similarity_pct}% similar)"

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens (words and punctuation)."""
        if not text:
            return []
        # Split on whitespace but keep punctuation attached
        return text.split()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))


# Singleton instance
_diff_viewer = DiffViewer()


def compare_texts(original: str, modified: str) -> DiffResult:
    """Compare two texts and return diff result."""
    return _diff_viewer.compare(original, modified)


def get_diff_html(original: str, modified: str) -> Tuple[str, str]:
    """Get HTML diff of two texts."""
    result = _diff_viewer.compare(original, modified)
    return _diff_viewer.to_html(result)


def get_change_summary(original: str, modified: str) -> str:
    """Get summary of changes between texts."""
    result = _diff_viewer.compare(original, modified)
    return _diff_viewer.get_change_summary(result)
