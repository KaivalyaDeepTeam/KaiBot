"""
Text Chunker - Smart text splitting for LLM processing.
Preserves sentence boundaries and handles various text structures.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text ready for LLM processing."""
    text: str
    start_idx: int  # Character index in original text
    end_idx: int
    chunk_type: str = "paragraph"  # paragraph, list, header, etc.


class TextChunker:
    """
    Splits text into LLM-friendly chunks while preserving
    sentence boundaries and maintaining context.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        chars_per_token: float = 4.0
    ):
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk (approximate)
            overlap_tokens: Overlap between chunks for context
            chars_per_token: Estimated characters per token
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chars_per_token = chars_per_token
        self.max_chars = int(max_tokens * chars_per_token)
        self.overlap_chars = int(overlap_tokens * chars_per_token)

        # Sentence ending patterns
        self.sentence_end_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        )

        # List item pattern
        self.list_pattern = re.compile(
            r'^[\s]*[-*\u2022\u2023\u25E6\u2043\u2219]\s+|^[\s]*\d+[.)\]]\s+',
            re.MULTILINE
        )

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks suitable for LLM processing.
        Preserves sentence boundaries where possible.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # If text is short enough, return as single chunk
        if len(text) <= self.max_chars:
            return [TextChunk(
                text=text,
                start_idx=0,
                end_idx=len(text),
                chunk_type=self._detect_chunk_type(text)
            )]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Determine chunk end position
            chunk_end = min(current_pos + self.max_chars, len(text))

            # If we're not at the end, try to break at a sentence boundary
            if chunk_end < len(text):
                chunk_end = self._find_sentence_boundary(
                    text, current_pos, chunk_end
                )

            chunk_text = text[current_pos:chunk_end].strip()

            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_idx=current_pos,
                    end_idx=chunk_end,
                    chunk_type=self._detect_chunk_type(chunk_text)
                ))

            # Move position, accounting for overlap
            if chunk_end >= len(text):
                break

            # Find a good starting point for next chunk (with overlap)
            next_start = chunk_end - self.overlap_chars
            # Ensure we don't go backwards
            next_start = max(next_start, current_pos + 1)
            # Try to start at a sentence boundary
            next_start = self._find_sentence_start(text, next_start, chunk_end)

            current_pos = next_start

        return chunks

    def _find_sentence_boundary(
        self,
        text: str,
        start: int,
        max_end: int
    ) -> int:
        """Find the best sentence boundary before max_end."""
        search_text = text[start:max_end]

        # Find all sentence endings in the chunk
        sentence_ends = []
        for match in self.sentence_end_pattern.finditer(search_text):
            sentence_ends.append(start + match.start())

        if sentence_ends:
            # Return the last sentence boundary
            return sentence_ends[-1] + 1

        # No sentence boundary found, try to break at newline
        newline_pos = search_text.rfind('\n')
        if newline_pos > len(search_text) * 0.5:  # Only if in latter half
            return start + newline_pos + 1

        # Try to break at space
        space_pos = search_text.rfind(' ')
        if space_pos > len(search_text) * 0.7:
            return start + space_pos + 1

        # Fall back to max_end
        return max_end

    def _find_sentence_start(
        self,
        text: str,
        target_start: int,
        max_start: int
    ) -> int:
        """Find a good sentence start near target_start."""
        search_text = text[target_start:max_start]

        # Look for capital letter after sentence ending
        match = self.sentence_end_pattern.search(search_text)
        if match:
            return target_start + match.end()

        # Look for newline
        newline_pos = search_text.find('\n')
        if newline_pos != -1:
            return target_start + newline_pos + 1

        return target_start

    def _detect_chunk_type(self, text: str) -> str:
        """Detect the type of text chunk."""
        text_stripped = text.strip()

        # Check for list items
        if self.list_pattern.match(text_stripped):
            return "list"

        # Check for header (short, no period at end)
        lines = text_stripped.split('\n')
        if len(lines) == 1 and len(text_stripped) < 100:
            if not text_stripped.endswith('.'):
                return "header"

        # Check for mostly bullet points
        list_matches = len(self.list_pattern.findall(text_stripped))
        if list_matches > 2:
            return "list"

        return "paragraph"

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text."""
        return int(len(text) / self.chars_per_token)


class SmartChunker(TextChunker):
    """
    Enhanced chunker that handles special content types
    like tables, code blocks, and maintains better context.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Pattern for table-like content (aligned columns)
        self.table_pattern = re.compile(
            r'(?:^[^\n]*\t[^\n]*$\n?){2,}|(?:^[^\n]*\|[^\n]*$\n?){2,}',
            re.MULTILINE
        )

        # Pattern for code-like content
        self.code_pattern = re.compile(
            r'(?:^[ ]{4,}[^\n]+$\n?){2,}|```[\s\S]*?```',
            re.MULTILINE
        )

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Enhanced chunking that preserves special structures.
        """
        if not text or not text.strip():
            return []

        # Pre-process: identify special regions that shouldn't be split
        protected_regions = self._find_protected_regions(text)

        if not protected_regions:
            return super().chunk_text(text)

        # Split text around protected regions
        chunks = []
        current_pos = 0

        for region_start, region_end, region_type in protected_regions:
            # Process text before the protected region
            if current_pos < region_start:
                pre_text = text[current_pos:region_start].strip()
                if pre_text:
                    pre_chunks = super().chunk_text(pre_text)
                    for chunk in pre_chunks:
                        chunk.start_idx += current_pos
                        chunk.end_idx += current_pos
                    chunks.extend(pre_chunks)

            # Add the protected region as a single chunk
            region_text = text[region_start:region_end].strip()
            if region_text:
                chunks.append(TextChunk(
                    text=region_text,
                    start_idx=region_start,
                    end_idx=region_end,
                    chunk_type=region_type
                ))

            current_pos = region_end

        # Process remaining text after last protected region
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                remaining_chunks = super().chunk_text(remaining)
                for chunk in remaining_chunks:
                    chunk.start_idx += current_pos
                    chunk.end_idx += current_pos
                chunks.extend(remaining_chunks)

        return chunks

    def _find_protected_regions(
        self,
        text: str
    ) -> List[Tuple[int, int, str]]:
        """Find regions that should not be split."""
        regions = []

        # Find tables
        for match in self.table_pattern.finditer(text):
            regions.append((match.start(), match.end(), "table"))

        # Find code blocks
        for match in self.code_pattern.finditer(text):
            regions.append((match.start(), match.end(), "code"))

        # Sort by start position and merge overlapping
        regions.sort(key=lambda x: x[0])
        merged = []

        for region in regions:
            if merged and region[0] <= merged[-1][1]:
                # Overlapping, extend the previous region
                merged[-1] = (
                    merged[-1][0],
                    max(merged[-1][1], region[1]),
                    merged[-1][2]
                )
            else:
                merged.append(region)

        return merged


def combine_chunks_to_text(chunks: List[TextChunk]) -> str:
    """Combine processed chunks back into a single text."""
    if not chunks:
        return ""

    # Sort by start index
    sorted_chunks = sorted(chunks, key=lambda c: c.start_idx)

    result = []
    for chunk in sorted_chunks:
        result.append(chunk.text)

    return "\n\n".join(result)
