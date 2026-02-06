"""
PDF Processor - Handles PDF reading, text extraction, and reconstruction.
Uses PyMuPDF (fitz) for precise text replacement to preserve layout.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import re
import copy


@dataclass
class TextSpan:
    """Represents a single text span with exact position and styling."""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_name: str
    font_size: float
    font_color: Tuple[float, float, float]
    flags: int = 0  # Font flags (bold, italic, etc.)
    origin: Tuple[float, float] = (0, 0)  # Text origin point


@dataclass
class TextLine:
    """Represents a line of text containing multiple spans."""
    spans: List[TextSpan] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)


@dataclass
class TextBlock:
    """Represents a text block extracted from PDF with position info."""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_num: int
    lines: List[TextLine] = field(default_factory=list)
    font_name: str = "helv"
    font_size: float = 11.0
    font_color: Tuple[float, float, float] = (0, 0, 0)
    block_no: int = 0
    paraphrased_text: Optional[str] = None


@dataclass
class PageData:
    """Contains all extracted data from a single PDF page."""
    page_num: int
    width: float
    height: float
    text_blocks: List[TextBlock] = field(default_factory=list)


class PDFProcessor:
    """
    Extracts text from PDFs while preserving structure,
    and reconstructs PDFs with paraphrased text maintaining exact layout.
    """

    def __init__(self):
        self.doc: Optional[fitz.Document] = None
        self.pages_data: List[PageData] = []
        self.original_path: str = ""

    def load_pdf(self, path: str) -> bool:
        """Load a PDF file for processing."""
        try:
            self.doc = fitz.open(path)
            self.original_path = path
            self.pages_data = []
            return True
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False

    def extract_text_blocks(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TextBlock]:
        """
        Extract all text blocks from the PDF with detailed position and style info.
        """
        if not self.doc:
            return []

        all_blocks = []
        total_pages = len(self.doc)

        for page_num in range(total_pages):
            page = self.doc[page_num]
            page_data = PageData(
                page_num=page_num,
                width=page.rect.width,
                height=page.rect.height
            )

            # Get detailed text structure
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block_idx, block in enumerate(blocks):
                if block["type"] == 0:  # Text block
                    lines_data = []
                    block_text = ""
                    primary_font = "helv"
                    primary_size = 11.0
                    primary_color = (0, 0, 0)

                    for line in block.get("lines", []):
                        line_spans = []
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            if span_text:
                                color_int = span.get("color", 0)
                                text_span = TextSpan(
                                    text=span_text,
                                    bbox=tuple(span.get("bbox", [0,0,0,0])),
                                    font_name=span.get("font", "helv"),
                                    font_size=span.get("size", 11.0),
                                    font_color=(
                                        ((color_int >> 16) & 255) / 255.0,
                                        ((color_int >> 8) & 255) / 255.0,
                                        (color_int & 255) / 255.0
                                    ),
                                    flags=span.get("flags", 0),
                                    origin=tuple(span.get("origin", [0, 0]))
                                )
                                line_spans.append(text_span)

                                # Track primary font (first span with substantial text)
                                if len(span_text.strip()) > 3 and primary_font == "helv":
                                    primary_font = text_span.font_name
                                    primary_size = text_span.font_size
                                    primary_color = text_span.font_color

                        if line_spans:
                            text_line = TextLine(
                                spans=line_spans,
                                bbox=tuple(line.get("bbox", [0,0,0,0]))
                            )
                            lines_data.append(text_line)
                            block_text += text_line.text + "\n"

                    block_text = block_text.strip()

                    if block_text:
                        text_block = TextBlock(
                            text=block_text,
                            bbox=tuple(block["bbox"]),
                            page_num=page_num,
                            lines=lines_data,
                            font_name=primary_font,
                            font_size=primary_size,
                            font_color=primary_color,
                            block_no=block_idx
                        )
                        page_data.text_blocks.append(text_block)
                        all_blocks.append(text_block)

            self.pages_data.append(page_data)

            if progress_callback:
                progress_callback(page_num + 1, total_pages)

        return all_blocks

    def get_processable_text(self, min_words: int = 3) -> List[TextBlock]:
        """Get text blocks that should be paraphrased."""
        all_blocks = []
        for page_data in self.pages_data:
            for block in page_data.text_blocks:
                word_count = len(block.text.split())
                if word_count >= min_words:
                    all_blocks.append(block)
        return all_blocks

    def generate_paraphrased_pdf(
        self,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Generate a new PDF with paraphrased text while preserving exact layout.
        Uses line-by-line replacement to maintain structure.
        """
        if not self.doc:
            return False

        try:
            # Open a fresh copy of the original
            output_doc = fitz.open(self.original_path)

            total_blocks = sum(len(pd.text_blocks) for pd in self.pages_data)
            processed = 0

            for page_data in self.pages_data:
                page = output_doc[page_data.page_num]

                # Collect all blocks that need replacement on this page
                blocks_to_replace = []
                for block in page_data.text_blocks:
                    if block.paraphrased_text and block.paraphrased_text.strip() != block.text.strip():
                        blocks_to_replace.append(block)

                # Process each block
                for block in blocks_to_replace:
                    self._replace_block_text(page, block)

                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_blocks)

            # Save with optimization
            output_doc.save(output_path, garbage=4, deflate=True, clean=True)
            output_doc.close()

            return True

        except Exception as e:
            print(f"Error generating PDF: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _replace_block_text(self, page: fitz.Page, block: TextBlock):
        """
        Replace text in a block while preserving layout.
        Uses redaction for removal and careful text insertion for replacement.
        """
        rect = fitz.Rect(block.bbox)

        # Split paraphrased text into lines to match original structure
        original_lines = block.text.split('\n')
        para_lines = self._split_to_match_lines(
            block.paraphrased_text,
            len(original_lines),
            block.font_size,
            rect.width
        )

        # Create redaction to remove original text
        page.add_redact_annot(rect, fill=(1, 1, 1))  # White background
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # Get font info
        fontname = self._get_base_font(block.font_name)

        # Insert replacement text line by line
        if block.lines and len(block.lines) > 0:
            # Use original line positions for accurate placement
            for i, (orig_line, new_text) in enumerate(zip(block.lines, para_lines)):
                if not new_text.strip():
                    continue

                line_rect = fitz.Rect(orig_line.bbox)

                # Get styling from first span of original line
                if orig_line.spans:
                    first_span = orig_line.spans[0]
                    font_size = first_span.font_size
                    color = first_span.font_color
                else:
                    font_size = block.font_size
                    color = block.font_color

                # Adjust font size if text is longer
                adjusted_size = self._adjust_font_for_width(
                    new_text, line_rect.width, font_size, fontname
                )

                # Insert at the original line's vertical position
                text_point = fitz.Point(line_rect.x0, line_rect.y1 - 2)

                page.insert_text(
                    text_point,
                    new_text.strip(),
                    fontsize=adjusted_size,
                    fontname=fontname,
                    color=color
                )
        else:
            # Fallback: insert as textbox if no line info
            page.insert_textbox(
                rect,
                block.paraphrased_text,
                fontsize=block.font_size * 0.9,
                fontname=fontname,
                color=block.font_color,
                align=fitz.TEXT_ALIGN_LEFT
            )

    def _split_to_match_lines(
        self,
        text: str,
        num_lines: int,
        font_size: float,
        max_width: float
    ) -> List[str]:
        """Split paraphrased text to approximately match original line count."""
        words = text.split()
        if not words:
            return [""] * num_lines

        if num_lines <= 1:
            return [text]

        # Estimate chars per line based on width and font size
        chars_per_line = int(max_width / (font_size * 0.5))
        chars_per_line = max(chars_per_line, 20)

        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > chars_per_line and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_len

        if current_line:
            lines.append(" ".join(current_line))

        # Adjust to match original line count approximately
        while len(lines) < num_lines:
            lines.append("")

        # If we have more lines than original, merge some
        while len(lines) > num_lines and num_lines > 0:
            # Merge shortest adjacent lines
            min_combined = float('inf')
            merge_idx = 0
            for i in range(len(lines) - 1):
                combined = len(lines[i]) + len(lines[i+1])
                if combined < min_combined:
                    min_combined = combined
                    merge_idx = i
            lines[merge_idx] = lines[merge_idx] + " " + lines[merge_idx + 1]
            lines.pop(merge_idx + 1)

        return lines[:num_lines] if num_lines > 0 else lines

    def _adjust_font_for_width(
        self,
        text: str,
        max_width: float,
        original_size: float,
        fontname: str
    ) -> float:
        """Reduce font size if text is too wide for the line."""
        # Estimate text width (approximate)
        estimated_width = len(text) * original_size * 0.5

        if estimated_width <= max_width:
            return original_size

        # Scale down proportionally
        scale = max_width / estimated_width
        new_size = original_size * scale

        # Don't go below minimum readable size
        return max(new_size, 6.0)

    def _get_base_font(self, original_font: str) -> str:
        """Map to a standard PDF base font."""
        lower = original_font.lower()

        if "bold" in lower and "italic" in lower:
            return "hebo"  # Helvetica Bold (no bold-italic in base14)
        elif "bold" in lower:
            return "hebo"  # Helvetica Bold
        elif "italic" in lower or "oblique" in lower:
            return "heit"  # Helvetica Italic
        elif "times" in lower:
            if "bold" in lower:
                return "tibo"
            elif "italic" in lower:
                return "tiit"
            return "tiro"  # Times Roman
        elif "courier" in lower or "mono" in lower:
            return "cour"  # Courier
        else:
            return "helv"  # Helvetica (default)

    def get_page_count(self) -> int:
        """Return the number of pages in the loaded PDF."""
        return len(self.doc) if self.doc else 0

    def get_total_text_blocks(self) -> int:
        """Return the total number of text blocks across all pages."""
        return sum(len(pd.text_blocks) for pd in self.pages_data)

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
            self.doc = None


def extract_text_for_preview(pdf_path: str, max_chars: int = 2000) -> str:
    """Quick extraction of text for preview purposes."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        return f"Error reading PDF: {e}"
