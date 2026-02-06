"""
PDF Processor - Handles PDF reading, text extraction, and reconstruction.
Uses PyMuPDF (fitz) for in-place text replacement to preserve layout.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import re


@dataclass
class TextBlock:
    """Represents a text block extracted from PDF with position info."""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_num: int
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
    images: List[Dict] = field(default_factory=list)


class PDFProcessor:
    """
    Extracts text from PDFs while preserving structure,
    and reconstructs PDFs with paraphrased text.
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
        Extract all text blocks from the PDF with position and style info.
        Returns a flat list of all text blocks across all pages.
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

            # Extract text blocks with detailed info
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block_idx, block in enumerate(blocks):
                if block["type"] == 0:  # Text block
                    # Combine all spans in the block
                    block_text = ""
                    font_name = "helv"
                    font_size = 11.0
                    font_color = (0, 0, 0)

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                            # Use the first span's font info as representative
                            if font_name == "helv":
                                font_name = span.get("font", "helv")
                                font_size = span.get("size", 11.0)
                                color_int = span.get("color", 0)
                                # Convert integer color to RGB tuple
                                font_color = (
                                    ((color_int >> 16) & 255) / 255.0,
                                    ((color_int >> 8) & 255) / 255.0,
                                    (color_int & 255) / 255.0
                                )
                        block_text += "\n"

                    block_text = block_text.strip()

                    if block_text:  # Only add non-empty blocks
                        text_block = TextBlock(
                            text=block_text,
                            bbox=tuple(block["bbox"]),
                            page_num=page_num,
                            font_name=font_name,
                            font_size=font_size,
                            font_color=font_color,
                            block_no=block_idx
                        )
                        page_data.text_blocks.append(text_block)
                        all_blocks.append(text_block)

                elif block["type"] == 1:  # Image block
                    page_data.images.append({
                        "bbox": block["bbox"],
                        "block_no": block_idx
                    })

            self.pages_data.append(page_data)

            if progress_callback:
                progress_callback(page_num + 1, total_pages)

        return all_blocks

    def get_processable_text(self, min_words: int = 3) -> List[TextBlock]:
        """
        Get text blocks that should be paraphrased.
        Filters out very short blocks (headers, page numbers, etc.)
        """
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
        Generate a new PDF with paraphrased text while preserving layout.
        Uses PyMuPDF's redaction API for clean text replacement.
        """
        if not self.doc:
            return False

        try:
            # Create a copy of the document
            output_doc = fitz.open(self.original_path)

            total_blocks = sum(len(pd.text_blocks) for pd in self.pages_data)
            processed = 0

            for page_data in self.pages_data:
                page = output_doc[page_data.page_num]

                for block in page_data.text_blocks:
                    if block.paraphrased_text and block.paraphrased_text != block.text:
                        # Create redaction annotation for the original text area
                        rect = fitz.Rect(block.bbox)

                        # Add redaction (this marks the area for removal)
                        page.add_redact_annot(rect, fill=(1, 1, 1))  # White fill

                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_blocks)

                # Apply all redactions on this page
                page.apply_redactions()

                # Now insert the paraphrased text
                for block in page_data.text_blocks:
                    if block.paraphrased_text and block.paraphrased_text != block.text:
                        rect = fitz.Rect(block.bbox)

                        # Calculate appropriate font size to fit text in the box
                        font_size = self._calculate_fitting_font_size(
                            block.paraphrased_text,
                            rect,
                            block.font_size
                        )

                        # Insert the paraphrased text
                        # Use a standard font that's always available
                        fontname = self._get_standard_font(block.font_name)

                        page.insert_textbox(
                            rect,
                            block.paraphrased_text,
                            fontsize=font_size,
                            fontname=fontname,
                            color=block.font_color,
                            align=fitz.TEXT_ALIGN_LEFT
                        )

            # Save the output
            output_doc.save(output_path, garbage=4, deflate=True)
            output_doc.close()

            return True

        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False

    def _calculate_fitting_font_size(
        self,
        text: str,
        rect: fitz.Rect,
        original_size: float
    ) -> float:
        """
        Calculate a font size that will fit the text in the given rectangle.
        Starts with original size and reduces if necessary.
        """
        # Estimate characters per line based on rect width
        # Average character width is roughly 0.5 * font_size
        min_size = 6.0

        for size in [original_size, original_size * 0.9, original_size * 0.8,
                     original_size * 0.7, original_size * 0.6, min_size]:
            chars_per_line = rect.width / (size * 0.5)
            lines_available = rect.height / (size * 1.2)

            # Estimate lines needed
            words = text.split()
            estimated_chars = len(text)
            estimated_lines = estimated_chars / max(chars_per_line, 1) + 1

            if estimated_lines <= lines_available:
                return size

        return min_size

    def _get_standard_font(self, original_font: str) -> str:
        """Map the original font to a standard PDF font."""
        original_lower = original_font.lower()

        if "bold" in original_lower and "italic" in original_lower:
            return "hebi"  # Helvetica Bold Italic
        elif "bold" in original_lower:
            return "hebo"  # Helvetica Bold
        elif "italic" in original_lower or "oblique" in original_lower:
            return "heit"  # Helvetica Italic
        elif "times" in original_lower:
            return "tiro"  # Times Roman
        elif "courier" in original_lower or "mono" in original_lower:
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
    """
    Quick extraction of text for preview purposes.
    Returns first max_chars characters of the PDF text.
    """
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
