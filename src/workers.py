"""
Workers - Background thread workers for non-blocking UI operations.
Handles PDF processing and LLM operations in separate threads.
"""

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from typing import Optional, List
import traceback

from .pdf_processor import PDFProcessor, TextBlock
from .paraphraser import Paraphraser, ParaphraserConfig, ParaphraserManager


class WorkerSignals(QObject):
    """Signals for worker communication."""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str)  # success, message
    error = pyqtSignal(str)  # error message
    status = pyqtSignal(str)  # status message


class ModelLoaderWorker(QThread):
    """Worker for loading LLM model in background."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, paraphraser: Paraphraser, model_path: str):
        super().__init__()
        self.paraphraser = paraphraser
        self.model_path = model_path

    def run(self):
        try:
            success = self.paraphraser.load_model(
                self.model_path,
                progress_callback=self._on_progress
            )

            if success:
                self.finished.emit(True, "Model loaded successfully")
            else:
                self.finished.emit(False, "Failed to load model")

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def _on_progress(self, message: str):
        self.progress.emit(message)


class PDFExtractWorker(QThread):
    """Worker for extracting text from PDF."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str, object)  # success, message, text_blocks

    def __init__(self, pdf_processor: PDFProcessor, pdf_path: str):
        super().__init__()
        self.pdf_processor = pdf_processor
        self.pdf_path = pdf_path

    def run(self):
        try:
            # Load the PDF
            if not self.pdf_processor.load_pdf(self.pdf_path):
                self.finished.emit(False, "Failed to load PDF", None)
                return

            # Extract text blocks
            text_blocks = self.pdf_processor.extract_text_blocks(
                progress_callback=self._on_progress
            )

            if text_blocks:
                self.finished.emit(
                    True,
                    f"Extracted {len(text_blocks)} text blocks",
                    text_blocks
                )
            else:
                self.finished.emit(
                    False,
                    "No text found in PDF",
                    None
                )

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}", None)

    def _on_progress(self, current: int, total: int):
        self.progress.emit(current, total, f"Extracting page {current}/{total}")


class ParaphraseWorker(QThread):
    """Worker for paraphrasing text blocks."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        paraphraser: Paraphraser,
        text_blocks: List[TextBlock],
        style: str = "default"
    ):
        super().__init__()
        self.paraphraser = paraphraser
        self.text_blocks = text_blocks
        self.style = style
        self.manager = ParaphraserManager(paraphraser)
        self._cancelled = False

    def run(self):
        try:
            processed = self.manager.process_text_blocks(
                self.text_blocks,
                style=self.style,
                progress_callback=self._on_progress
            )

            if self._cancelled:
                self.finished.emit(False, "Cancelled by user")
            else:
                self.finished.emit(
                    True,
                    f"Paraphrased {processed} text blocks"
                )

        except Exception as e:
            tb = traceback.format_exc()
            self.finished.emit(False, f"Error: {str(e)}\n{tb}")

    def cancel(self):
        """Cancel the paraphrasing operation."""
        self._cancelled = True
        self.manager.cancel()

    def _on_progress(self, current: int, total: int, message: str):
        self.progress.emit(current, total, message)


class PDFGenerateWorker(QThread):
    """Worker for generating the output PDF."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, pdf_processor: PDFProcessor, output_path: str):
        super().__init__()
        self.pdf_processor = pdf_processor
        self.output_path = output_path

    def run(self):
        try:
            success = self.pdf_processor.generate_paraphrased_pdf(
                self.output_path,
                progress_callback=self._on_progress
            )

            if success:
                self.finished.emit(True, f"PDF saved to: {self.output_path}")
            else:
                self.finished.emit(False, "Failed to generate PDF")

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def _on_progress(self, current: int, total: int):
        self.progress.emit(current, total, f"Processing block {current}/{total}")


class FullProcessWorker(QThread):
    """
    Worker that handles the complete workflow:
    Load PDF -> Extract -> Paraphrase -> Generate Output
    """

    progress = pyqtSignal(int, int, str)
    stage_changed = pyqtSignal(str)  # Current stage name
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        paraphraser: Paraphraser,
        input_path: str,
        output_path: str,
        style: str = "default",
        min_words: int = 3
    ):
        super().__init__()
        self.paraphraser = paraphraser
        self.input_path = input_path
        self.output_path = output_path
        self.style = style
        self.min_words = min_words
        self._cancelled = False
        self.pdf_processor = PDFProcessor()

    def run(self):
        try:
            # Stage 1: Load PDF
            self.stage_changed.emit("Loading PDF")
            self.progress.emit(0, 100, "Loading PDF...")

            if not self.pdf_processor.load_pdf(self.input_path):
                self.finished.emit(False, "Failed to load PDF")
                return

            if self._cancelled:
                self._cleanup()
                return

            # Stage 2: Extract text
            self.stage_changed.emit("Extracting Text")

            text_blocks = self.pdf_processor.extract_text_blocks(
                progress_callback=lambda c, t: self.progress.emit(
                    c, t, f"Extracting page {c}/{t}"
                )
            )

            if not text_blocks:
                self.finished.emit(False, "No text found in PDF")
                return

            if self._cancelled:
                self._cleanup()
                return

            # Get processable blocks (filter short ones)
            processable = self.pdf_processor.get_processable_text(self.min_words)

            # Stage 3: Paraphrase
            self.stage_changed.emit("Paraphrasing")
            manager = ParaphraserManager(self.paraphraser)

            def paraphrase_progress(c, t, msg):
                if self._cancelled:
                    manager.cancel()
                self.progress.emit(c, t, msg)

            processed = manager.process_text_blocks(
                processable,
                style=self.style,
                progress_callback=paraphrase_progress
            )

            if self._cancelled:
                self._cleanup()
                return

            # Stage 4: Generate output PDF
            self.stage_changed.emit("Generating PDF")

            success = self.pdf_processor.generate_paraphrased_pdf(
                self.output_path,
                progress_callback=lambda c, t: self.progress.emit(
                    c, t, f"Writing block {c}/{t}"
                )
            )

            if success:
                self.finished.emit(
                    True,
                    f"Successfully processed {processed} blocks.\n"
                    f"Output saved to: {self.output_path}"
                )
            else:
                self.finished.emit(False, "Failed to generate output PDF")

        except Exception as e:
            tb = traceback.format_exc()
            self.finished.emit(False, f"Error: {str(e)}\n{tb}")

        finally:
            self._cleanup()

    def cancel(self):
        """Cancel the operation."""
        self._cancelled = True

    def _cleanup(self):
        """Clean up resources."""
        if self.pdf_processor:
            self.pdf_processor.close()
        if self._cancelled:
            self.finished.emit(False, "Operation cancelled")
