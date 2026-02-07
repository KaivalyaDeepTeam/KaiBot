"""
Batch Processing Dialog - Process multiple text files at once.
Provides queue management and progress tracking for batch operations.
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QProgressBar, QComboBox,
    QFileDialog, QFrame, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class BatchItemStatus(Enum):
    """Status of a batch item."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """An item in the batch queue."""
    file_path: str
    status: BatchItemStatus = BatchItemStatus.PENDING
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    input_text: Optional[str] = None
    output_text: Optional[str] = None

    @property
    def filename(self) -> str:
        return Path(self.file_path).name


class BatchWorker(QThread):
    """Worker thread for batch processing."""
    progress = pyqtSignal(int, int, str)  # current, total, message
    item_started = pyqtSignal(int)  # item index
    item_finished = pyqtSignal(int, bool, str)  # index, success, message
    all_finished = pyqtSignal(int, int)  # completed, failed

    def __init__(self, humanizer, items: List[BatchItem], mode: str,
                 creativity: int, output_dir: str):
        super().__init__()
        self.humanizer = humanizer
        self.items = items
        self.mode = mode
        self.creativity = creativity
        self.output_dir = output_dir
        self._cancelled = False

    def run(self):
        completed = 0
        failed = 0
        total = len(self.items)

        for i, item in enumerate(self.items):
            if self._cancelled:
                item.status = BatchItemStatus.CANCELLED
                continue

            self.item_started.emit(i)
            item.status = BatchItemStatus.PROCESSING
            self.progress.emit(i + 1, total, f"Processing: {item.filename}")

            try:
                # Read input file
                with open(item.file_path, 'r', encoding='utf-8') as f:
                    item.input_text = f.read()

                if not item.input_text.strip():
                    raise ValueError("Empty file")

                # Humanize
                self.humanizer.creativity_level = self.creativity
                item.output_text = self.humanizer.humanize(
                    item.input_text,
                    progress_callback=lambda msg: self.progress.emit(i + 1, total, msg)
                )

                # Save output
                output_filename = Path(item.file_path).stem + "_humanized.txt"
                item.output_path = str(Path(self.output_dir) / output_filename)

                with open(item.output_path, 'w', encoding='utf-8') as f:
                    f.write(item.output_text)

                item.status = BatchItemStatus.COMPLETED
                completed += 1
                self.item_finished.emit(i, True, "Success")

            except Exception as e:
                item.status = BatchItemStatus.FAILED
                item.error_message = str(e)
                failed += 1
                self.item_finished.emit(i, False, str(e))

        self.all_finished.emit(completed, failed)

    def cancel(self):
        self._cancelled = True


class BatchDialog(QDialog):
    """Dialog for batch processing multiple files."""

    def __init__(self, humanizer, parent=None):
        super().__init__(parent)
        self.humanizer = humanizer
        self.items: List[BatchItem] = []
        self.worker: Optional[BatchWorker] = None
        self.output_dir = ""

        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(700, 500)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Process Multiple Files")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        desc = QLabel("Add text files to the queue and process them all at once.")
        desc.setStyleSheet("color: #6B7280;")
        layout.addWidget(desc)

        # File list
        list_group = QGroupBox("File Queue")
        list_layout = QVBoxLayout(list_group)

        # Toolbar
        toolbar = QHBoxLayout()

        add_btn = QPushButton("Add Files")
        add_btn.clicked.connect(self._add_files)
        toolbar.addWidget(add_btn)

        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self._add_folder)
        toolbar.addWidget(add_folder_btn)

        toolbar.addStretch()

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        toolbar.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        toolbar.addWidget(clear_btn)

        list_layout.addLayout(toolbar)

        # List widget
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        list_layout.addWidget(self.file_list)

        layout.addWidget(list_group)

        # Settings
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "humanize", "default", "fluency", "formal", "academic",
            "casual", "creative", "simple", "expand", "shorten", "technical"
        ])
        settings_layout.addWidget(self.mode_combo)

        settings_layout.addWidget(QLabel("Creativity:"))
        self.creativity_combo = QComboBox()
        self.creativity_combo.addItems(["Low (25)", "Medium (50)", "High (75)", "Max (100)"])
        self.creativity_combo.setCurrentIndex(1)
        settings_layout.addWidget(self.creativity_combo)

        settings_layout.addStretch()

        output_btn = QPushButton("Output Folder...")
        output_btn.clicked.connect(self._select_output)
        settings_layout.addWidget(output_btn)

        self.output_label = QLabel("Same as input")
        self.output_label.setStyleSheet("color: #6B7280;")
        settings_layout.addWidget(self.output_label)

        layout.addLayout(settings_layout)

        # Progress
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(progress_frame)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        self.process_btn = QPushButton("Process All")
        self.process_btn.setObjectName("primaryButton")
        self.process_btn.clicked.connect(self._start_processing)
        self.process_btn.setEnabled(False)
        btn_layout.addWidget(self.process_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _add_files(self):
        """Add text files to queue."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Text Files",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        for file_path in files:
            if not any(item.file_path == file_path for item in self.items):
                item = BatchItem(file_path=file_path)
                self.items.append(item)
                list_item = QListWidgetItem(f"  {item.filename}")
                self.file_list.addItem(list_item)

        self._update_buttons()

    def _add_folder(self):
        """Add all text files from folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            folder_path = Path(folder)
            for file_path in folder_path.glob("*.txt"):
                if not any(item.file_path == str(file_path) for item in self.items):
                    item = BatchItem(file_path=str(file_path))
                    self.items.append(item)
                    list_item = QListWidgetItem(f"  {item.filename}")
                    self.file_list.addItem(list_item)

        self._update_buttons()

    def _remove_selected(self):
        """Remove selected items from queue."""
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            if row < len(self.items):
                del self.items[row]

        self._update_buttons()

    def _clear_all(self):
        """Clear all items from queue."""
        self.file_list.clear()
        self.items.clear()
        self._update_buttons()

    def _select_output(self):
        """Select output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            self.output_label.setText(Path(folder).name)

    def _update_buttons(self):
        """Update button states."""
        has_items = len(self.items) > 0
        self.process_btn.setEnabled(has_items and self.humanizer.is_loaded)

    def _get_creativity(self) -> int:
        """Get creativity level from combo."""
        index = self.creativity_combo.currentIndex()
        return [25, 50, 75, 100][index]

    def _start_processing(self):
        """Start batch processing."""
        if not self.items:
            return

        # Determine output directory
        if not self.output_dir:
            self.output_dir = str(Path(self.items[0].file_path).parent)

        # Reset items
        for item in self.items:
            item.status = BatchItemStatus.PENDING
            item.output_text = None
            item.error_message = None

        # Update UI
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setMaximum(len(self.items))
        self.progress_bar.setValue(0)

        # Start worker
        self.worker = BatchWorker(
            self.humanizer,
            self.items,
            self.mode_combo.currentText(),
            self._get_creativity(),
            self.output_dir
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.item_started.connect(self._on_item_started)
        self.worker.item_finished.connect(self._on_item_finished)
        self.worker.all_finished.connect(self._on_all_finished)
        self.worker.start()

    def _cancel(self):
        """Cancel processing."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)

        self.cancel_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        self.progress_label.setText("Cancelled")

    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)

    def _on_item_started(self, index: int):
        """Handle item processing started."""
        if index < self.file_list.count():
            item = self.file_list.item(index)
            item.setText(f"  {self.items[index].filename}")

    def _on_item_finished(self, index: int, success: bool, message: str):
        """Handle item processing finished."""
        if index < self.file_list.count():
            item = self.file_list.item(index)
            if success:
                item.setText(f"  {self.items[index].filename}")
            else:
                item.setText(f"  {self.items[index].filename}")

    def _on_all_finished(self, completed: int, failed: int):
        """Handle all items finished."""
        self.cancel_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        self.progress_label.setText(f"Complete: {completed} succeeded, {failed} failed")

        if failed == 0:
            QMessageBox.information(
                self, "Batch Complete",
                f"Successfully processed {completed} files.\n"
                f"Output saved to: {self.output_dir}"
            )
        else:
            QMessageBox.warning(
                self, "Batch Complete",
                f"Processed {completed} files, {failed} failed.\n"
                f"Check the queue for error details."
            )

    def closeEvent(self, event):
        """Handle dialog close."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Processing Active",
                "Batch processing is still running. Cancel and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._cancel()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def show_batch_dialog(humanizer, parent=None) -> None:
    """Show batch processing dialog."""
    dialog = BatchDialog(humanizer, parent)
    dialog.exec()
