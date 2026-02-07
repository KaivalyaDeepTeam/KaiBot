"""
History Dialog - View and restore previous humanization sessions.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QTextEdit, QSplitter,
    QFrame, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.history_manager import get_history_manager, HistoryEntry


class HistoryDialog(QDialog):
    """Dialog for viewing and restoring history."""

    restore_requested = pyqtSignal(str, str)  # original, humanized

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_manager = get_history_manager()
        self.selected_entry: Optional[HistoryEntry] = None

        self.setWindowTitle("History")
        self.setMinimumSize(800, 600)
        self.setModal(True)

        self._setup_ui()
        self._load_entries()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("Session History")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Stats
        stats = self.history_manager.get_statistics()
        stats_label = QLabel(f"{stats['total_entries']} sessions | "
                            f"{stats['total_words_processed']:,} words processed")
        stats_label.setStyleSheet("color: #6B7280;")
        header_layout.addWidget(stats_label)

        layout.addLayout(header_layout)

        # Splitter for list and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # History list
        list_widget = QFrame()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        self.history_list = QListWidget()
        self.history_list.currentItemChanged.connect(self._on_selection_changed)
        self.history_list.itemDoubleClicked.connect(self._restore_entry)
        list_layout.addWidget(self.history_list)

        splitter.addWidget(list_widget)

        # Preview panel
        preview_widget = QFrame()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # Entry info
        self.info_label = QLabel("Select an entry to preview")
        self.info_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        preview_layout.addWidget(self.info_label)

        # Original text
        original_group = QGroupBox("Original")
        original_layout = QVBoxLayout(original_group)
        self.original_preview = QTextEdit()
        self.original_preview.setReadOnly(True)
        self.original_preview.setMaximumHeight(150)
        original_layout.addWidget(self.original_preview)
        preview_layout.addWidget(original_group)

        # Humanized text
        humanized_group = QGroupBox("Humanized")
        humanized_layout = QVBoxLayout(humanized_group)
        self.humanized_preview = QTextEdit()
        self.humanized_preview.setReadOnly(True)
        humanized_layout.addWidget(self.humanized_preview)
        preview_layout.addWidget(humanized_group)

        splitter.addWidget(preview_widget)
        splitter.setSizes([300, 500])

        layout.addWidget(splitter)

        # Buttons
        btn_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._delete_selected)
        btn_layout.addWidget(delete_btn)

        self.restore_btn = QPushButton("Restore")
        self.restore_btn.setObjectName("primaryButton")
        self.restore_btn.clicked.connect(self._restore_entry)
        self.restore_btn.setEnabled(False)
        btn_layout.addWidget(self.restore_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _load_entries(self):
        """Load history entries into list."""
        self.history_list.clear()
        entries = self.history_manager.get_all_entries()

        for entry in entries:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, entry.id)

            # Format display text
            time_str = entry.get_formatted_timestamp()
            preview = entry.get_preview(50)
            words = entry.word_count_original

            text = f"{time_str}\n{preview}\n{words} words | {entry.mode}"
            item.setText(text)

            # Add score indicator if available
            if entry.ai_score_after is not None:
                if entry.ai_score_after < 30:
                    item.setBackground(Qt.GlobalColor.green)
                elif entry.ai_score_after < 60:
                    item.setBackground(Qt.GlobalColor.yellow)

            self.history_list.addItem(item)

    def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle selection change."""
        if not current:
            self.selected_entry = None
            self.restore_btn.setEnabled(False)
            self.original_preview.clear()
            self.humanized_preview.clear()
            self.info_label.setText("Select an entry to preview")
            return

        entry_id = current.data(Qt.ItemDataRole.UserRole)
        self.selected_entry = self.history_manager.get_entry(entry_id)

        if self.selected_entry:
            self.restore_btn.setEnabled(True)

            # Update info
            info_parts = [
                self.selected_entry.get_formatted_timestamp(),
                f"Mode: {self.selected_entry.mode}",
                f"Creativity: {self.selected_entry.creativity_level}%"
            ]
            if self.selected_entry.ai_score_before is not None:
                info_parts.append(f"AI Score: {self.selected_entry.ai_score_before:.0f}% -> "
                                f"{self.selected_entry.ai_score_after:.0f}%")
            self.info_label.setText(" | ".join(info_parts))

            # Update previews
            self.original_preview.setPlainText(self.selected_entry.original_text)
            self.humanized_preview.setPlainText(self.selected_entry.humanized_text)

    def _restore_entry(self):
        """Restore selected entry to main window."""
        if self.selected_entry:
            self.restore_requested.emit(
                self.selected_entry.original_text,
                self.selected_entry.humanized_text
            )
            self.accept()

    def _delete_selected(self):
        """Delete selected entry."""
        if not self.selected_entry:
            return

        reply = QMessageBox.question(
            self, "Delete Entry",
            "Are you sure you want to delete this entry?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.history_manager.delete_entry(self.selected_entry.id)
            self._load_entries()
            self.selected_entry = None
            self.restore_btn.setEnabled(False)
            self.original_preview.clear()
            self.humanized_preview.clear()

    def _clear_history(self):
        """Clear all history."""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear all history?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.history_manager.clear_history()
            self._load_entries()
            self.selected_entry = None
            self.restore_btn.setEnabled(False)
            self.original_preview.clear()
            self.humanized_preview.clear()


def show_history_dialog(parent=None) -> Optional[tuple]:
    """
    Show history dialog.

    Returns:
        Tuple of (original, humanized) if restored, None if cancelled
    """
    result = [None]

    def on_restore(original, humanized):
        result[0] = (original, humanized)

    dialog = HistoryDialog(parent)
    dialog.restore_requested.connect(on_restore)
    dialog.exec()

    return result[0]
