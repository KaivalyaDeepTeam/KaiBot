"""
Model Selector Widget - Beautiful dropdown for selecting AI models.

Displays available models with recommendations, descriptions, and
download status. Handles automatic downloads with progress tracking.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDialog, QProgressBar, QFrame, QMessageBox,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPainter, QColor

from src.model_registry import (
    get_all_models, get_model_by_id, get_recommended_model, ModelInfo
)
from src.model_downloader import (
    get_model_manager, ModelManager, DownloadProgress
)


class ModelItemDelegate(QStyledItemDelegate):
    """Custom delegate for rendering model items in dropdown."""

    def __init__(self, model_manager: ModelManager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        model_id = index.data(Qt.ItemDataRole.UserRole)
        if not model_id:
            super().paint(painter, option, index)
            return

        model = get_model_by_id(model_id)
        if not model:
            super().paint(painter, option, index)
            return

        painter.save()

        # Background
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor("#E0E7FF"))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(option.rect, QColor("#F3F4F6"))

        # Icon/Status
        x = option.rect.x() + 12
        y = option.rect.y() + 8

        is_installed = self.model_manager.is_model_installed(model_id)

        # Star for recommended
        if model.recommended:
            painter.setPen(QColor("#F59E0B"))
            painter.setFont(QFont("", 14))
            painter.drawText(x, y + 14, "★")
            x += 20

        # Installed checkmark or download arrow
        if is_installed:
            painter.setPen(QColor("#10B981"))
            painter.setFont(QFont("", 12))
            painter.drawText(x, y + 14, "✓")
        else:
            painter.setPen(QColor("#6B7280"))
            painter.setFont(QFont("", 10))
            painter.drawText(x, y + 14, "↓")
        x += 20

        # Model name
        painter.setPen(QColor("#1F2937"))
        painter.setFont(QFont("", 13, QFont.Weight.Medium))
        painter.drawText(x, y + 14, model.name)

        # Description line
        painter.setPen(QColor("#6B7280"))
        painter.setFont(QFont("", 11))
        desc_y = y + 32
        painter.drawText(x, desc_y, f"Best for: {model.best_for}")

        # Size and speed
        painter.setPen(QColor("#9CA3AF"))
        painter.setFont(QFont("", 10))
        info_y = y + 48
        status = "Ready" if is_installed else f"Download {model.size_display}"
        painter.drawText(x, info_y, f"{model.size_display} | {model.speed} | {status}")

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        return QSize(350, 65)


class DownloadDialog(QDialog):
    """Dialog for downloading a model with progress."""

    def __init__(self, model: ModelInfo, parent=None):
        super().__init__(parent)
        self.model = model
        self.worker = None

        self.setWindowTitle(f"Download {model.name}")
        self.setFixedSize(450, 250)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel(f"Download {self.model.name}")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Description
        desc = QLabel(self.model.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6B7280;")
        layout.addWidget(desc)

        # Best for
        best_for = QLabel(f"Best for: {self.model.best_for}")
        best_for.setStyleSheet("color: #4B5563; font-style: italic;")
        layout.addWidget(best_for)

        # Size info
        size_label = QLabel(f"Size: {self.model.size_display}")
        size_label.setStyleSheet("color: #6B7280;")
        layout.addWidget(size_label)

        layout.addStretch()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #6B7280;")
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        btn_layout.addWidget(self.cancel_btn)

        self.download_btn = QPushButton("Download")
        self.download_btn.setObjectName("primaryButton")
        self.download_btn.clicked.connect(self._start_download)
        btn_layout.addWidget(self.download_btn)

        layout.addLayout(btn_layout)

    def _start_download(self):
        self.download_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.show()
        self.progress_bar.setValue(0)

        manager = get_model_manager()
        self.worker = manager.create_download_worker(self.model.id)

        if self.worker:
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.worker.start()

    def _on_progress(self, progress: DownloadProgress):
        self.progress_bar.setValue(int(progress.percent))

        # Format status
        downloaded_mb = progress.downloaded_mb
        total_mb = progress.total_mb
        speed = progress.speed_mbps

        if progress.eta > 60:
            eta_str = f"{progress.eta // 60}m {progress.eta % 60}s"
        else:
            eta_str = f"{progress.eta}s"

        self.status_label.setText(
            f"{downloaded_mb:.1f} / {total_mb:.1f} MB | "
            f"{speed:.1f} MB/s | ETA: {eta_str}"
        )

    def _on_finished(self, success: bool, message: str):
        if success:
            self.accept()
        else:
            self.download_btn.setEnabled(True)
            self.progress_bar.hide()
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #EF4444;")

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self.reject()


class ModelSelector(QWidget):
    """Widget for selecting AI models."""

    model_changed = pyqtSignal(str, str)  # model_id, model_path
    model_loading = pyqtSignal(str)  # status message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_manager = get_model_manager()
        self.current_model_id: Optional[str] = None

        self._setup_ui()
        self._populate_models()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Model dropdown
        self.combo = QComboBox()
        self.combo.setMinimumWidth(220)
        self.combo.setMinimumHeight(36)
        self.combo.setItemDelegate(ModelItemDelegate(self.model_manager, self))
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self.combo)

    def _populate_models(self):
        self.combo.clear()

        for model in get_all_models():
            # Display text
            is_installed = self.model_manager.is_model_installed(model.id)
            prefix = "★ " if model.recommended else ""
            suffix = "" if is_installed else f" ({model.size_display})"
            display = f"{prefix}{model.name}{suffix}"

            self.combo.addItem(display, model.id)

        # Set view to use custom height
        self.combo.view().setMinimumWidth(380)

    def _on_selection_changed(self, index: int):
        if index < 0:
            return

        model_id = self.combo.itemData(index)
        if not model_id:
            return

        model = get_model_by_id(model_id)
        if not model:
            return

        # Check if model is installed
        if self.model_manager.is_model_installed(model_id):
            self._load_model(model_id)
        else:
            # Show download dialog
            dialog = DownloadDialog(model, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._populate_models()  # Refresh list
                self._load_model(model_id)
            else:
                # Revert selection if cancelled
                if self.current_model_id:
                    self._set_model_selection(self.current_model_id)

    def _load_model(self, model_id: str):
        model_path = self.model_manager.get_model_path(model_id)
        if model_path:
            self.current_model_id = model_id
            self.model_changed.emit(model_id, str(model_path))

    def _set_model_selection(self, model_id: str):
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == model_id:
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(i)
                self.combo.blockSignals(False)
                break

    def set_current_model(self, model_id: str):
        """Set the current model selection."""
        self._set_model_selection(model_id)
        self.current_model_id = model_id

    def get_current_model_id(self) -> Optional[str]:
        """Get the currently selected model ID."""
        return self.current_model_id

    def refresh(self):
        """Refresh the model list."""
        current = self.current_model_id
        self._populate_models()
        if current:
            self._set_model_selection(current)

    def try_auto_load(self, preferred_model_id: Optional[str] = None):
        """Try to auto-load a model (previously selected or first installed)."""
        # Try preferred model first
        if preferred_model_id and self.model_manager.is_model_installed(preferred_model_id):
            self._set_model_selection(preferred_model_id)
            self._load_model(preferred_model_id)
            return True

        # Try to find any installed model
        for model in get_all_models():
            if self.model_manager.is_model_installed(model.id):
                self._set_model_selection(model.id)
                self._load_model(model.id)
                return True

        return False
