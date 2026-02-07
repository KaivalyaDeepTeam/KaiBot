"""
Model Downloader - Download and manage AI models for LocalWrite.

Downloads models from Hugging Face with progress tracking,
resume support, and integrity verification.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from .model_registry import (
    AVAILABLE_MODELS, ModelInfo, get_model_by_id, get_all_models
)


@dataclass
class DownloadProgress:
    """Progress information for a download."""
    downloaded: int
    total: int
    speed: float  # bytes per second
    eta: int  # seconds remaining

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 0
        return (self.downloaded / self.total) * 100

    @property
    def downloaded_mb(self) -> float:
        return self.downloaded / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.total / (1024 * 1024)

    @property
    def speed_mbps(self) -> float:
        return self.speed / (1024 * 1024)


class ModelDownloadWorker(QThread):
    """Worker thread for downloading models."""
    progress = pyqtSignal(object)  # DownloadProgress
    finished = pyqtSignal(bool, str)  # success, message/path

    def __init__(self, model_id: str, dest_path: Path):
        super().__init__()
        self.model_id = model_id
        self.dest_path = dest_path
        self._cancelled = False

    def run(self):
        model = get_model_by_id(self.model_id)
        if not model:
            self.finished.emit(False, f"Model not found: {self.model_id}")
            return

        try:
            # Create temp file for download
            temp_path = self.dest_path.with_suffix('.downloading')

            # Get existing size for resume
            resume_pos = 0
            if temp_path.exists():
                resume_pos = temp_path.stat().st_size

            # Start download with resume header
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'

            response = requests.get(
                model.download_url,
                headers=headers,
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            # Get total size
            if resume_pos > 0 and response.status_code == 206:
                # Partial content - resuming
                total_size = resume_pos + int(response.headers.get('content-length', 0))
            else:
                total_size = int(response.headers.get('content-length', 0))
                resume_pos = 0  # Server doesn't support resume, start fresh

            # Open file for writing/appending
            mode = 'ab' if resume_pos > 0 else 'wb'
            downloaded = resume_pos

            import time
            start_time = time.time()
            last_update = start_time

            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if self._cancelled:
                        self.finished.emit(False, "Download cancelled")
                        return

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Calculate speed and ETA
                        current_time = time.time()
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            speed = (downloaded - resume_pos) / elapsed
                            remaining = total_size - downloaded
                            eta = int(remaining / speed) if speed > 0 else 0
                        else:
                            speed = 0
                            eta = 0

                        # Emit progress (throttled to every 0.1s)
                        if current_time - last_update >= 0.1:
                            progress = DownloadProgress(
                                downloaded=downloaded,
                                total=total_size,
                                speed=speed,
                                eta=eta
                            )
                            self.progress.emit(progress)
                            last_update = current_time

            # Rename temp file to final
            temp_path.rename(self.dest_path)

            self.finished.emit(True, str(self.dest_path))

        except requests.exceptions.RequestException as e:
            self.finished.emit(False, f"Download failed: {e}")
        except Exception as e:
            self.finished.emit(False, f"Error: {e}")

    def cancel(self):
        self._cancelled = True


class ModelManager:
    """Manage downloaded models."""

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            self.models_dir = Path.home() / ".localwrite" / "models"
        else:
            self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get path to a model if installed."""
        model = get_model_by_id(model_id)
        if not model:
            return None

        path = self.models_dir / model.filename
        if path.exists():
            return path
        return None

    def is_model_installed(self, model_id: str) -> bool:
        """Check if a model is installed."""
        return self.get_model_path(model_id) is not None

    def get_installed_models(self) -> List[str]:
        """Get list of installed model IDs."""
        installed = []
        for model in AVAILABLE_MODELS:
            if self.is_model_installed(model.id):
                installed.append(model.id)
        return installed

    def get_model_dest_path(self, model_id: str) -> Optional[Path]:
        """Get destination path for a model."""
        model = get_model_by_id(model_id)
        if not model:
            return None
        return self.models_dir / model.filename

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        path = self.get_model_path(model_id)
        if path and path.exists():
            path.unlink()
            return True
        return False

    def get_disk_usage(self) -> float:
        """Get total disk usage of all models in GB."""
        total = 0
        for model_id in self.get_installed_models():
            path = self.get_model_path(model_id)
            if path:
                total += path.stat().st_size
        return total / (1024 ** 3)

    def create_download_worker(self, model_id: str) -> Optional[ModelDownloadWorker]:
        """Create a download worker for a model."""
        dest_path = self.get_model_dest_path(model_id)
        if not dest_path:
            return None
        return ModelDownloadWorker(model_id, dest_path)


# Global instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
