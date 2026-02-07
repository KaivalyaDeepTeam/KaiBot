"""
Settings Manager - Persistent JSON settings storage for LocalWrite.
Handles saving and loading user preferences, window state, and app configuration.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List


@dataclass
class AppSettings:
    """Application settings that persist between sessions."""

    # Theme
    theme: str = "light"  # "light" or "dark"

    # Window state
    window_width: int = 1200
    window_height: int = 800
    window_x: Optional[int] = None
    window_y: Optional[int] = None
    window_maximized: bool = False

    # Writing enhancement settings
    default_mode: str = "enhance"
    selected_model_id: str = ""  # ID of selected model from registry
    creativity_level: int = 50  # 0-100 slider value
    auto_check_ai: bool = True

    # Model settings
    last_model_path: str = ""
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = -1
    temperature: float = 0.85
    top_p: float = 0.92
    top_k: int = 50
    repeat_penalty: float = 1.15
    max_tokens: int = 1024

    # PDF settings
    min_words_per_block: int = 3

    # UI preferences
    show_stats_panel: bool = True
    show_ai_score: bool = True
    auto_copy_result: bool = False
    confirm_on_close: bool = True

    # Recent files
    recent_files: List[str] = field(default_factory=list)
    max_recent_files: int = 10

    # Export settings
    last_export_format: str = "txt"
    last_export_directory: str = ""

    # First run
    first_run: bool = True
    version: str = "1.0.0"


class SettingsManager:
    """Manages persistent application settings."""

    DEFAULT_CONFIG_DIR = ".localwrite"
    SETTINGS_FILE = "settings.json"

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize settings manager.

        Args:
            config_dir: Custom config directory path. Defaults to ~/.localwrite/
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / self.DEFAULT_CONFIG_DIR

        self.settings_file = self.config_dir / self.SETTINGS_FILE
        self.settings = AppSettings()

        # Ensure config directory exists
        self._ensure_config_dir()

        # Load existing settings
        self.load()

    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create config directory: {e}")

    def load(self) -> AppSettings:
        """Load settings from file."""
        if not self.settings_file.exists():
            return self.settings

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Update settings with loaded values
            for key, value in data.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

            # Mark as not first run after loading
            if data.get('first_run') is not None:
                self.settings.first_run = data['first_run']

        except Exception as e:
            print(f"Warning: Could not load settings: {e}")

        return self.settings

    def save(self) -> bool:
        """Save settings to file."""
        try:
            self._ensure_config_dir()

            data = asdict(self.settings)

            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return getattr(self.settings, key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set a setting value."""
        if hasattr(self.settings, key):
            setattr(self.settings, key, value)
            return True
        return False

    def update(self, **kwargs) -> None:
        """Update multiple settings at once."""
        for key, value in kwargs.items():
            self.set(key, value)

    def reset(self) -> None:
        """Reset all settings to defaults."""
        self.settings = AppSettings()
        self.save()

    def add_recent_file(self, file_path: str) -> None:
        """Add a file to recent files list."""
        # Remove if already exists
        if file_path in self.settings.recent_files:
            self.settings.recent_files.remove(file_path)

        # Add to beginning
        self.settings.recent_files.insert(0, file_path)

        # Limit list size
        self.settings.recent_files = self.settings.recent_files[:self.settings.max_recent_files]

    def clear_recent_files(self) -> None:
        """Clear recent files list."""
        self.settings.recent_files = []

    def get_window_geometry(self) -> Dict[str, Any]:
        """Get window geometry settings."""
        return {
            'width': self.settings.window_width,
            'height': self.settings.window_height,
            'x': self.settings.window_x,
            'y': self.settings.window_y,
            'maximized': self.settings.window_maximized
        }

    def save_window_geometry(self, width: int, height: int,
                             x: int, y: int, maximized: bool) -> None:
        """Save window geometry."""
        self.settings.window_width = width
        self.settings.window_height = height
        self.settings.window_x = x
        self.settings.window_y = y
        self.settings.window_maximized = maximized

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration settings."""
        return {
            'model_path': self.settings.last_model_path,
            'n_ctx': self.settings.n_ctx,
            'n_threads': self.settings.n_threads,
            'n_gpu_layers': self.settings.n_gpu_layers,
            'temperature': self.settings.temperature,
            'top_p': self.settings.top_p,
            'top_k': self.settings.top_k,
            'repeat_penalty': self.settings.repeat_penalty,
            'max_tokens': self.settings.max_tokens
        }

    def save_model_config(self, **kwargs) -> None:
        """Save model configuration."""
        mapping = {
            'model_path': 'last_model_path',
            'n_ctx': 'n_ctx',
            'n_threads': 'n_threads',
            'n_gpu_layers': 'n_gpu_layers',
            'temperature': 'temperature',
            'top_p': 'top_p',
            'top_k': 'top_k',
            'repeat_penalty': 'repeat_penalty',
            'max_tokens': 'max_tokens'
        }
        for key, value in kwargs.items():
            setting_key = mapping.get(key, key)
            self.set(setting_key, value)

    def mark_first_run_complete(self) -> None:
        """Mark that the first run is complete."""
        self.settings.first_run = False
        self.save()

    def is_first_run(self) -> bool:
        """Check if this is the first run."""
        return self.settings.first_run

    def export_settings(self, file_path: str) -> bool:
        """Export settings to a custom file."""
        try:
            data = asdict(self.settings)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def import_settings(self, file_path: str) -> bool:
        """Import settings from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for key, value in data.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

            self.save()
            return True
        except Exception:
            return False


# Global settings instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def get_settings() -> AppSettings:
    """Get the current settings."""
    return get_settings_manager().settings


def save_settings() -> bool:
    """Save current settings."""
    return get_settings_manager().save()
