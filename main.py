#!/usr/bin/env python3
"""
LocalWrite - Private AI Writing Assistant

A privacy-first desktop application that enhances your writing using
local AI models. 100% offline - your words never leave your device.

Features:
- 11 writing enhancement modes (Professional, Creative, Scholarly, etc.)
- Smart model selection with 5 curated AI models
- Auto-download models from Hugging Face
- Real-time writing statistics
- PDF enhancement with layout preservation
- Dark mode support
- No accounts, no tracking, no cloud

Privacy Promise:
- All AI processing happens locally on your device
- No internet required after model download
- No telemetry or analytics
- Open source (MIT License)

Usage:
    python main.py

Requirements:
    - macOS 11.0+
    - PyQt6 >= 6.5.0
    - 8GB RAM minimum
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def fix_qt_plugin_path():
    """Set correct Qt plugin path for PyQt6 in conda environments."""
    try:
        import PyQt6
        pyqt6_path = os.path.dirname(PyQt6.__file__)
        plugin_path = os.path.join(pyqt6_path, "Qt6", "plugins")
        if os.path.exists(plugin_path):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    except Exception:
        pass


fix_qt_plugin_path()

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QIcon

from ui.main_window import MainWindow
from src.settings_manager import get_settings_manager


def setup_light_palette(app: QApplication):
    """Setup light theme palette."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(249, 250, 251))  # #F9FAFB
    palette.setColor(QPalette.ColorRole.WindowText, QColor(17, 24, 39))  # #111827
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(243, 244, 246))  # #F3F4F6
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(31, 41, 55))  # Dark tooltip
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(17, 24, 39))
    palette.setColor(QPalette.ColorRole.Button, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(17, 24, 39))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(239, 68, 68))  # Red
    palette.setColor(QPalette.ColorRole.Link, QColor(79, 70, 229))  # Indigo
    palette.setColor(QPalette.ColorRole.Highlight, QColor(79, 70, 229))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def setup_dark_palette(app: QApplication):
    """Setup dark theme palette."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(17, 24, 39))  # #111827
    palette.setColor(QPalette.ColorRole.WindowText, QColor(249, 250, 251))  # #F9FAFB
    palette.setColor(QPalette.ColorRole.Base, QColor(31, 41, 55))  # #1F2937
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(55, 65, 81))  # #374151
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(55, 65, 81))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(249, 250, 251))
    palette.setColor(QPalette.ColorRole.Text, QColor(249, 250, 251))
    palette.setColor(QPalette.ColorRole.Button, QColor(31, 41, 55))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(249, 250, 251))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(248, 113, 113))  # Red-400
    palette.setColor(QPalette.ColorRole.Link, QColor(129, 140, 248))  # Indigo-400
    palette.setColor(QPalette.ColorRole.Highlight, QColor(129, 140, 248))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def setup_style(app: QApplication, dark_mode: bool = False):
    """Setup application styling based on theme preference."""
    # Use Fusion style for consistent cross-platform look
    app.setStyle("Fusion")

    # Load appropriate stylesheet
    if dark_mode:
        stylesheet_path = os.path.join(project_root, "resources", "styles_dark.qss")
        setup_dark_palette(app)
    else:
        stylesheet_path = os.path.join(project_root, "resources", "styles.qss")
        setup_light_palette(app)

    if os.path.exists(stylesheet_path):
        with open(stylesheet_path, "r") as f:
            app.setStyleSheet(f.read())


def main():
    """Main entry point."""
    # Enable high DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("LocalWrite")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Svetozar Technologies")

    # Set app icon
    icon_path = os.path.join(project_root, "resources", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Load settings
    settings_manager = get_settings_manager()
    dark_mode = settings_manager.settings.theme == "dark"

    # Setup styling based on saved preference
    setup_style(app, dark_mode)

    # Check for first run
    if settings_manager.is_first_run():
        settings_manager.mark_first_run_complete()

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
