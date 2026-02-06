#!/usr/bin/env python3
"""
KaiBot - AI-Powered PDF Paraphraser for macOS

A desktop app that paraphrases PDF content using local LLMs (Mistral 7B)
to produce human-like text that bypasses AI detection tools.

Usage:
    python main.py

Requirements:
    - PyQt6
    - PyMuPDF
    - llama-cpp-python
    - A GGUF model file (e.g., Mistral 7B)
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Fix Qt plugin path for conda environments on macOS
def fix_qt_plugin_path():
    """Set correct Qt plugin path for PyQt6 in conda."""
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
from PyQt6.QtGui import QPalette, QColor

from ui.main_window import MainWindow


def setup_style(app: QApplication):
    """Setup application styling."""
    # Use fusion style for consistent cross-platform look
    app.setStyle("Fusion")

    # Optional: Load custom stylesheet
    stylesheet_path = os.path.join(project_root, "resources", "styles.qss")
    if os.path.exists(stylesheet_path):
        with open(stylesheet_path, "r") as f:
            app.setStyleSheet(f.read())

    # Set palette for a clean look
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)


def main():
    """Main entry point."""
    # Enable high DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("KaiBot")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("KaiBot")

    # Setup styling
    setup_style(app)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
