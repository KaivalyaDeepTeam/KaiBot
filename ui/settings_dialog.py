"""
Settings Dialog - Configuration dialog for model and paraphrasing settings.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QGroupBox, QTextEdit, QTabWidget,
    QWidget, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt

from src.paraphraser import ParaphraserConfig, SYSTEM_PROMPTS


class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""

    def __init__(self, config: ParaphraserConfig, parent=None):
        super().__init__(parent)
        self.config = config

        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 450)

        self._setup_ui()
        self._load_values()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Tabs
        tabs = QTabWidget()

        # Model tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)

        model_group = QGroupBox("Model Settings")
        model_form = QFormLayout(model_group)

        # Model path
        path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to GGUF model file")
        path_layout.addWidget(self.model_path_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(browse_btn)

        model_form.addRow("Model Path:", path_layout)

        # Context window
        self.n_ctx_spin = QSpinBox()
        self.n_ctx_spin.setRange(512, 32768)
        self.n_ctx_spin.setSingleStep(512)
        self.n_ctx_spin.setToolTip("Maximum context window size")
        model_form.addRow("Context Size:", self.n_ctx_spin)

        # Threads
        self.n_threads_spin = QSpinBox()
        self.n_threads_spin.setRange(1, 32)
        self.n_threads_spin.setToolTip("Number of CPU threads to use")
        model_form.addRow("CPU Threads:", self.n_threads_spin)

        # GPU layers
        self.n_gpu_spin = QSpinBox()
        self.n_gpu_spin.setRange(-1, 100)
        self.n_gpu_spin.setToolTip("-1 = auto (use all GPU layers), 0 = CPU only")
        model_form.addRow("GPU Layers:", self.n_gpu_spin)

        model_layout.addWidget(model_group)
        model_layout.addStretch()

        tabs.addTab(model_tab, "Model")

        # Generation tab
        gen_tab = QWidget()
        gen_layout = QVBoxLayout(gen_tab)

        gen_group = QGroupBox("Generation Settings")
        gen_form = QFormLayout(gen_group)

        # Temperature
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setToolTip("Higher = more creative, Lower = more consistent")
        gen_form.addRow("Temperature:", self.temp_spin)

        # Top P
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.1, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setToolTip("Nucleus sampling threshold")
        gen_form.addRow("Top P:", self.top_p_spin)

        # Top K
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)
        self.top_k_spin.setToolTip("Number of top tokens to consider")
        gen_form.addRow("Top K:", self.top_k_spin)

        # Repeat penalty
        self.repeat_spin = QDoubleSpinBox()
        self.repeat_spin.setRange(1.0, 2.0)
        self.repeat_spin.setSingleStep(0.05)
        self.repeat_spin.setDecimals(2)
        self.repeat_spin.setToolTip("Penalty for repeating tokens")
        gen_form.addRow("Repeat Penalty:", self.repeat_spin)

        # Max tokens
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(128, 4096)
        self.max_tokens_spin.setSingleStep(128)
        self.max_tokens_spin.setToolTip("Maximum tokens to generate per response")
        gen_form.addRow("Max Tokens:", self.max_tokens_spin)

        gen_layout.addWidget(gen_group)
        gen_layout.addStretch()

        tabs.addTab(gen_tab, "Generation")

        # Prompts tab
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)

        prompt_group = QGroupBox("System Prompts")
        prompt_inner = QVBoxLayout(prompt_group)

        self.prompt_combo = QComboBox()
        self.prompt_combo.addItems(list(SYSTEM_PROMPTS.keys()))
        self.prompt_combo.currentTextChanged.connect(self._on_prompt_selected)
        prompt_inner.addWidget(self.prompt_combo)

        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setMinimumHeight(200)
        prompt_inner.addWidget(self.prompt_text)

        note_label = QLabel("Note: Custom prompts coming in future version")
        note_label.setStyleSheet("color: #666; font-style: italic;")
        prompt_inner.addWidget(note_label)

        prompt_layout.addWidget(prompt_group)

        tabs.addTab(prompt_tab, "Prompts")

        layout.addWidget(tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save_and_close)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _load_values(self):
        """Load current config values into the UI."""
        self.model_path_edit.setText(self.config.model_path)
        self.n_ctx_spin.setValue(self.config.n_ctx)
        self.n_threads_spin.setValue(self.config.n_threads)
        self.n_gpu_spin.setValue(self.config.n_gpu_layers)

        self.temp_spin.setValue(self.config.temperature)
        self.top_p_spin.setValue(self.config.top_p)
        self.top_k_spin.setValue(self.config.top_k)
        self.repeat_spin.setValue(self.config.repeat_penalty)
        self.max_tokens_spin.setValue(self.config.max_tokens)

        # Load first prompt
        self._on_prompt_selected(self.prompt_combo.currentText())

    def _browse_model(self):
        """Open file dialog to select model."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model",
            "",
            "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _on_prompt_selected(self, style: str):
        """Show the selected prompt."""
        prompt = SYSTEM_PROMPTS.get(style, "")
        self.prompt_text.setPlainText(prompt)

    def _reset_defaults(self):
        """Reset to default values."""
        default = ParaphraserConfig()

        self.n_ctx_spin.setValue(default.n_ctx)
        self.n_threads_spin.setValue(default.n_threads)
        self.n_gpu_spin.setValue(default.n_gpu_layers)

        self.temp_spin.setValue(default.temperature)
        self.top_p_spin.setValue(default.top_p)
        self.top_k_spin.setValue(default.top_k)
        self.repeat_spin.setValue(default.repeat_penalty)
        self.max_tokens_spin.setValue(default.max_tokens)

    def _save_and_close(self):
        """Save settings and close dialog."""
        self.config.model_path = self.model_path_edit.text()
        self.config.n_ctx = self.n_ctx_spin.value()
        self.config.n_threads = self.n_threads_spin.value()
        self.config.n_gpu_layers = self.n_gpu_spin.value()

        self.config.temperature = self.temp_spin.value()
        self.config.top_p = self.top_p_spin.value()
        self.config.top_k = self.top_k_spin.value()
        self.config.repeat_penalty = self.repeat_spin.value()
        self.config.max_tokens = self.max_tokens_spin.value()

        self.accept()
