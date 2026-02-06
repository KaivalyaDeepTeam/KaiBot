# KaiBot - AI PDF Paraphraser

A macOS desktop application that paraphrases PDF content using local LLMs to produce human-like text that bypasses AI detection tools like Copyleaks and ZeroGPT.

## Features

- **Local LLM Processing** - Uses Mistral 7B via llama.cpp (no cloud API needed)
- **Layout Preservation** - Maintains original PDF structure, images, tables, and formatting
- **Multiple Styles** - Default, Academic, Casual, and Technical paraphrasing modes
- **GPU Acceleration** - Metal support on Apple Silicon Macs
- **Drag & Drop** - Simple interface for processing PDFs

## Requirements

- macOS 11.0+
- Python 3.9+
- ~8GB RAM (for Mistral 7B model)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KaivalyaDeepTeam/KaiBot.git
cd KaiBot
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download Mistral 7B model:
```bash
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q5_K_M.gguf --local-dir models
```

## Usage

1. Run the app:
```bash
source venv/bin/activate
python main.py
```

2. Click "Load Model" and select the downloaded `.gguf` file from `models/`
3. Drag & drop a PDF or click to browse
4. Select paraphrasing style
5. Click "Process PDF" and choose output location

## Project Structure

```
KaiBot/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── models/                 # GGUF model files
├── src/
│   ├── pdf_processor.py    # PDF extraction & reconstruction
│   ├── paraphraser.py      # LLM integration
│   ├── text_chunker.py     # Smart text splitting
│   └── workers.py          # Background threading
└── ui/
    ├── main_window.py      # Main PyQt6 interface
    └── settings_dialog.py  # Settings configuration
```

## Paraphrasing Styles

| Style | Use Case |
|-------|----------|
| **Default** | General purpose, balanced output |
| **Academic** | Research papers, formal documents |
| **Casual** | Blog posts, informal content |
| **Technical** | Documentation, technical writing |

## License

MIT License
