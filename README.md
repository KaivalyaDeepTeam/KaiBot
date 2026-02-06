# KaiBot - AI Humanizer & PDF Paraphraser

A powerful macOS desktop application that humanizes AI-generated text and paraphrases PDFs using local LLMs. Designed to bypass AI detection tools like Copyleaks and ZeroGPT.

## Features

- **Text Humanizer** - Paste text and get human-like output instantly (like Copyleaks AI Humanizer)
- **PDF Paraphraser** - Process entire PDFs while preserving layout
- **Local LLM Processing** - Uses Qwen 2.5 7B via llama.cpp (no cloud API, 100% private)
- **Multiple Styles** - Default, Academic, Casual, and Technical modes
- **GPU Acceleration** - Metal support on Apple Silicon Macs
- **Copy to Clipboard** - One-click copy for humanized text
- **Word Count Tracking** - Monitor input/output word counts
- **Drag & Drop** - Simple PDF processing interface

## Requirements

- macOS 11.0+
- Python 3.9+
- ~8GB RAM (for Qwen 2.5 7B model)

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

3. Download Qwen 2.5 7B model:
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q5_k_m.gguf --local-dir models
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
