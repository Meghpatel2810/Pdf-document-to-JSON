# EOB Document to JSON Automation

This repository contains the research and implementation of a hybrid AI pipeline that automates data extraction from medical  documents (PDFs/images) and transforms them into structured JSON for integration into an **Electronic Health Record (EHR)** system.

---

## 🚀 Features

- **PDF to Image Conversion** using Poppler
- **Image Preprocessing** (resizing, noise removal, thresholding) with Pillow / OpenCV
- **OCR (Optical Character Recognition)** using EasyOCR for text extraction
- **Large Language Model (LLM) Post-processing** with a predefined JSON schema
---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**:
  - [Poppler](https://poppler.freedesktop.org/) – PDF to image conversion
  - [Pillow](https://python-pillow.org/) – Image processing
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR) – OCR engine
  - [PyTorch](https://pytorch.org/) – Deep learning framework
  - [Qwen Hugging Face Model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) – LLM inference


---

## 📂 Directory Structure

├── App/ <br>
│ ├── # Main script: PDF → JSON <br>
├── Debug/ # Intermediate OCR outputs, debug images  <br>
├── Input/ # PDFs to process <br>
├── Outputs/ # Generated JSON files <br>
├── Structure/ # Predefined JSON schemas <br>
├── Logs/ # Python logging outputs <br>
├── requirements.txt # Python dependencies <br>
└── .gitignore <br>

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/eob-to-json.git
   cd eob-to-json

2. Create and activate a virtual environment and Install dependencies

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
# Pytorch library for specific cuda version from https://pytorch.org/