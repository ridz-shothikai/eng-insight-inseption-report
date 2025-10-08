# 🧠 Intelligent PDF Report Generation API

This project is a FastAPI-based backend for generating intelligent PDF reports using OCR (Optical Character Recognition), category classification, and automatic merging of visual and textual content.  
It supports handling multiple file uploads, extracting structured data, and generating professional inception-style PDF reports.

---

## ⚙️ Features

- Extract text and images from uploaded PDFs using **Tesseract OCR**
- Support for **multiple languages** including **Hindi**
- Generate **combined visual + text reports** with correct sequence placement
- Modular architecture with utilities for file handling, validation, and coordinate processing
- FastAPI backend with CORS and async support

---

## 🧩 Project Structure

```
src/
│
├── main.py                   # FastAPI entry point
├── pdf_merger.py             # Merges inception.pdf and image pages
│
├── ocr/
│   ├── __init__.py
│   ├── text_extractor.py     # Handles OCR text extraction
│   ├── image_extractor.py    # Handles image extraction
│
├── utils/
│   ├── __init__.py
│   ├── file_handler.py       # File read/write operations
│   ├── coordinate_processor.py # Lat/Long data processor
│   ├── validators.py         # Input validators
│
└── requirements.txt          # Dependencies
```

---

## 🚀 How to Run the Server

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/intelligent-pdf-report-gen.git
cd intelligent-pdf-report-gen
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

Then open your browser at:  
👉 **http://localhost:8001/docs** to test via Swagger UI.

---

## 🧠 Setting up Tesseract OCR

### 🪟 On Windows

1. Download Tesseract installer from:  
   👉 https://github.com/UB-Mannheim/tesseract/wiki

2. During installation, note the install path (e.g. `C:\Program Files\Tesseract-OCR`).

3. Add it to your environment variables:
   ```
   setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
   ```

4. Verify installation:
   ```bash
   tesseract --version
   ```

### 🐧 On Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install tesseract-ocr -y
```

Check version:
```bash
tesseract --version
```

---

## 🇮🇳 Setting Up Hindi Language for OCR

### 1. Install the Hindi Language Pack

```bash
sudo apt install tesseract-ocr-hin
```

### 2. Verify Language Support

```bash
tesseract --list-langs
```

You should see:
```
hin
eng
```

### 3. Use Hindi in Your Code

When initializing OCR:
```python
pytesseract.image_to_string(image, lang='hin')
```

You can also combine:
```python
pytesseract.image_to_string(image, lang='hin+eng')
```

---

## 🔐 Environment Variables

Create a `.env` file in the **project root directory** with the following variables:

```bash
# .env

# Google API credentials (for route and map generation)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here

# OCR & PDF Directories
UPLOAD_DIR=uploads/
OUTPUT_DIR=output/
```

> ⚠️ Make sure to create the folders mentioned in `UPLOAD_DIR` and `OUTPUT_DIR` before running the server.

---

## 🧾 Example Workflow

1. Upload `inception.pdf` and `images_combined.pdf`
2. The system will:
   - Extract text and visual data
   - Merge both files such that images appear **after the first page**
   - Output a final report: **`final_merged.pdf`**
3. Download and view your structured report.

---

## 🧪 API Endpoints Overview

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `POST` | `/upload` | Upload files for report generation |
| `GET`  | `/health` | Server health check |
| `GET`  | `/download/{filename}` | Download generated report |

---


## 🧑‍💻 Developer Notes

- Make sure **Tesseract** is accessible in your system PATH.
- For parallel OCR processing, use multiprocessing or asyncio batch execution.
- Recommended Python version: **3.9+**

---

## 🧾 Output Example

Generated output files:

```
output/
├── text_extracted.txt
├── images_combined.pdf
└── final_merged.pdf
```

---

## 📦 Docker & GitHub Container Registry (GHCR)

You can build and publish this project as a Docker image to GitHub Container Registry (ghcr.io). The repository already includes a `Dockerfile` and a GitHub Actions workflow: `.github/workflows/docker-publish.yml`.

Image name and tag convention used by the workflow:
- Image: `ghcr.io/<owner>/<repo>` (for example `ghcr.io/ridz-shothikai/eng-insight-inseption-report`)
- Tag: the workflow pushes a tag equal to the first 7 characters of the commit SHA (short SHA). Example: `ghcr.io/<owner>/<repo>:a1b2c3d`

How it works in CI:
- On `push` to `main`, the workflow builds the image and pushes three tags: the full commit SHA, a `-full` alias, and the short 7-char SHA.
- The short SHA is derived from the `GITHUB_SHA` environment variable and used as the primary short tag.

Locally you can build and run the image (optional):

```bash
# Build locally (tags with short sha from git)
SHORT_SHA=$(git rev-parse --short=7 HEAD)
docker build -t ghcr.io/${GITHUB_REPOSITORY}:${SHORT_SHA} .

# Run (adjust ports if your app listens on a different port)
docker run --rm -p 8080:8080 ghcr.io/${GITHUB_REPOSITORY}:${SHORT_SHA}
```


