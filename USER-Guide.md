# Docling Document Processor - User Guide

## Project Overview

This is a **Docling Document Processor** - an interactive tool for processing documents (PDFs, DOCX, PPTX, HTML, XML, Markdown) using IBM's Docling library with various OCR and AI pipelines.

## How to Run It

### 1. Setup Environment (First Time Only)

```bash
# Load Python module (if in HPC environment)
module load python/3.11.11

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Add Documents to Process

```bash
# Place your documents in the documents/ folder
cp your_file.pdf documents/
```

Supported formats: PDF, DOCX, PPTX, HTML, XML, Markdown

### 3. Run the Interactive Processor

```bash
python scripts/ingest.py
```

Or make it executable and run directly:
```bash
chmod +x scripts/ingest.py
./scripts/ingest.py
```

## What Happens When You Run It

The script provides an **interactive menu** that:

1. **Lists available documents** from the `documents/` folder
2. **Shows pipeline options** (7 different processing pipelines)
3. **Processes the document** and saves outputs to `out/`

## Available Pipelines

| Pipeline | Speed | Quality | Use Case | Notes |
|----------|-------|---------|----------|-------|
| **Simple PDF** | ⚡ Fastest | Basic | Born-digital PDFs, quick preview | No OCR, no models |
| **Standard PDF** | 🐢 Medium | High | Complex layouts, tables, figures | ~500MB models download |
| **RapidOCR** | 🐢 Medium | Good | Scanned documents, embedded images | Local OCR engine |
| **EasyOCR** | 🐌 Slow | Best | Multi-language, high accuracy | GPU recommended |
| **Tesseract** | 🐢 Medium | Good | Traditional OCR | Requires Tesseract install |
| **Vision LLM** | 🐌 Slowest | Best | AI image descriptions | 🔑 Requires API key |
| **Formula-Aware** | 🐢 Medium | High | Mathematical papers, LaTeX formulas | Preserves equations |

## Output Files

After processing, you'll get:

### Structured Data
```
out/
├── {filename}_docling.json      # Full DoclingDocument (structured data)
└── {filename}_preview.md        # Markdown preview (human-readable)
```

### Extracted Figures and Tables
```
assets/
├── {filename}_fig1.png          # Extracted figure 1
├── {filename}_fig2.png          # Extracted figure 2
├── {filename}_table1.png        # Extracted table 1 (as image)
└── ...
```

**Note:** All pipelines now extract figures and tables automatically. Images are saved at 2x resolution (144 DPI) for better quality. You can disable this by setting `EXTRACT_FIGURES=false` in `.Docling_env`.

### AI-Generated Image Captions (Vision LLM Only)
When using the **Vision LLM pipeline**, AI descriptions are automatically generated for each figure:
```
out/
├── {filename}_captions.json     # AI descriptions for all figures
```

The captions file contains detailed descriptions of each image, useful for:
- Understanding complex diagrams
- Accessibility and documentation
- Multimodal RAG (combining text + image descriptions)

## Configuration Files

### Pipeline Options (`.Docling_env`)
Edit `.Docling_env` to customize:
- Pipeline behavior
- OCR engines
- Output formats
- Feature flags
- Debug mode

**This file is tracked in git** - no sensitive data here.

### API Keys (`.Key_env`)
For Vision LLM pipeline, add your API keys to `.Key_env`:

```bash
# OpenAI (for GPT-4 Vision)
OPENAI_API_KEY=sk-...

# OR Anthropic (for Claude 3.5 Sonnet)
ANTHROPIC_API_KEY=sk-ant-...
```

**IMPORTANT:** The Vision LLM pipeline will **only appear in the menu** if you have a valid API key. If you see only 6 pipelines instead of 7, check that you've replaced `your_openai_api_key_here` with a real API key in `.Key_env`.

**This file is in .gitignore** - never commit API keys!

## Quick Example Workflow

```bash
# 1. Setup
module load python/3.11.11
source venv/bin/activate

# 2. Add document
cp research_paper.pdf documents/

# 3. Process
python scripts/ingest.py

# 4. Follow the interactive prompts:
#    - Select document (enter number)
#    - Choose pipeline (enter number)
#    - Wait for processing

# 5. View results
cat out/research_paper_preview.md
```

## Interactive Menu Example

```
============================================================
  DOCLING DOCUMENT PROCESSOR
  Interactive Document Ingestion Tool
============================================================

📂 Available documents:
   1. research_paper.pdf (2.45 MB)
   2. scanned_doc.pdf (8.12 MB)
   0. Exit

Select document number: 1

🔧 Available Pipelines:
   1. ⚡ Simple PDF Pipeline
      Fast processing without OCR (born-digital PDFs only)
   2. 📦 Standard PDF Pipeline
      Layout analysis with tables and figures
   3. 📦 RapidOCR Pipeline
      Fast OCR for scanned documents
   ...
   0. Back

Select pipeline number: 2

🔧 Setting up Standard PDF Pipeline...
   ⚠️  First run will download ~500MB of models
   ✓ Pipeline ready!

📄 Processing: research_paper.pdf
...
```

## Troubleshooting

### No documents found
```bash
# Make sure documents exist
ls documents/

# Add a test document
cp /path/to/file.pdf documents/
```

### Pipeline fails
- **First run**: ML models will download (~500MB), this is normal
- **OCR errors**: Try a different OCR engine or Simple pipeline
- **Memory issues**: Use Simple pipeline for large documents

### Vision LLM pipeline not showing in menu
- The Vision LLM pipeline is **hidden** if no valid API key is detected
- Check `.Key_env` has a real API key (not the placeholder `your_openai_api_key_here`)
- Key format: `OPENAI_API_KEY=sk-...` (no quotes, no extra spaces)
- Restart script after adding keys
- You should see 7 pipelines if the API key is valid, only 6 if not

### Tesseract not found
```bash
# Install Tesseract OCR
# On RHEL/CentOS:
sudo yum install tesseract

# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# On macOS:
brew install tesseract
```

## Mathematical Formulas

Docling preserves mathematical formulas in LaTeX format:

- **Born-digital PDFs**: Formulas extracted as LaTeX directly
- **Scanned docs**: Use OCR pipelines (formulas may need manual review)
- **Greek letters**: Preserved in Unicode (α, β, γ, etc.)

Example output:
```markdown
The equation is: $$E = mc^2$$

Greek symbols: α + β = γ
```

## Advanced Usage

### Debug Mode

Enable verbose logging in `.Docling_env`:
```bash
DEBUG_MODE=true
```

### Using JSON Output

The JSON output (`*_docling.json`) contains the complete document structure:

```json
{
  "pages": [...],
  "elements": [...],
  "tables": [...],
  "figures": [...]
}
```

Use this for:
- Custom processing pipelines
- Integration with other tools
- Building vector databases (Phase 2)

## Project Structure

```
DoclingTest/
├── documents/           # Input documents (add your files here)
├── out/                 # Processed outputs (JSON, markdown)
├── assets/              # Extracted figures/images
├── scripts/             # Processing scripts
│   ├── ingest.py       # Main interactive processor
│   └── config.py       # Configuration loader
├── venv/                # Python virtual environment
├── .Docling_env         # Pipeline configs (tracked in git)
├── .Key_env             # API keys (NOT in git)
├── README.md            # Developer documentation
└── USER-Guide.md        # This file
```

## Next Steps

After Phase 1 (document ingestion), proceed to:

1. **Phase 2**: Vector database creation (`scripts/embed_index.py`)
2. **Phase 3**: RAG retrieval system (`scripts/answer.py`)
3. **Phase 4**: Advanced multimodal search

See `EXECUTION_PLAN.md` for full roadmap.

## Support

For issues with:
- **Docling library**: https://github.com/DS4SD/docling
- **This project**: Check `EXECUTION_PLAN.md` or `README.md`

---

**Happy document processing! 📄✨**
