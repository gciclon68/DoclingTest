# Docling Document Processor

Interactive document processing tool using IBM Docling for multimodal RAG pipelines, with folder navigation and batch processing.

## Quick Start

1. Activate environment  
   ```bash
   module load python/3.11.11
   source venv/bin/activate
   ```
2. Install dependencies (includes new `tabulate`)  
   ```bash
   pip install -r requirements.txt
   # or: pip install tabulate
   ```
3. Add documents (subfolders are supported)  
   ```bash
   cp your_file.pdf documents/
   cp spec.pdf documents/project_a/spec.pdf
   ```
   Supported formats: PDF, DOCX, PPTX, HTML, XML, Markdown
4. Run interactive processor  
   ```bash
   python scripts/ingest.py
   ```

## Navigation & Selection

- Folders use `d` prefixes: `d1`, `d2`, …; documents use numbers: `1`, `2`, `3`, …
- `..` moves up one level; `0` exits.
- Multi-select with commas/ranges: `1-3`, `1,3,5`, `1-3,5,8-10`, or `all`.
- Large batches (>20 files) trigger a size/time warning before running.

Example prompt:
```
📂 Current: documents/project_a/

  d1. 📁 subfolder_1/
  d2. 📁 subfolder_2/
   1. 📄 spec.pdf (45.2 MB)
   2. 📄 manual.docx (12.8 MB)

  .. (go up)  |  0 (exit)

Select folder (d1,d2...) or documents (1-3, 1,3, all):
```

## Available Pipelines

| Pipeline | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| **Simple PDF** | ⚡ Fastest | Basic | Born-digital PDFs, quick preview |
| **Standard PDF** | 🐢 Medium | High | Complex layouts, tables, figures |
| **RapidOCR** | 🐢 Medium | Good | Scanned documents, embedded images |
| **EasyOCR** | 🐌 Slow | Best | Multi-language, high accuracy |
| **Tesseract** | 🐢 Medium | Good | Traditional OCR (requires install) |
| **Vision LLM** | 🐌 Slowest | Best | AI image descriptions (requires API key) |
| **Formula-Aware** | 🐢 Medium | High | Mathematical papers, LaTeX formulas |

## Configuration

### Pipeline Options (`.Docling_env`)

Edit `.Docling_env` to customize:
- Pipeline behavior
- OCR engines
- Output formats
- Feature flags

**This file is tracked in git** - no sensitive data here.

### API Keys (`.Key_env`)

For Vision LLM pipeline, add your API keys to `.Key_env`:

```bash
# OpenAI (for GPT-4 Vision)
OPENAI_API_KEY=sk-...

# OR Anthropic (for Claude 3.5 Sonnet)
ANTHROPIC_API_KEY=sk-ant-...
```

**This file is in .gitignore** - never commit API keys!

## Output Structure

Outputs mirror the `documents/` folder structure:

```
documents/project_a/spec.pdf      -> out/project_a/spec_docling.json
documents/project_a/spec.pdf      -> assets/project_a/spec_fig1.png
```

Files per document:
```
out/
├── document_name_docling.json      # Full DoclingDocument (structured data)
├── document_name_preview.md        # Markdown preview (human-readable)
└── document_name_captions.json     # AI image descriptions (Vision LLM only)

assets/
├── document_name_fig1.png          # Extracted figures
├── document_name_fig2.png
├── document_name_table1.png        # Extracted tables (as images)
└── ...
```

Extracted figures and tables are saved at 2x resolution (144 DPI).

**Note:** All pipelines now automatically extract figures and tables. Set `EXTRACT_FIGURES=false` in `.Docling_env` to disable.

## AI Image Descriptions (Vision LLM)

When using the **Vision LLM pipeline**, AI-generated descriptions are automatically created for each extracted figure:

- Descriptions appear in the markdown preview under an "AI Image Descriptions" section
- Each figure includes a clickable link to the image file
- Descriptions are also saved to `{filename}_captions.json` for programmatic access
- Uses OpenAI GPT-4o or Anthropic Claude 3.5 Sonnet (configurable in `.Docling_env`)

**Requirements:**
- Valid API key in `.Key_env` (OpenAI or Anthropic)
- Vision LLM pipeline only appears in menu when valid API key is detected

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

## Batch Processing

- Single-document mode remains unchanged; batch mode runs when you select multiple files.
- A single pipeline is applied to the entire batch (chosen once).
- Pre-flight validation flags missing files and very large documents before starting.
- Progress shows `[n/total]` with per-file timing; errors are captured per document.
- Keyboard interrupts offer to save partial results.
- Summaries are tabulated (`tabulate`) with columns: File | Time | Pipeline | Pages | Text | Pics | Tables | Formulas | Status.

## Troubleshooting

- **No documents found**: `ls documents/` then add files to that tree.
- **Pipeline fails**: First run downloads models (~500MB). Try a simpler pipeline for large files or OCR issues.
- **Vision LLM not available**: Ensure `.Key_env` has a valid key (`OPENAI_API_KEY=sk-...`), then restart.

## Example Workflow

```bash
# 1. Setup
module load python/3.11.11
source venv/bin/activate
pip install -r requirements.txt

# 2. Add documents (optionally in subfolders)
cp research_paper.pdf documents/papers/

# 3. Process (navigate + multi-select)
python scripts/ingest.py

# 4. View results (mirrors input folders)
cat out/papers/research_paper_preview.md
ls assets/papers/
```

## Next Steps

After ingestion:
1. Build vector index: `scripts/embed_index.py`
2. Run retrieval: `scripts/answer.py`
3. Explore multimodal search: see `EXECUTION_PLAN.md`

## Project Structure

```
DoclingTest/
├── documents/           # Input documents (supports nested folders)
├── out/                 # Processed outputs (mirrors document tree)
├── assets/              # Extracted figures/images (mirrors document tree)
├── scripts/             # Processing scripts
│   ├── ingest.py        # Main interactive processor
│   └── config.py        # Configuration loader
├── venv/                # Python virtual environment
├── .Docling_env         # Pipeline configs (tracked in git)
├── .Key_env             # API keys (NOT in git)
└── README.md            # This file
```
