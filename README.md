# Docling Document Processor

Interactive document processing tool using IBM Docling for multimodal RAG pipelines.

## Quick Start

### 1. Activate Environment

```bash
# Load Python module
module load python/3.11.11

# Activate virtual environment
source venv/bin/activate
```

### 2. Add Documents

Place your documents in the `documents/` folder:
```bash
cp your_file.pdf documents/
```

Supported formats: PDF, DOCX, PPTX, HTML, XML, Markdown

### 3. Run Interactive Processor

```bash
python scripts/ingest.py
```

The script will guide you through:
1. Selecting a document from `documents/`
2. Choosing a processing pipeline
3. Processing and viewing results

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

After processing, outputs are saved to `out/`:

```
out/
├── document_name_docling.json      # Full DoclingDocument (structured data)
├── document_name_preview.md        # Markdown preview (human-readable)
└── document_name_captions.json     # AI image descriptions (Vision LLM only)
```

Extracted figures and tables go to `assets/` at 2x resolution (144 DPI):

```
assets/
├── document_name_fig1.png          # Extracted figures
├── document_name_fig2.png
├── document_name_table1.png        # Extracted tables (as images)
└── ...
```

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

### Vision LLM not available
- Check `.Key_env` has valid API key
- Key format: `OPENAI_API_KEY=sk-...` (no quotes)
- Restart script after adding keys

## Example Workflow

```bash
# 1. Setup
module load python/3.11.11
source venv/bin/activate

# 2. Add document
cp research_paper.pdf documents/

# 3. Process
python scripts/ingest.py

# 4. View results
cat out/research_paper_preview.md
```

## Advanced Usage

### Debug Mode

Enable verbose logging in `.Docling_env`:
```bash
DEBUG_MODE=true
```

### Custom Output Formats

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

## Next Steps

After Phase 1 (document ingestion), proceed to:

1. **Phase 2**: Vector database creation ([embed_index.py](scripts/embed_index.py))
2. **Phase 3**: RAG retrieval system ([answer.py](scripts/answer.py))
3. **Phase 4**: Advanced multimodal search

See [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for full roadmap.

## Project Structure

```
DoclingTest/
├── documents/           # Input documents
├── out/                 # Processed outputs (JSON, markdown)
├── assets/              # Extracted figures/images
├── scripts/             # Processing scripts
│   ├── ingest.py       # Main interactive processor
│   └── config.py       # Configuration loader
├── venv/                # Python virtual environment
├── .Docling_env         # Pipeline configs (tracked in git)
├── .Key_env             # API keys (NOT in git)
└── README.md            # This file
```

## Support

For issues with:
- **Docling library**: https://github.com/DS4SD/docling
- **This project**: Check [EXECUTION_PLAN.md](EXECUTION_PLAN.md) or open an issue

---

**Happy document processing! 📄✨**
