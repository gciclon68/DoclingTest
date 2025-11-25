# Repository Log

This file tracks the commit history and rationale for each change made in this repository. Update the log immediately after every commit using the template below.

## Log Template

```
## <YYYY-MM-DD> — <short description>
- Commit: `<hash>`
- Message: `<git commit message>`
- Details:
  - Summary of the change (bullet list as needed)
  - Tests or validations performed
  - Follow-up actions or TODOs (if any)
```

## 2025-11-25 — Add image extraction, AI captions, and OCR fixes

### Changes Made

**1. Image Extraction System**
- Added comprehensive image extraction functionality to all 7 pipelines
- Implemented `_export_images()` method using Docling's official API (`element.get_image(doc).save()`)
- Configured all pipelines with `images_scale = 2.0` and `generate_picture_images = True`
- Images saved to `assets/` folder at 2x resolution (144 DPI) for better quality
- Supports both figures (PictureItem) and tables (TableItem) extraction

**2. AI-Powered Image Descriptions (Vision LLM Pipeline)**
- Integrated OpenAI GPT-4o Vision API for automatic image captioning
- Implemented base64 encoding for image transmission to API
- Added `_generate_image_captions()` method with configurable max tokens
- Captions saved to `{filename}_captions.json` for reference
- Added comprehensive "AI Image Descriptions" section to markdown previews with:
  - Figure numbers and clickable image links
  - Detailed AI-generated descriptions
  - Proper formatting and navigation

**3. Statistics Counting Fix**
- Fixed `_extract_statistics()` to use `doc.export_to_dict()` instead of `iterate_items()`
- Now correctly counts text blocks, tables, and figures from OCR-processed documents
- Switched to direct array access (`doc_dict['texts']`, `doc_dict['tables']`, `doc_dict['pictures']`)

**4. Markdown Preview Enhancement**
- Fixed markdown preview generation for OCR documents
- Added fallback text extraction when `doc.export_to_markdown()` returns minimal content
- Manually extracts text from `doc_dict['texts']` array for readable output
- Preserves all extracted text blocks with proper formatting

**5. Dependencies**
- Added `openai>=1.54.0` to requirements.txt for Vision LLM functionality
- Added necessary imports: `base64`, `PictureItem`, `TableItem`

**6. Documentation**
- Created comprehensive USER-Guide.md with:
  - Step-by-step setup instructions
  - Pipeline comparison table with use cases
  - Troubleshooting section for common issues
  - Notes about Vision LLM requiring valid API key
  - Information about AI-generated captions and output structure

### Testing Performed
- Tested all 7 pipelines on scanned PDF documents
- Verified OCR text extraction (RapidOCR, EasyOCR, Tesseract pipelines)
- Confirmed image extraction at 2x resolution
- Validated Vision LLM pipeline generates AI descriptions
- Checked markdown preview displays extracted text correctly
- Verified statistics counting shows accurate text block counts

### Files Modified
- `scripts/ingest.py`: Added 235 lines (image export, AI captions, statistics fix, markdown enhancement)
- `requirements.txt`: Updated openai package version
- `USER-Guide.md`: Created new comprehensive user documentation

---

## Initial Setup
- Repository initialized with `.gitignore` tailored for Python, data processing artifacts, and optional Node tooling.
- `GIT.md` created to document commit history.
