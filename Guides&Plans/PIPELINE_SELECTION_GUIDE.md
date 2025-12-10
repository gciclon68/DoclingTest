## CHANGELOG

**Last Updated:** 2025-12-07 17:35:00  
**Version:** 1.0  
**Changes:**  
- Initial comprehensive documentation created
- Organized from initial RAG system design session

---

# Docling Pipeline Selection Guide

## Quick Decision Tree

```
Is your PDF scanned/image-based?
├─ YES → Use OCR pipeline (RapidOCR, EasyOCR, or Tesseract)
│   ├─ Has formulas? → EasyOCR (best accuracy)
│   ├─ Need speed? → RapidOCR
│   └─ Already have Tesseract installed? → Tesseract
│
└─ NO (born-digital PDF)
    ├─ Has complex math/formulas? → Formula-Aware
    ├─ Need AI image descriptions? → Vision LLM ($$)
    ├─ Has tables/layouts? → Standard
    └─ Just text? → Simple
```

## Complete Pipeline Comparison

| Pipeline | Speed | Quality | Best For | Limitations | Cost |
|----------|-------|---------|----------|-------------|------|
| **1. Simple** | ⚡⚡⚡ Very Fast | 🟡 Basic | Plain text PDFs | No OCR, no tables, no formulas | Free |
| **2. Standard** | ⚡⚡ Fast | 🟢 Good | Documents with tables/layouts | No OCR, downloads 500MB models | Free |
| **3. RapidOCR** | ⚡⚡ Fast | 🟡 Medium | Scanned docs, images | Poor formula recognition | Free |
| **4. EasyOCR** | ⚡ Slow | 🟢 Good | Multi-language scans | Slower, 200MB models, poor formulas | Free |
| **5. Tesseract** | ⚡⚡ Fast | 🟡 Medium | When Tesseract installed | Needs system install, poor formulas | Free |
| **6. Vision LLM** | ⚡ Slow | 🟢🟢 Excellent | AI image descriptions | API costs, needs internet | ~$1-3/doc |
| **7. Formula-Aware** | ⚡⚡ Fast | 🟢 Good | Math documents (LaTeX) | Only born-digital PDFs with embedded formulas | Free |

## Detailed Use Cases

### Case 1: **Plain Text Documents**
**Example:** Reports, articles, books (no images, no formulas)

**Best Pipeline:** `Simple` (1)
- ✅ Fastest processing
- ✅ No model downloads
- ❌ Won't extract images or formulas

**When to process:**
```bash
python scripts/ingest.py
# Select: Simple PDF
```

---

### Case 2: **Documents with Tables**
**Example:** Financial reports, data sheets, spreadsheets in PDF

**Best Pipeline:** `Standard` (2)
- ✅ Table detection with TableFormer
- ✅ Structured grid output
- ✅ Layout analysis
- ❌ No OCR (needs born-digital PDF)

**Output:**
- Tables as structured JSON (`grid` array)
- Table images as base64
- Text content

---

### Case 3: **Scanned Documents** (no special content)
**Example:** Scanned books, old documents, photocopied papers

**Best Pipelines:**
1. **RapidOCR** (3) - Fast, good enough
2. **EasyOCR** (4) - Better accuracy, slower

**Choose RapidOCR if:**
- Processing many documents
- English/common languages
- Speed matters

**Choose EasyOCR if:**
- Need high accuracy
- Multi-language content (80+ languages)
- Complex text layouts

---

### Case 4: **Mathematical/Scientific Documents**
**Example:** Physics papers, engineering docs, equations

**Situation A: Born-digital PDF (LaTeX-generated)**
**Best Pipeline:** `Formula-Aware` (7)
- ✅ Preserves LaTeX formulas
- ✅ Detects equation blocks
- ✅ Best for academic PDFs
- ❌ Doesn't work on scanned docs

**Situation B: Scanned scientific document**
**Best Pipeline:** `Vision LLM` (6) or `EasyOCR` (4)
- Vision LLM: AI describes formulas semantically
- EasyOCR: Attempts OCR (often fails on complex math)

---

### Case 5: **Engineering Diagrams & Schematics**
**Example:** Circuit diagrams, technical drawings, CAD exports

**Best Pipeline:** `Vision LLM` (6)
- ✅ AI describes diagrams in detail
- ✅ Identifies components
- ✅ Useful for RAG search
- 💰 Costs ~$0.01-0.03 per image

**Alternative:** `Standard` (2) + manual review
- Extracts images but no descriptions
- You describe them manually later

---

### Case 6: **Mixed Content** (text + images + tables + formulas)
**Example:** Technical manuals, research papers, presentations

**Strategy: Multi-pipeline approach**

**Step 1:** Determine PDF type
```bash
# Check if scanned or born-digital
pdfinfo document.pdf | grep "Page size"
# or open PDF and select text - if you can select, it's born-digital
```

**Step 2A: If born-digital**
```bash
# First pass: Extract structure
python scripts/ingest.py
# Select: Formula-Aware (or Standard if no formulas)

# Second pass: Add AI descriptions (optional)
python scripts/ingest.py
# Select: Vision LLM
```

**Step 2B: If scanned**
```bash
# Use OCR + Vision LLM
python scripts/ingest.py
# Select: Vision LLM (includes RapidOCR)
```

**Result:** Multiple JSON files
- `doc_formula-aware_docling.json` - Structure
- `doc_vision-llm_docling.json` - With descriptions
- Merge them in embedding phase

---

### Case 7: **Presentations (Google Slides, PowerPoint)**
**Example:** clase10.pdf (your current case)

**Characteristics:**
- Created with presentation software
- Formulas as images (rasterized)
- Mix of text and graphics on each slide

**Best Pipeline:** `Vision LLM` (6)
- ✅ Describes slide content
- ✅ Interprets formulas semantically
- ✅ Good for presentations
- 💰 Moderate cost

**Alternative:** `Standard` (2) + manual formula annotation
- Free but requires manual work

---

## Pipeline Detector: Do You Need One?

### Short Answer: **Maybe, but not critical**

### Why you might want one:
```python
def detect_optimal_pipeline(pdf_path):
    """Auto-detect best pipeline for a PDF."""

    # Check if scanned
    is_scanned = check_if_scanned(pdf_path)

    # Check content
    has_formulas = check_for_formulas(pdf_path)
    has_tables = check_for_tables(pdf_path)
    has_images = check_for_images(pdf_path)

    # Decide
    if is_scanned:
        if has_formulas or has_images:
            return "vision_llm"  # Best quality, costs money
        else:
            return "rapidocr"    # Fast OCR
    else:
        if has_formulas:
            return "formula_aware"
        elif has_tables:
            return "standard"
        else:
            return "simple"
```

### When it's useful:
- ✅ Processing hundreds of PDFs
- ✅ Unknown document types
- ✅ Automation/batch workflows

### When you don't need it:
- ❌ Small doc collections (you know what they are)
- ❌ Consistent document types
- ❌ You want manual control

---

## Recommended Workflows

### Workflow 1: Single Document Type
```bash
# You know your documents → Pick one pipeline
python scripts/ingest.py
# Always select the same pipeline
```

### Workflow 2: Mixed Document Collection
```bash
# Process each type separately
for doc in *.pdf; do
    if [[ $doc == *"scan"* ]]; then
        python scripts/ingest.py --pipeline rapidocr --input $doc
    elif [[ $doc == *"math"* ]]; then
        python scripts/ingest.py --pipeline formula_aware --input $doc
    else
        python scripts/ingest.py --pipeline standard --input $doc
    fi
done
```

### Workflow 3: Quality-First (Recommended for RAG)
```bash
# Pass 1: Structure extraction
python scripts/ingest.py --pipeline formula_aware --input doc.pdf

# Pass 2: AI enhancement (if budget allows)
python scripts/ingest.py --pipeline vision_llm --input doc.pdf

# Result: Best of both worlds
# - Structured text/tables from formula_aware
# - AI descriptions from vision_llm
```

---

## Cost Considerations

### Free Pipelines (1-5, 7):
- Unlimited processing
- May need to download models (one-time)
- Local computation only

### Vision LLM (6):
**Costs per document:**
- Small doc (5 images): ~$0.05-0.15
- Medium doc (20 images): ~$0.20-0.60
- Large doc (50+ images): ~$0.50-1.50

**GPT-4V pricing:**
- $0.01 per image (low res)
- $0.03 per image (high res)

**Budget strategy:**
- Use Vision LLM for **key documents only**
- Or **important images only** (diagrams, complex tables)
- Use free pipelines for bulk content

---

## Summary Recommendation for Your Use Case

**Your documents:** Engineering PDFs with formulas, diagrams, tables

**Recommended approach:**

1. **Default pipeline:** `Standard` (2)
   - For most technical docs with tables
   - Free, good quality

2. **For math-heavy docs:** `Formula-Aware` (7)
   - Born-digital PDFs with LaTeX
   - Academic papers, equation-heavy content

3. **For critical docs:** `Vision LLM` (6)
   - Key reference materials
   - Documents where formulas/diagrams are essential
   - Budget permitting

4. **For scanned docs:** `EasyOCR` (4)
   - Old manuals, photocopies
   - Multi-language content

**Don't use:**
- Simple (too basic for engineering docs)
- Tesseract (unless already installed)
- RapidOCR (EasyOCR is better for technical content)
