## CHANGELOG

**Last Updated:** 2025-12-07 17:35:00  
**Version:** 1.0  
**Changes:**  
- Initial comprehensive documentation created
- Organized from initial RAG system design session

---

# Formula Detection in Docling - Explanation

## Current Status

### What Works ✓
1. **Formula Detection**: Docling DOES detect formulas in PDFs - it identifies 99 formulas in clase10.pdf
2. **Formula Counting**: The statistics now correctly show formula counts (see line 13 in preview.md)
3. **Formula Extraction**: The code now extracts formula text from JSON and displays it in markdown with LaTeX formatting (`$$formula$$`)

### What's Limited ⚠️

#### The OCR Formula Problem
**Why you see "formula-not-decoded":**
- OCR engines (RapidOCR, EasyOCR, Tesseract) can read **text**, but they struggle with **mathematical notation**
- Complex LaTeX formulas with symbols (∫, ∑, √, Greek letters, fractions) are often misread or not read at all
- Example from clase10.pdf:
  - Real formula: `R_{DS(on)} = k(V_{GS} - V_T)^{-1}`
  - OCR reads it as: `V + V = 0` or just `dt` or completely empty

#### Mismatch Between Detection and Extraction
From clase10.pdf analysis:
- **99 formulas detected** (Docling knows they exist)
- **Only 72 have placeholders** in markdown (Docling's export skips some)
- **Many have empty or wrong text** (OCR failed to decode them)

This is why you still see `<!-- formula-not-decoded -->` - it means "formula exists here, but couldn't extract it properly"

## Solutions

### 1. Use Formula-Aware Pipeline (Best for Born-Digital PDFs)
```bash
python scripts/ingest.py
# Select: Formula-Aware Pipeline
```

**Best for:**
- PDFs with embedded LaTeX (academic papers created with LaTeX, Word equation editor)
- Born-digital PDFs (not scanned)
- Documents where formulas are actual text, not images

**How it works:**
- Extracts formulas directly from PDF structure
- Preserves LaTeX notation
- No OCR needed for formulas

### 2. Current Behavior After My Fix
With the updated code:
- Formulas are counted correctly ✓
- Non-empty formula text is displayed in LaTeX format: `$$formula$$`
- Empty formulas show as: `*(formula not available)*`

**Example output:**
```markdown
- Formulas: 99

The equation is:
$$V + V_L = 0$$

where:
$$\frac{di}{dt}$$
```

### 3. For Scanned Documents with Complex Math
If you have **scanned PDFs** with complex math formulas:

**Option A: Vision LLM Pipeline** (requires OpenAI API key)
- Uses GPT-4V to understand formula images
- Can generate descriptions like: "This shows Ohm's law: V = IR"
- Won't give you LaTeX, but gives semantic understanding

**Option B: Use Specialized Tools**
For proper LaTeX extraction from scanned math documents, you need specialized tools:
- **Mathpix**: Commercial API for math OCR
- **Nougat**: AI model specialized for academic PDFs (mentioned in EXECUTION_PLAN.md)
- **LaTeX-OCR**: Open-source models specifically for math notation

## Testing the Fix

### Reprocess a Document
To see the improvements, reprocess your document:

```bash
cd /home/cuervo/AI-Projects/DoclingTest/DoclingTest
python scripts/ingest.py
# Select your document (e.g., clase10.pdf)
# Choose a pipeline (try Formula-Aware for best results)
```

### Check the Results
1. **Statistics Section**: Should show correct formula count
2. **Formulas in Content**: Should show as `$$formula text$$` instead of `<!-- formula-not-decoded -->`
3. **Empty Formulas**: Will show as `*(formula not available)*`

### Example Comparison

**Before Fix:**
```markdown
- Formulas: 0

The circuit equation is:
<!-- formula-not-decoded -->
```

**After Fix:**
```markdown
- Formulas: 99

The circuit equation is:
$$V + V_L = 0$$
```

## Why Some Formulas Still Can't Be Decoded

This is a fundamental OCR limitation, not a bug:

1. **OCR reads characters, not math symbols**: OCR was designed for text documents, not mathematical notation
2. **Complex formulas are images**: In scanned PDFs, formulas are just pictures of symbols
3. **LaTeX is semantic, OCR is visual**: OCR sees lines and curves, but doesn't understand "this is a fraction" or "this is an integral"

### What Each Pipeline Can Do

| Pipeline | Formula Detection | Formula Extraction | Quality |
|----------|------------------|-------------------|---------|
| **Simple** | ❌ No | ❌ No | N/A |
| **Standard** | ✓ Yes | 🟡 Basic | Low |
| **RapidOCR** | ✓ Yes | 🟡 Text only | Low-Medium |
| **EasyOCR** | ✓ Yes | 🟡 Text only | Medium |
| **Formula-Aware** | ✓ Yes | ✓ LaTeX | **High** (born-digital only) |
| **Vision LLM** | ✓ Yes | 🟡 Descriptions | Medium (semantic) |

## Recommendations

### For Your Use Case (Engineering PDFs with Formulas)

1. **If PDFs are born-digital** (created with LaTeX, Word, etc.):
   - Use **Formula-Aware Pipeline**
   - Formulas will be extracted as proper LaTeX

2. **If PDFs are scanned/photos**:
   - Use **Vision LLM Pipeline** for semantic understanding
   - Or accept that formulas will be detected but not fully decoded
   - Consider manual review for critical formulas

3. **Current Setup**:
   - Your current fix will show whatever OCR could extract
   - At least you'll see partial formulas instead of just placeholders
   - Formula count is accurate

## Next Steps

Want to improve formula extraction further? You could:
1. Add Mathpix API integration (paid service, excellent math OCR)
2. Integrate Nougat model (free, specialized for academic PDFs)
3. Use Formula-Aware pipeline for compatible PDFs
4. Manually verify critical formulas from the JSON output

Let me know which direction you want to go!
