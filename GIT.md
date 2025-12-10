# Git Commit Log

## Upcoming Commit - 2025-12-07 17:40

### Summary
Multimodal RAG system design and formula detection improvements

### Changes
**Documentation:**
- Created comprehensive RAG design in `Guides&Plans/`
- Added pipeline selection guide with decision tree
- Created implementation plan with phase breakdown
- Added formula detection analysis and explanations
- All docs include changelogs

**Code:**
- Fixed formula counting in `scripts/ingest.py`
- Enhanced markdown formula display with LaTeX format
- Simplified pipeline naming to use only numbers (p1-p7)
- Updated pipeline config with numbered names

**Configuration:**
- Created `.Key_env` for OpenAI API key storage
- Added to `.gitignore` for security

### Files Modified
- `scripts/ingest.py` - Formula fixes, pipeline naming
- `scripts/config.py` - Pipeline numbering
- `requirements.txt` - Dependencies updated
- `Claude.md` - Session summary created
- `GIT.md` - This file

### Files Added
- `Guides&Plans/MULTIMODAL_RAG_DESIGN.md`
- `Guides&Plans/PIPELINE_SELECTION_GUIDE.md`
- `Guides&Plans/CLEAR_IMPLEMENTATION_PLAN.md`
- `Guides&Plans/FORMULA_DETECTION_EXPLAINED.md`
- `Guides&Plans/CLASE10_FORMULA_ANALYSIS.md`
- `.Key_env`

### Batch ID
Will be added after commit

---

## Commit History

### 2025-12-07 03:30 - Document batch ingest features
**Batch ID:** 83cae3e
**Changes:**
- Ignored OPAMP generation folder
- Documented batch processing

### 2025-12-07 02:00 - Image extraction and OCR fixes
**Batch ID:** 54970b0
**Changes:**
- Added image extraction to assets
- AI caption generation with Vision LLM
- OCR accuracy improvements

### 2025-12-07 00:15 - Initial commit
**Batch ID:** 43f2b21
**Changes:**
- Initial Docling tool setup
- Basic pipeline configuration
- Document processing infrastructure
