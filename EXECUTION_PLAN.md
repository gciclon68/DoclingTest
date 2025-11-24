# Docling-first Multimodal RAG: Execution Plan (v1)

## Objectives

- Build a robust, locally runnable document parsing pipeline that preserves text, tables, figures, and equations from engineering PDFs.
- Produce an intermediate, human/audit-friendly artifact (Markdown + assets) from the Docling document graph that captures all objects and metadata.
- Evaluate local OCR options (RapidOCR/Tesseract/Nougat) vs selective remote augmentation (e.g., VLM captioning) for quality and speed.
- Index multimodal signals (text, table-serialized text, image/figure embeddings) into a local vector DB and enable RAG chat.
- Automate ingestion and Q&A with n8n; prepare MCP config to expose project resources to Codex tooling.

---

## Architecture Overview

- Ingestion: Docling PDF pipeline → DoclingDocument (structured) → Exporter → RAG Markdown + assets.
- Embeddings/Indexing:
  - Text embeddings from paragraph/section chunks.
  - Table-aware linearization and table embeddings.
  - Image/figure embeddings via local CLIP (cross-modal text↔image retrieval).
  - Stored in Chroma (3 collections: `text_chunks`, `table_chunks`, `image_chunks`).
- Retrieval & Chat: Query routing across modalities → fusion/rerank → prompt builder with citations → LLM (Ollama/OpenAI/Mistral).
- Evaluation: OCR fidelity, table recall/precision, retrieval quality, latency; manual inspection via rendered Markdown.
- Automation: n8n workflows for ingestion (file drops/webhook) and Q&A (webhook).

---

## Phase 1 — OCR & Docling Pipelines (Local-first)

1. Baseline (Windows-friendly, offline)
   - Use Docling `StandardPdfPipeline` if model downloads are allowed; otherwise keep the existing `UltraSimplePdfPipeline` in `ingest.py` as a fallback.
   - Configure OCR:
     - RapidOCR (preferred in Docling, good balance of quality/speed).
     - Tesseract as a fallback (install system binary; set path; table accuracy is weaker; keep for comparison).
     - Nougat (local vision-transformer OCR; heavier but good on academic PDFs). Keep optional due to model size.
   - Ensure Windows font path patch (Arial) remains to prevent OCR crashes.

2. Optional remote augmentation (clarification)
   - Mistral is not an OCR API; treat it as a VLM/LLM for: figure/table captioning, table repair (header inference), and diagram/text summarization.
   - True remote OCR alternatives (if needed): Azure Document Intelligence, Google Document AI, AWS Textract (paid services). Use sparingly for specific pages.

3. Page selection heuristics (cost-aware)
   - If default OCR is low-confidence or pages contain complex figures/tables, selectively augment with VLM captioning or higher-fidelity OCR.
   - Confidence signals: OCR per-page confidence if available, density of detected lines/graphics, presence of embedded images.

Deliverables
   - Keep `ingest.py` minimal for now (works offline) but add flags to switch pipelines as environment allows.

---

## Phase 2 — Intermediate Artifact: RAG Markdown (RMMD v0.2)

Design a durable, auditable, multimodal artifact produced from Docling’s JSON/graph.

1. Format
   - One `.md` per input file; assets in `assets/<doc_stem>/`. Sidecar `.json` optional for raw Docling export.
   - YAML frontmatter: document metadata (title, pages, created_at, pipeline, ocr_provider, versions, checksum).
   - Body composed of object blocks with stable `object_id` and references:

Example skeleton

```
---
doc_id: testMOS
source: documents/testMOS.pdf
pages: 42
pipeline: standard_pdf | ultra_simple
ocr: rapidocr | tesseract | nougat
created_at: 2025-10-27T19:00:00Z
version: rmmd-0.2
---

# Title (if detected)

<!-- obj:text -->
[[obj:txt_0001]] (page: 1 bbox: [x1,y1,x2,y2])
Paragraph text…

<!-- obj:table -->
[[obj:tbl_0001]] (page: 2 bbox: […]) caption: MOSFET Ratings

```table-csv
Voltage,Max,Unit
V_DS,600,V
…
```

<!-- obj:figure -->
[[obj:fig_0003]] (page: 5 bbox: […]) caption: Cross-section of power MOSFET
![fig_0003](assets/testMOS/fig_0003.png)

```alt-text
Generated or extracted caption/description suitable for retrieval.
```

<!-- obj:eq -->
[[obj:eq_0010]] (page: 6 bbox: […])
```latex
R_{DS(on)} = k (V_{GS} - V_T)^{-1}
```
```

2. Table serialization
   - Primary: CSV code block as above (lossless headers, rows). Include caption, title, unit hints.
   - Secondary linearization for embedding: “Table: [caption]. Row: [header→value] …” saved as an additional text chunk per row or per group.

3. Figures
   - Save cropped figure images; store caption from source or VLM-generated alt-text.
   - Optionally OCR embedded text inside figures (e.g., schematic labels) using ROI OCR.

4. Cross-references
   - Maintain `[[obj:…]]` anchors and map them to metadata used in vector DB for precise citations.

---

## Phase 3 — Exporter Implementation Plan

Create `export_docling.py` to convert DoclingDocument → RMMD + assets.

Steps
1. Load Docling conversion result (use existing `DocumentConverter` as in `ingest.py`).
2. Traverse pages and objects: paragraphs, headings, lists, tables, figures, equations, references.
3. Save figures to `assets/<doc_stem>/fig_XXXX.png` (use Docling crops); preserve bbox and page.
4. Normalize tables: capture header rows; emit CSV block; also generate linearized text rows for embeddings.
5. Emit YAML frontmatter and body blocks with stable `object_id`.
6. Optional VLM pass (local Ollama multimodal or remote Mistral) to generate/repair captions and table summaries when missing.
7. CLI usage: `python export_docling.py --pdf documents/testMOS.pdf --out out/ --vlm=none|ollama|mistral`

Notes
- Keep the exporter decoupled from embedding to enable manual QA of RMMD outputs.

---

## Phase 4 — Embeddings and Indexing (Multimodal)

1. Text embeddings (local-first)
   - Sentence-transformers local model: `all-MiniLM-L6-v2` (fast, ~384-d). Alternative: `bge-small-en-v1.5`.
   - Chunking: by section/paragraph with heading context; overlap 20–40 tokens; keep `object_id`, `page`, `section_path` in metadata.

2. Table embeddings
   - Strategy A (recommended first): embed linearized rows or small row groups as text.
   - Strategy B (optional): embed full CSV-as-text with truncated rows for global table retrieval.

3. Image embeddings
   - Local CLIP: `clip-ViT-B-32` (via sentence-transformers) for figure images and their captions; store both image and caption vectors.

4. Index layout (Chroma)
   - `text_chunks` (id: `txt_*`)
   - `table_chunks` (id: `tbl_*` + row suffix where applicable)
   - `image_chunks` (id: `fig_*`)
   - Store: `object_id`, `doc_id`, `page`, `bbox`, `section_path`, `modality`, `caption`.

5. Fusion retrieval
   - Query fan-out: run text retrieval against `text_chunks` and `table_chunks`.
   - If query hints at visuals (e.g., “figure”, “diagram”, “schematic”, “see graph”), also run text→image retrieval via CLIP against `image_chunks` captions and optionally run image space similarity if an image is provided later.
   - Merge top-k with reciprocal-rank fusion; de-duplicate by `object_id`.

---

## Phase 5 — Chat & Prompting

1. Query understanding
   - Lightweight classifier or heuristics to detect modality intent (text/table/figure).

2. Context builder
   - Build a structured context block with buckets:
     - Text facts (top 6–10 chunks)
     - Table snippets (top 3–5 rows) with headers preserved
     - Figure captions (top 2–3) and links to assets
   - Enforce token budget; prefer diversity across sections and documents.

3. Prompt template
   - Provide precise, cite-by-`object_id` context in the prompt; instruct the LLM to reference `[[obj:…]]` in answers.

4. LLMs
   - Local: Ollama (e.g., `llama3` or `qwen2.5` for better reasoning).
   - Remote (optional): OpenAI/Mistral for comparison.

5. Output
   - Answer + citations (object_ids). Optionally render a short “evidence” appendix listing table rows and figure captions used.

---

## Phase 6 — Evaluation Plan

1. OCR fidelity
   - Per-page text length vs embedded text length; sample WER against hand-corrected ground truth for 10–20 paragraphs.
   - Table detection: count vs expected; manual spot-check of headers and unit extraction.

2. Retrieval quality
   - Synthetic query set per document covering: definitions, numeric values from tables, figure interpretation.
   - Metrics: Recall@k, MRR; manual judged precision for top-5.

3. Latency & cost
   - Per stage timing (PDF → RMMD, embedding, query latency) and optional remote augmentation usage.

4. Reporting
   - Save an evaluation report per doc (JSON + Markdown summary); keep screenshots of RMMD where issues are found.

---

## Phase 7 — Automation with n8n (Design)

Workflow A – Ingest (webhook)
- Trigger: Webhook receives PDF file or URL.
- Nodes:
  - Download/Filesystem: save input.
  - Execute Command: `python export_docling.py --pdf <path> --out out/`.
  - Execute Command: `python embed_index.py --rmmd out/<doc>.md`.
  - Respond to Webhook: return doc_id, stats, errors.

Workflow B – Q&A (webhook)
- Trigger: Webhook receives `{ query: string }`.
- Nodes:
  - Execute Command: `python answer.py --query "..."`.
  - Respond to Webhook: return `{ answer, citations: [object_id], chunks }`.

Note: Provide both workflows as JSON exports once the scripts above are in place.

---

## Phase 8 — MCP Config (for Codex)

Provide an MCP config exposing the project folder and optional local endpoints.

`mcp.json` (template)

```
{
  "clients": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "env": {
        "MCP_FS_ROOT": "./local-rag-project"
      }
    }
  ],
  "resources": [
    { "uri": "file://local-rag-project", "name": "Project Root" },
    { "uri": "file://local-rag-project/documents", "name": "Documents" },
    { "uri": "file://local-rag-project/out", "name": "RMMD Outputs" }
  ]
}
```

Notes
- If you later expose a small REST API (e.g., FastAPI for query/ingest), add an MCP OpenAPI server pointing to the local OpenAPI spec.

---

## Phase 9 — Roadmap & Risks

Short-term (Week 1)
- Implement `export_docling.py` (RMMD v0.2) and a thin `embed_index.py` using sentence-transformers + CLIP local models.
- Add `answer.py` retrieval + prompt builder; integrate with `chat.py` later.
- Produce 10–15 synthetic queries and perform first evaluation + manual RMMD QA.

Mid-term (Week 2)
- Enable full Docling pipeline (layout models) if downloads permitted; compare against minimal pipeline.
- Add VLM caption repair (Ollama or Mistral) only on low-confidence pages.
- Ship n8n workflow exports and MCP OpenAPI config if server is added.

Risks & Mitigations
- Model downloads blocked/SSL: keep minimal pipeline; support offline model cache; allow manual artifact placement.
- OCR variance on schematics: add ROI OCR and VLM-generated captions for diagrams.
- Table complexity: favor per-row linearization for retrieval; keep CSV blocks for auditability.

---

## Implementation Checklist

- [ ] `export_docling.py`: DoclingDocument → RMMD + assets
- [ ] `embed_index.py`: Build Chroma collections for text/table/image
- [ ] `answer.py`: Query routing, fusion, prompt builder, LLM call
- [ ] `n8n` exports: ingest and Q&A workflows
- [ ] `mcp.json`: finalize and test filesystem server

