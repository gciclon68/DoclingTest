## CHANGELOG

**Last Updated:** 2025-12-07 17:35:00  
**Version:** 1.0  
**Changes:**  
- Initial comprehensive documentation created
- Organized from initial RAG system design session

---

# Multimodal RAG System Design

## Table of Contents
1. [JSON Structure Documentation](#json-structure-documentation)
2. [Embedding Strategy](#embedding-strategy)
3. [Retrieval Architecture](#retrieval-architecture)
4. [Implementation Plan](#implementation-plan)

---

## JSON Structure Documentation

### What Each Pipeline Produces

Starting now, JSON files are named with pipeline suffix: `{document}_{pipeline}_docling.json`

Examples:
- `clase10_formula-aware_docling.json`
- `clase10_vision-llm_docling.json`
- `clase10_rapidocr_docling.json`

### Core JSON Structure (All Pipelines)

```json
{
  "schema_name": "DoclingDocument",
  "version": "1.8.0",
  "name": "document_name",
  "origin": {
    "mimetype": "application/pdf",
    "binary_hash": 518374136297649070,
    "filename": "document.pdf"
  },
  "furniture": {...},  // Page headers/footers
  "body": {...},       // Main document hierarchy
  "texts": [...],      // All text elements
  "pictures": [...],   // All images/figures
  "tables": [...],     // All tables
  "key_value_items": [...] // Optional metadata
}
```

### TEXT Elements (`texts` array)

**What You Get:**
```json
{
  "self_ref": "#/texts/0",
  "parent": {"$ref": "#/body"},
  "label": "text",  // or "section_header", "formula", "list_item", "page_footer"
  "prov": [{
    "page_no": 1,
    "bbox": {
      "l": 104.778,  // left
      "t": 303.04,   // top
      "r": 628.73,   // right
      "b": 191.18,   // bottom
      "coord_origin": "BOTTOMLEFT"
    },
    "charspan": [0, 35]
  }],
  "text": "The actual text content",
  "orig": "Original text before processing",
  "level": 1  // Hierarchy level for headers
}
```

**Label Types:**
- `text`: Regular paragraphs
- `section_header`: Headings (with `level` 1-6)
- `formula`: Mathematical formulas (often empty text)
- `list_item`: Bullet/numbered lists
- `page_footer`: Page numbers, footers

**For Embedding:**
- ✅ `text`: Full text content
- ✅ `prov`: Page number + position
- ✅ `label`: Element type
- ✅ `level`: Hierarchy for context

### PICTURE Elements (`pictures` array)

**What You Get:**
```json
{
  "self_ref": "#/pictures/0",
  "label": "picture",
  "prov": [{
    "page_no": 5,
    "bbox": {"l": 100, "t": 200, "r": 500, "b": 400, "coord_origin": "BOTTOMLEFT"}
  }],
  "image": {
    "mimetype": "image/png",
    "dpi": 144,
    "size": {"width": 1297.0, "height": 592.0},
    "uri": "data:image/png;base64,iVBORw0KG..." // BASE64 EMBEDDED!
  },
  "captions": [],      // Associated captions
  "references": [],    // Cross-references
  "footnotes": [],     // Related footnotes
  "annotations": [],   // Metadata annotations
  "children": [        // Nested elements (like formulas in slides)
    {"$ref": "#/texts/12"}
  ]
}
```

**For Embedding:**
- ✅ `image.uri`: BASE64-encoded PNG image (can decode and embed)
- ✅ `image.size`: Width/height for aspect ratio
- ✅ `prov`: Page + position
- ✅ `captions`: Text descriptions (if available)
- ⚠️ `children`: May contain formulas/text (check refs)

**Pipeline Differences:**
- **Simple/Standard/Formula-Aware**: Images without captions
- **Vision LLM**: Images WITH AI-generated captions (separate JSON file)
- **OCR Pipelines**: May include OCR'd text from images

### TABLE Elements (`tables` array)

**What You Get:**
```json
{
  "self_ref": "#/tables/0",
  "label": "table",
  "prov": [{
    "page_no": 2,
    "bbox": {...}
  }],
  "data": {
    "grid": [
      [{"text": "Header1"}, {"text": "Header2"}],
      [{"text": "Row1Col1"}, {"text": "Row1Col2"}]
    ],
    "num_rows": 2,
    "num_cols": 2
  },
  "captions": ["Table 1: Circuit Parameters"],
  "image": {  // Table as image
    "uri": "data:image/png;base64,..."
  }
}
```

**For Embedding:**
- ✅ `data.grid`: Structured cell content
- ✅ `captions`: Table title/description
- ✅ `num_rows/num_cols`: Table dimensions
- ✅ `image.uri`: Visual representation (for vision embeddings)
- ✅ `prov`: Page + position

**Linearization Options:**
1. **CSV-style**: "Header1,Header2\nRow1Col1,Row1Col2"
2. **Semantic**: "Table 1 shows Circuit Parameters with columns Header1 and Header2. Row 1: Header1 is Row1Col1, Header2 is Row1Col2"
3. **Hybrid**: Store both for flexible retrieval

### FORMULA Elements (in `texts` array with `label: "formula"`)

**What You Get:**
```json
{
  "self_ref": "#/texts/42",
  "label": "formula",
  "prov": [{
    "page_no": 7,
    "bbox": {...}
  }],
  "text": "",  // Usually EMPTY (OCR failed)
  "orig": ""
}
```

**Pipeline-Specific:**
- **Formula-Aware**: Detects formulas, text often empty
- **OCR Pipelines**: May have partial text (often wrong)
- **Vision LLM**: Can generate descriptions

**For Embedding:**
- ⚠️ Limited value (empty text)
- ✅ Can extract bbox and create formula image crops
- ✅ Best with Vision LLM descriptions

---

## Embedding Strategy

### 1. TEXT Embeddings

**Source:** `texts` array elements
**Model:** `sentence-transformers` (e.g., `all-MiniLM-L6-v2` or `bge-small-en-v1.5`)

**Chunking Strategy:**
```python
def create_text_chunks(doc_dict):
    chunks = []
    for text in doc_dict['texts']:
        if text['label'] in ['text', 'section_header', 'list_item']:
            chunk = {
                'id': f"{doc_name}_{text['self_ref']}",
                'text': text['text'],
                'metadata': {
                    'type': text['label'],
                    'page': text['prov'][0]['page_no'],
                    'bbox': text['prov'][0]['bbox'],
                    'doc_id': doc_name,
                    'level': text.get('level', 0)
                }
            }
            chunks.append(chunk)
    return chunks
```

**Embedding Process:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v1')
embeddings = model.encode([chunk['text'] for chunk in chunks])
```

**Storage:** Chroma collection `text_chunks`

### 2. TABLE Embeddings

**Source:** `tables` array
**Strategy:** Dual embedding (text + image)

#### A. Text-based Table Embedding
```python
def linearize_table(table):
    caption = table.get('captions', [''])[0]
    grid = table['data']['grid']

    # CSV format
    csv_text = "\n".join([",".join([cell['text'] for cell in row]) for row in grid])

    # Semantic format
    headers = [cell['text'] for cell in grid[0]]
    rows_text = []
    for row in grid[1:]:
        row_desc = " | ".join([f"{headers[i]}: {cell['text']}" for i, cell in enumerate(row)])
        rows_text.append(row_desc)

    semantic_text = f"{caption}\nColumns: {', '.join(headers)}\n" + "\n".join(rows_text)

    return {
        'csv': csv_text,
        'semantic': semantic_text,
        'combined': f"{caption}\n\n{csv_text}"
    }
```

**Store:**
- `table_text_chunks` collection: Linearized text embeddings
- `table_image_chunks` collection: Visual embeddings (see below)

#### B. Image-based Table Embedding
```python
import base64
from PIL import Image
from io import BytesIO

def decode_table_image(table):
    if 'image' in table and 'uri' in table['image']:
        # Extract base64 from data:image/png;base64,{data}
        base64_data = table['image']['uri'].split(',')[1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        return image
    return None
```

**Embedding:** Use CLIP for image embeddings

### 3. PICTURE/FIGURE Embeddings

**Source:** `pictures` array
**Model:** CLIP (`clip-ViT-B-32`)

**Strategy:** Multimodal (image + caption)

```python
from sentence_transformers import SentenceTransformer
import base64
from PIL import Image
from io import BytesIO

# CLIP model for image/text
clip_model = SentenceTransformer('clip-ViT-B-32')

def embed_picture(picture, captions_dict=None):
    # Decode base64 image
    base64_data = picture['image']['uri'].split(',')[1]
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))

    # Get caption (from Vision LLM if available)
    pic_id = picture['self_ref'].split('/')[-1]
    caption = captions_dict.get(f"figure{int(pic_id)+1}", "")

    # Create embeddings
    image_embedding = clip_model.encode(image, convert_to_tensor=True)

    if caption:
        text_embedding = clip_model.encode(caption, convert_to_tensor=True)
        # Store both for hybrid search
        return {
            'image_emb': image_embedding,
            'text_emb': text_embedding,
            'caption': caption
        }
    else:
        return {
            'image_emb': image_embedding,
            'text_emb': None,
            'caption': None
        }
```

**Storage:** Chroma collection `image_chunks`

### 4. FORMULA Embeddings

**Challenge:** Most formulas have empty text

**Solutions:**

#### Option A: Vision LLM Descriptions (Recommended)
```python
# Use Vision LLM pipeline to generate descriptions
# Store description as text embedding
if formula_description:
    formula_embedding = text_model.encode(formula_description)
```

#### Option B: Formula Image Crops
```python
def extract_formula_region(page_image, bbox):
    # Crop formula from page using bbox coordinates
    left, top, right, bottom = bbox['l'], bbox['t'], bbox['r'], bbox['b']
    formula_img = page_image.crop((left, top, right, bottom))
    return formula_img

# Embed with CLIP
formula_img_embedding = clip_model.encode(formula_img)
```

#### Option C: Skip Empty Formulas
```python
# Only embed formulas with text content
if formula['text'].strip():
    formula_embedding = text_model.encode(formula['text'])
```

---

## Retrieval Architecture

### Vector Database Structure (Chroma)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Collection 1: Text chunks
text_collection = client.create_collection(
    name="text_chunks",
    metadata={"description": "Document text paragraphs and sections"}
)

# Collection 2: Table text
table_text_collection = client.create_collection(
    name="table_chunks",
    metadata={"description": "Linearized table content"}
)

# Collection 3: Images (figures + formulas)
image_collection = client.create_collection(
    name="image_chunks",
    metadata={"description": "CLIP embeddings of figures and diagrams"}
)

# Optional Collection 4: Table images
table_image_collection = client.create_collection(
    name="table_images",
    metadata={"description": "Visual table embeddings"}
)
```

### Metadata Schema

**Text Chunks:**
```python
metadata = {
    'doc_id': 'clase10',
    'page': 7,
    'type': 'text',  # or 'section_header', 'list_item'
    'level': 0,      # Hierarchy level
    'bbox': {'l': 100, 't': 200, 'r': 500, 'b': 400},
    'pipeline': 'formula-aware'
}
```

**Table Chunks:**
```python
metadata = {
    'doc_id': 'clase10',
    'page': 5,
    'type': 'table',
    'caption': 'Table 1: Circuit Parameters',
    'num_rows': 10,
    'num_cols': 4,
    'has_image': True,
    'pipeline': 'standard'
}
```

**Image Chunks:**
```python
metadata = {
    'doc_id': 'clase10',
    'page': 12,
    'type': 'figure',  # or 'formula'
    'caption': 'Figure 3: Power MOSFET cross-section',
    'has_caption': True,
    'width': 1297,
    'height': 592,
    'pipeline': 'vision-llm'
}
```

### Query Routing Strategy

```python
def route_query(query: str) -> dict:
    """Determine which collections to search based on query intent."""

    query_lower = query.lower()

    # Heuristics for routing
    visual_keywords = ['figure', 'diagram', 'image', 'graph', 'chart', 'schematic', 'show', 'picture']
    table_keywords = ['table', 'data', 'values', 'compare', 'row', 'column']
    formula_keywords = ['equation', 'formula', 'calculate', 'solve']

    routes = {
        'text': True,  # Always search text
        'tables': any(kw in query_lower for kw in table_keywords),
        'images': any(kw in query_lower for kw in visual_keywords),
        'formulas': any(kw in query_lower for kw in formula_keywords)
    }

    return routes
```

### Retrieval Process

```python
def retrieve_multimodal(query: str, top_k: int = 10):
    """Perform multimodal retrieval across all collections."""

    routes = route_query(query)
    all_results = []

    # 1. Text search (always)
    if routes['text']:
        text_results = text_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        all_results.extend(format_results(text_results, 'text'))

    # 2. Table search (if relevant)
    if routes['tables']:
        table_results = table_text_collection.query(
            query_texts=[query],
            n_results=max(3, top_k//3)
        )
        all_results.extend(format_results(table_results, 'table'))

    # 3. Image search (if relevant)
    if routes['images']:
        # Text-to-image search using CLIP
        query_embedding = clip_model.encode(query)
        image_results = image_collection.query(
            query_embeddings=[query_embedding],
            n_results=max(3, top_k//3)
        )
        all_results.extend(format_results(image_results, 'image'))

    # 4. Fusion: Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(all_results, k=60)

    # 5. De-duplicate by object_id
    deduplicated = deduplicate_results(fused_results)

    return deduplicated[:top_k]
```

### Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    """Combine multiple ranked lists using RRF."""

    scores = {}
    for results in results_lists:
        for rank, item in enumerate(results, 1):
            item_id = item['id']
            if item_id not in scores:
                scores[item_id] = {'item': item, 'score': 0}
            scores[item_id]['score'] += 1.0 / (k + rank)

    # Sort by score
    ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    return [r['item'] for r in ranked]
```

### Context Window Strategy

**Question:** *"Send chunks or entire chapter?"*

**Answer:** Hybrid approach

```python
def build_context(retrieved_items: list, max_tokens: int = 4000):
    """Build context for LLM with smart chunking."""

    context_parts = {
        'summaries': [],   # Chunk summaries
        'exact': [],       # Exact chunks
        'expanded': [],    # Expanded context
        'images': [],      # Image descriptions
        'tables': []       # Table snippets
    }

    for item in retrieved_items:
        metadata = item['metadata']

        if metadata['type'] == 'text':
            # For text: Include chunk + surrounding context
            exact_text = item['text']

            # Optional: Fetch surrounding chunks from same section
            if metadata.get('level', 0) > 0:  # Is a section
                section_text = get_section_context(metadata['doc_id'], metadata['page'])
                context_parts['expanded'].append(section_text)
            else:
                context_parts['exact'].append(exact_text)

        elif metadata['type'] == 'table':
            # For tables: Include caption + header + sample rows
            table_snippet = f"{metadata['caption']}\n{item['text'][:200]}..."
            context_parts['tables'].append(table_snippet)

        elif metadata['type'] == 'figure':
            # For images: Include caption + description
            image_desc = f"Figure on page {metadata['page']}: {metadata['caption']}"
            context_parts['images'].append(image_desc)

    # Assemble context with token budget
    final_context = assemble_with_budget(context_parts, max_tokens)
    return final_context
```

### Citation Strategy

```python
def format_context_with_citations(context_parts: dict) -> str:
    """Format context with citation markers."""

    formatted = "**Relevant Information:**\n\n"

    # Text chunks
    for i, chunk in enumerate(context_parts['exact'], 1):
        formatted += f"[[text-{i}]] {chunk}\n\n"

    # Tables
    for i, table in enumerate(context_parts['tables'], 1):
        formatted += f"[[table-{i}]] {table}\n\n"

    # Images
    for i, img in enumerate(context_parts['images'], 1):
        formatted += f"[[fig-{i}]] {img}\n\n"

    formatted += "\n**Instructions:** Use [[text-N]], [[table-N]], [[fig-N]] to cite sources in your answer."

    return formatted
```

---

## Implementation Plan

### Phase 1: JSON Processing & Embedding (Week 1)

**Files to Create:**
1. `scripts/embed_documents.py` - Main embedding pipeline
2. `scripts/models.py` - Model loading and configuration
3. `scripts/chunking.py` - Chunking strategies
4. `scripts/vectordb.py` - Chroma database management

**Steps:**
```bash
# 1. Process documents with different pipelines
python scripts/ingest.py  # Select Formula-Aware
python scripts/ingest.py  # Select Vision LLM

# 2. Create embeddings
python scripts/embed_documents.py \
    --input out/clase10_formula-aware_docling.json \
    --output chroma_db

# 3. Verify embeddings
python scripts/vectordb.py --list-collections
```

### Phase 2: Retrieval System (Week 2)

**Files to Create:**
1. `scripts/retriever.py` - Query routing and retrieval
2. `scripts/fusion.py` - Result fusion algorithms
3. `scripts/context_builder.py` - Context assembly

**API:**
```python
from retriever import MultimodalRetriever

retriever = MultimodalRetriever(db_path="./chroma_db")

results = retriever.query(
    "What is the formula for RL circuit transient response?",
    top_k=5,
    filters={'doc_id': 'clase10', 'page': [7, 8, 9]}
)

context = retriever.build_context(results, max_tokens=4000)
```

### Phase 3: RAG Chat Interface (Week 3)

**Files to Create:**
1. `scripts/chat.py` - LLM integration
2. `scripts/prompts.py` - Prompt templates
3. `scripts/evaluation.py` - Quality metrics

**Usage:**
```python
from chat import RAGChat

rag = RAGChat(
    retriever=retriever,
    llm="ollama/llama3",  # or "openai/gpt-4"
)

answer = rag.ask(
    "Explain the RL circuit transient behavior",
    include_citations=True,
    include_images=True
)

print(answer['text'])
print(answer['citations'])
print(answer['images'])
```

### Phase 4: Evaluation & Optimization (Week 4)

**Metrics:**
- Retrieval quality: Recall@K, MRR, NDCG
- Answer quality: Manual judging, BLEU/ROUGE scores
- Latency: Query time, embedding time
- Cost: Vision LLM API usage

**Test Queries:**
```python
test_queries = [
    "What is Ohm's law?",  # Text retrieval
    "Show the circuit diagram for RL transient",  # Image retrieval
    "Compare voltage values in Table 1",  # Table retrieval
    "What is the differential equation for LC circuit?"  # Formula retrieval
]
```

---

## Next Steps

1. **Immediate:** Test pipeline naming (reprocess one document)
2. **This Week:** Implement `embed_documents.py` for text chunks
3. **Next Week:** Add CLIP embeddings for images
4. **Future:** Build chat interface with citations

Would you like me to:
1. Create the `embed_documents.py` script?
2. Set up the Chroma database structure?
3. Build the retriever with query routing?
4. Design the chat interface prompts?

Let me know which component to tackle first!
