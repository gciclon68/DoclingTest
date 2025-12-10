## CHANGELOG

**Last Updated:** 2025-12-07 17:35:00  
**Version:** 1.0  
**Changes:**  
- Initial comprehensive documentation created
- Organized from initial RAG system design session

---

# CLEAR IMPLEMENTATION PLAN - Multimodal RAG System

## Current Status: What We Have NOW

### ✅ COMPLETED
1. **Docling Pipelines** - 7 pipelines configured with numbers
2. **JSON Output** - Files now named: `{doc}_p{N}-{pipeline}_docling.json`
   - Example: `clase10_p7-formulaaware_docling.json`
3. **Formula Detection** - Counting works, display in MD improved
4. **Base64 Images** - Images embedded in JSON (NOT embeddings yet!)
5. **Table Structure** - Both grid data AND images in JSON

### ❌ NOT DONE YET
1. **NO embeddings** - Text, images, tables NOT embedded
2. **NO chunking** - Text not split into chunks
3. **NO vector database** - Chroma not set up
4. **NO retrieval** - Can't search yet
5. **NO RAG** - No LLM integration

---

## PHASE 1: Document Processing (DONE ✅)

**Goal:** Extract structured data from PDFs

**What happens:**
```
PDF → Docling Pipeline → JSON with base64 images
```

**Output example:**
- `clase10_p2-standard_docling.json` - With tables
- `clase10_p7-formulaaware_docling.json` - With formulas
- `clase10_p6-visionllm_docling.json` - With AI descriptions

**Files in JSON:**
- `texts[]` - All paragraphs, headers, formulas
- `pictures[]` - Images as base64
- `tables[]` - Grid structure + image

---

## PHASE 2: Embedding Generation (NEXT STEP)

**Goal:** Convert JSON data into vector embeddings

### Step 2.1: Setup Models

**Install dependencies:**
```bash
pip install sentence-transformers chromadb pillow torch torchvision
```

**Models to download:**
1. **Text Model:** `all-MiniLM-L6-v2` (22MB, sentence-transformers)
2. **Image Model:** `clip-ViT-B-32` (350MB, CLIP)

**Why these models:**
- MiniLM: Fast, good quality, works offline
- CLIP: Multimodal (text+image), industry standard

### Step 2.2: Text Embedding

**Input:** `texts[]` array from JSON
**Output:** Vector embeddings in Chroma

**Process:**
```python
from sentence_transformers import SentenceTransformer

# Load model (runs locally, no API)
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# For each text element
for text in doc['texts']:
    if text['label'] in ['text', 'section_header']:
        # Create embedding (384 dimensions)
        embedding = text_model.encode(text['text'])

        # Store in Chroma with metadata
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[text['text']],
            metadatas=[{
                'doc_id': 'clase10',
                'page': text['prov'][0]['page_no'],
                'type': text['label']
            }],
            ids=[f"text_{text['self_ref'].split('/')[-1]}"]
        )
```

**File:** `scripts/embed_text.py`

**Chroma Collection:** `text_chunks`

**Storage size:** ~1KB per text chunk

### Step 2.3: Image Embedding

**Input:** `pictures[]` array from JSON (base64)
**Output:** CLIP embeddings in Chroma

**Process:**
```python
from sentence_transformers import SentenceTransformer
from PIL import Image
import base64
from io import BytesIO

# Load CLIP model (runs locally)
clip_model = SentenceTransformer('clip-ViT-B-32')

# For each picture
for pic in doc['pictures']:
    # Decode base64 to image
    base64_data = pic['image']['uri'].split(',')[1]
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_bytes))

    # Create embedding (512 dimensions)
    embedding = clip_model.encode(image)

    # Store in Chroma
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[pic.get('captions', [''])[0]],  # Caption if available
        metadatas=[{
            'doc_id': 'clase10',
            'page': pic['prov'][0]['page_no'],
            'type': 'figure',
            'width': pic['image']['size']['width'],
            'height': pic['image']['size']['height']
        }],
        ids=[f"img_{pic['self_ref'].split('/')[-1]}"]
    )
```

**File:** `scripts/embed_images.py`

**Chroma Collection:** `image_chunks`

**Storage size:** ~2KB per image

### Step 2.4: Table Embedding (DUAL APPROACH)

**Tables get TWO embeddings:**

#### A. Text Embedding (for semantic search)
```python
def linearize_table(table):
    """Convert table grid to searchable text."""

    caption = table.get('captions', ['Table'])[0]
    grid = table['data']['grid']

    # CSV format
    rows = []
    for row in grid:
        row_text = ','.join([cell['text'] for cell in row])
        rows.append(row_text)
    csv_text = '\n'.join(rows)

    # Combined: Caption + CSV
    full_text = f"{caption}\n\n{csv_text}"
    return full_text

# Embed with text model
table_text = linearize_table(table)
text_embedding = text_model.encode(table_text)

# Store in table_text collection
```

**Chroma Collection:** `table_text_chunks`

#### B. Image Embedding (for visual search)
```python
# Decode table image (same as pictures)
if 'image' in table:
    base64_data = table['image']['uri'].split(',')[1]
    img_bytes = base64.b64decode(base64_data)
    table_image = Image.open(BytesIO(img_bytes))

    # Embed with CLIP
    image_embedding = clip_model.encode(table_image)

    # Store in table_image collection
```

**Chroma Collection:** `table_image_chunks`

### Step 2.5: Formula Handling

**Challenge:** Formulas usually have empty text

**Solution A: Skip for now** (simplest)
```python
# Only embed formulas with text
if formula['text'].strip():
    embedding = text_model.encode(formula['text'])
```

**Solution B: Vision LLM descriptions** (best quality)
```python
# If processed with Vision LLM pipeline
if vision_llm_descriptions:
    formula_desc = vision_llm_descriptions[formula_id]
    embedding = text_model.encode(formula_desc)
```

**Recommendation:** Use Solution A initially, upgrade to B later

---

## PHASE 3: Vector Database Setup

**Technology:** ChromaDB (local, persistent)

### Database Structure

```python
import chromadb

# Create persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Collection 1: Text chunks
text_collection = client.get_or_create_collection(
    name="text_chunks",
    metadata={"hnsw:space": "cosine"}  # Similarity metric
)

# Collection 2: Images
image_collection = client.get_or_create_collection(
    name="image_chunks",
    metadata={"hnsw:space": "cosine"}
)

# Collection 3: Table text
table_text_collection = client.get_or_create_collection(
    name="table_text_chunks",
    metadata={"hnsw:space": "cosine"}
)

# Collection 4: Table images
table_image_collection = client.get_or_create_collection(
    name="table_image_chunks",
    metadata={"hnsw:space": "cosine"}
)
```

### Directory Structure
```
chroma_db/
├── text_chunks/
│   ├── index/
│   └── metadata.json
├── image_chunks/
│   ├── index/
│   └── metadata.json
├── table_text_chunks/
└── table_image_chunks/
```

**Storage estimate:**
- 100-page doc with 500 text chunks: ~500KB
- 20 images: ~40KB
- 10 tables (dual): ~60KB
- **Total per document: ~600KB-1MB**

---

## PHASE 4: Retrieval System

**Goal:** Search across all collections and fuse results

### Query Flow

```
User Query: "What is the RL circuit equation?"
    ↓
[1. Query Analysis]
    Detect intent: formula + circuit
    ↓
[2. Multi-Collection Search]
    ├─ Text search: "RL circuit equation"
    ├─ Image search: "circuit diagram"
    └─ Formula search: "equation"
    ↓
[3. Reciprocal Rank Fusion]
    Combine results with scores
    ↓
[4. Re-ranking]
    Sort by relevance
    ↓
[5. Context Assembly]
    Build LLM context with citations
```

### Implementation

**File:** `scripts/retriever.py`

```python
class MultimodalRetriever:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.text_collection = self.client.get_collection("text_chunks")
        self.image_collection = self.client.get_collection("image_chunks")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.clip_model = SentenceTransformer('clip-ViT-B-32')

    def query(self, query_text, top_k=10, filters=None):
        """Perform multimodal search."""

        results = {
            'text': [],
            'images': [],
            'tables': []
        }

        # Text search
        text_results = self.text_collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filters
        )
        results['text'] = text_results

        # Image search (text-to-image with CLIP)
        query_embedding = self.clip_model.encode(query_text)
        image_results = self.image_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max(3, top_k//3),
            where=filters
        )
        results['images'] = image_results

        # Fusion
        fused = self._fuse_results(results)
        return fused[:top_k]

    def _fuse_results(self, results, k=60):
        """Reciprocal Rank Fusion algorithm."""
        scores = {}
        for collection_results in results.values():
            for rank, item in enumerate(collection_results, 1):
                item_id = item['id']
                if item_id not in scores:
                    scores[item_id] = {'item': item, 'score': 0}
                scores[item_id]['score'] += 1.0 / (k + rank)

        ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [r['item'] for r in ranked]
```

---

## PHASE 5: RAG Integration

**Goal:** Connect retriever to LLM for answers

### LLM Options

| Option | Cost | Speed | Quality | Offline |
|--------|------|-------|---------|---------|
| **Ollama (llama3)** | Free | Fast | Good | ✅ Yes |
| **OpenAI GPT-4** | $$$ | Fast | Excellent | ❌ No |
| **OpenAI GPT-3.5** | $ | Very Fast | Good | ❌ No |

**Recommendation:** Start with Ollama (free, local)

### RAG Implementation

**File:** `scripts/rag_chat.py`

```python
from retriever import MultimodalRetriever
import requests  # For Ollama

class RAGChat:
    def __init__(self, db_path="./chroma_db", llm="ollama"):
        self.retriever = MultimodalRetriever(db_path)
        self.llm = llm

    def ask(self, question, top_k=5, include_images=True):
        """Ask a question and get RAG answer."""

        # 1. Retrieve relevant chunks
        results = self.retriever.query(question, top_k=top_k)

        # 2. Build context
        context = self._build_context(results)

        # 3. Create prompt
        prompt = f"""Answer the question based on the following context.
Use [[text-N]], [[table-N]], [[fig-N]] to cite sources.

Context:
{context}

Question: {question}

Answer:"""

        # 4. Call LLM
        if self.llm == "ollama":
            response = self._call_ollama(prompt)
        else:
            response = self._call_openai(prompt)

        # 5. Extract citations
        citations = self._extract_citations(response, results)

        return {
            'answer': response,
            'citations': citations,
            'sources': results
        }

    def _call_ollama(self, prompt):
        """Call local Ollama API."""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()['response']

    def _build_context(self, results):
        """Assemble context from retrieved chunks."""
        context_parts = []
        for i, result in enumerate(results, 1):
            if result['type'] == 'text':
                context_parts.append(f"[[text-{i}]] {result['text']}")
            elif result['type'] == 'figure':
                context_parts.append(f"[[fig-{i}]] {result['caption']}")
            elif result['type'] == 'table':
                context_parts.append(f"[[table-{i}]] {result['text'][:200]}...")

        return "\n\n".join(context_parts)
```

---

## N8N Workflow Integration

### Should You Use N8N?

**Pros:**
- ✅ Visual workflow editor
- ✅ Easy to modify and debug
- ✅ Can integrate with external systems
- ✅ Good for production deployment
- ✅ Built-in scheduling and monitoring

**Cons:**
- ❌ Additional complexity
- ❌ Learning curve
- ❌ Overkill for initial development

**Recommendation:** **Start WITHOUT N8N, add it later**

### When to Add N8N

**Use N8N when you have:**
1. **Multiple document sources** (Google Drive, S3, email)
2. **Scheduled processing** (nightly batch jobs)
3. **External integrations** (Slack notifications, webhooks)
4. **Team collaboration** (non-technical users need to modify)
5. **Production deployment** (need monitoring, error handling)

**Don't use N8N for:**
- Initial development and testing
- Single-user local workflows
- Simple command-line processing

### Example N8N Workflow (Future)

```
[1. Watch Folder] → [2. Detect Pipeline] → [3. Run Docling] → [4. Embed] → [5. Update Chroma] → [6. Notify Slack]
```

**N8N MCP Server:**
- Yes, Claude Code CAN create N8N workflows
- But you don't have the MCP server installed yet
- Not needed for Phase 1-5

**When you're ready for N8N:**
1. Install n8n: `npm install -g n8n`
2. Start: `n8n start`
3. Open: `http://localhost:5678`
4. I can help create the workflow JSON

---

## COMPLETE EMBEDDING & DATABASE PLAN

### Object Types → Models → Storage

| Object Type | Source | Embedding Model | Dimensions | Collection | Why This Model |
|-------------|--------|----------------|------------|------------|----------------|
| **Text** | `texts[]` | sentence-transformers/all-MiniLM-L6-v2 | 384 | `text_chunks` | Fast, good quality, small (22MB) |
| **Images** | `pictures[]` base64 | sentence-transformers/clip-ViT-B-32 | 512 | `image_chunks` | Multimodal, industry standard |
| **Tables (text)** | `tables[]` grid | sentence-transformers/all-MiniLM-L6-v2 | 384 | `table_text_chunks` | Same as text for consistency |
| **Tables (image)** | `tables[]` image | sentence-transformers/clip-ViT-B-32 | 512 | `table_image_chunks` | Visual understanding |
| **Formulas** | `texts[]` label=formula | Skip (empty) or Vision LLM descriptions | 384 | `text_chunks` | Most formulas empty |

### Storage Details

**ChromaDB:**
- **Type:** SQLite + HNSW index (local, persistent)
- **Location:** `./chroma_db/`
- **Similarity:** Cosine distance
- **Compression:** Automatic

**Size per document:**
- Text chunks (500): ~500KB
- Images (20): ~40KB
- Tables (10): ~30KB
- **Total:** ~600KB/document

**Performance:**
- Query speed: <100ms for 10K chunks
- Insert speed: ~1000 chunks/second
- Scales to millions of chunks

---

## STEP-BY-STEP EXECUTION PLAN

### Week 1: Embeddings (Scripts Only)

**Day 1-2: Text Embedding**
```bash
# 1. Install
pip install sentence-transformers chromadb

# 2. Create script
# File: scripts/embed_text.py

# 3. Test
python scripts/embed_text.py --input out/clase10_p7-formulaaware_docling.json

# Expected output:
# ✓ Loaded 321 text elements
# ✓ Created 247 text embeddings
# ✓ Saved to chroma_db/text_chunks
```

**Day 3-4: Image Embedding**
```bash
# 1. Install
pip install pillow torch torchvision

# 2. Create script
# File: scripts/embed_images.py

# 3. Test
python scripts/embed_images.py --input out/clase10_p2-standard_docling.json

# Expected output:
# ✓ Loaded 13 images
# ✓ Decoded base64 images
# ✓ Created CLIP embeddings
# ✓ Saved to chroma_db/image_chunks
```

**Day 5: Table Embedding**
```bash
# Create script: scripts/embed_tables.py
# Dual embedding (text + image)
```

**Day 6-7: Integration & Testing**
```bash
# Single command to embed all
python scripts/embed_all.py --input out/*.json
```

### Week 2: Retrieval

**Day 1-2: Basic Retrieval**
```bash
# File: scripts/retriever.py
# Implement MultimodalRetriever class
```

**Day 3-4: Query Routing & Fusion**
```bash
# Add RRF algorithm
# Add query routing logic
```

**Day 5-7: Testing & Optimization**
```bash
# Test queries
# Measure Recall@K
# Tune parameters
```

### Week 3: RAG Integration

**Day 1-2: Ollama Setup**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Download model
ollama pull llama3

# Test
ollama run llama3 "Hello"
```

**Day 2-4: RAG Implementation**
```bash
# File: scripts/rag_chat.py
# Implement RAGChat class
```

**Day 5-7: UI (Optional)**
```bash
# Simple Streamlit UI
pip install streamlit
streamlit run app.py
```

---

## ABSOLUTE CLARITY SUMMARY

### What EXISTS Now ✅
1. JSON files with structure + base64 images
2. Numbered pipelines (p1-p7)
3. Formula detection
4. Table grid structure

### What DOESN'T Exist ❌
1. Vector embeddings (no models running)
2. Chroma database (empty)
3. Retrieval system (can't search)
4. RAG (no LLM integration)

### Next Immediate Actions 🎯

**THIS WEEK:**
1. Create `scripts/embed_text.py`
2. Install `sentence-transformers` and `chromadb`
3. Process ONE document to test
4. Verify Chroma database created

**NEXT WEEK:**
1. Add image embeddings
2. Build retriever
3. Test search queries

**WEEK 3:**
1. Install Ollama
2. Build RAG chat
3. Test end-to-end

### N8N Decision 💡

**NOW:** Skip N8N, use Python scripts
**LATER:** Add N8N when you need:
- Automated workflows
- External integrations
- Team collaboration

**I can help with N8N workflows when you're ready!**

---

## YOUR QUESTIONS ANSWERED

### Q: "Images embedded because of CLIP?"
**A:** NO. Docling embeds images as base64. CLIP will be used LATER to create vector embeddings.

### Q: "Text embedded yet?"
**A:** NO. JSON has text, but no vector embeddings. Need to run embedding scripts.

### Q: "Table as text or image?"
**A:** BOTH! Tables have grid (text) AND image. Embed both for best results.

### Q: "Which pipeline for what?"
**A:** See [PIPELINE_SELECTION_GUIDE.md](PIPELINE_SELECTION_GUIDE.md) - Full decision tree included.

### Q: "Use N8N?"
**A:** Later. Start with Python scripts first. N8N is for production/automation.

---

## Ready to Start?

**Create the first embedding script?**
```bash
# I can generate scripts/embed_text.py now
# Just say: "Create embed_text.py"
```

**Or test pipeline naming?**
```bash
# Reprocess a document to see new naming
python scripts/ingest.py
# Select any pipeline, check filename format
```

**What would you like to do first?**
