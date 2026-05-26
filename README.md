# RAG Evaluation Benchmark

A systematic benchmark comparing seven RAG (Retrieval-Augmented Generation) pipeline architectures across five evaluation frameworks. Each iteration isolates one retrieval improvement — from naive fixed-size chunking up to query expansion with hierarchical semantic retrieval — so the performance delta of each technique can be measured clearly.

**Source document:** Paul Graham's autobiographical essay  
**LLM:** `claude-haiku-4-5-20251001` (Anthropic)  
**Dataset:** Combined golden + synthetic Q&A pairs (`data/combined_dataset.json`)

---

## RAG Iterations

| # | Iteration | Chunking | Retrieval Strategy |
|---|-----------|----------|--------------------|
| 1 | `basic_rag` | Fixed-size character (500 / 50 overlap) | Top-k=3 dense vector search |
| 2 | `semantic_rag` | SemanticChunker (90th percentile) | Top-k=3 dense vector search |
| 3 | `reranker_rag` | SemanticChunker (90th percentile) | Top-k=25 dense → BGE rerank → top-3 |
| 4 | `parent_child_rag` | Two-layer char split (1500 parent / 200 child) | Top-k=25 dense on children → BGE rerank → context swap to parents |
| 5 | `parent_child_semantic_rag` | Two-layer semantic (95th percentile parent / 90th child) | Top-k=25 dense on children → BGE rerank → context swap to parents |
| 6 | `sub_query_rag` | Two-layer semantic (same as #5) | Sub-query expansion → multi-query dense on children → BGE rerank → context swap |
| 7 | `sub_query_semantic_rag` | Single-layer SemanticChunker (90th percentile) | Sub-query expansion → multi-query dense → BGE rerank (no parent swap) |
| 8 | `hyde_rag` | Two-layer semantic (same as #5) | HyDE generation → hypothetical-vector child sweep → BGE rerank → context swap |

### Pipeline Descriptions

**Basic RAG** — Baseline. Text is split into fixed 500-character chunks with 50-character overlap. LlamaIndex in-memory vector store with local BAAI/bge-small-en-v1.5 embeddings.

**Semantic RAG** — Replaces fixed splitting with LangChain's `SemanticChunker`, which segments text at cosine-similarity breakpoints between sentences (90th-percentile threshold). Switches to OpenAI `text-embedding-3-small`.

**Reranker RAG** — Adds a BGE Cross-Encoder (`BAAI/bge-reranker-large`) reranking stage on top of semantic chunking. Retrieves 25 candidates, scores each query-chunk pair, and keeps the top 3.

**Parent-Child RAG** — Introduces a two-layer hierarchy using `RecursiveCharacterTextSplitter`. Small child chunks (200 chars) are embedded for precise matching; winning children are swapped for their larger parent chunk (1500 chars) before generation to give the LLM broader context.

**Parent-Child Semantic RAG** — Same small-to-large retrieval pattern but with semantic boundaries at both layers (95th percentile for parents, 90th for children), producing more coherent context windows.

**Sub-Query RAG** — Adds a query transformation stage. Claude Haiku decomposes the user query into 2–3 focused sub-queries via forced Anthropic Tool Use (Pydantic-validated JSON). Multi-query dense search expands semantic coverage; BGE Cross-Encoder re-ranks against the original query before the parent context swap.

**Sub-Query Semantic RAG** — Same sub-query expansion as above but on a flat single-layer semantic index (no parent-child hierarchy). Top-4 semantic chunks are passed directly to the LLM.

**HyDE RAG** — Adds Hypothetical Document Embeddings. Before any vector retrieval, Claude Haiku generates a hypothetical autobiographical paragraph answering the query; that paragraph is embedded and used as the vector query (bridging the query-document lexical gap). The original query is still used for reranking and generation.

---

## Evaluation Frameworks

Every iteration is evaluated by all five frameworks in the same run:

| Framework | Metrics |
|-----------|---------|
| **RAGAS** | `faithfulness`, `answer_relevancy`, `context_precision` |
| **DeepEval** | `Faithfulness`, `AnswerRelevancy`, `ContextualPrecision` |
| **LangSmith** | `correctness` (QA eval), `relevance` (criteria eval) |
| **Arize Phoenix** | `faithfulness`, `correctness` |
| **TruLens** | `answer_relevance`, `context_relevance`, `groundedness` |

All frameworks use `claude-haiku-4-5-20251001` as the judge LLM.

---

## Project Structure

```
.
├── data/
│   ├── paul_graham_essay.txt         # Source document
│   ├── paul_graham_golden_dataset.json
│   ├── synthetic_dataset.json
│   └── combined_dataset.json         # Used by all eval scripts
│
├── results/
│   ├── comparison.json               # Cross-iteration metric summary
│   ├── basic_rag/
│   │   ├── rag_results.json
│   │   ├── metrics_ragas.json
│   │   ├── metrics_deepeval.json
│   │   ├── metrics_langsmith.json
│   │   ├── metrics_arize_phoenix.json
│   │   ├── metrics_trulens.json
│   │   └── summary.json
│   └── <iteration>/                  # Same structure for every other iteration
│
├── prompts/
│   └── <iteration>/                  # Runtime prompt templates saved per run
│       ├── pipeline_prompts.json
│       ├── initial_prompt.md
│       ├── eval_framework_prompts.md
│       └── conversation_prompts.md
│
├── download_data.py                  # Download essay + generate golden dataset
├── basic_rag_eval.py
├── semantic_rag_eval.py
├── reranker_rag_eval.py
├── parent_child_rag_eval.py
├── parent_child_semantic_rag_eval.py
├── sub_query_rag_eval.py
├── sub_query_semantic_rag_eval.py
└── hyde_rag_eval.py
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install llama-index llama-index-llms-anthropic llama-index-embeddings-huggingface \
    langchain langchain-anthropic langchain-openai langchain-experimental \
    langchain-community sentence-transformers \
    ragas deepeval langsmith arize-phoenix trulens trulens-providers-litellm \
    anthropic python-dotenv pandas
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...            # Required for semantic/reranker/parent-child/sub-query/hyde variants
LANGSMITH_API_KEY=ls__...        # Optional — enables LangSmith tracing
```

### 3. Download data

```bash
python download_data.py
```

This downloads the Paul Graham essay and generates the golden Q&A dataset.

---

## Running Evaluations

Run any iteration individually:

```bash
python basic_rag_eval.py
python semantic_rag_eval.py
python reranker_rag_eval.py
python parent_child_rag_eval.py
python parent_child_semantic_rag_eval.py
python sub_query_rag_eval.py
python sub_query_semantic_rag_eval.py
python hyde_rag_eval.py
```

Control the number of evaluation samples:

```bash
EVAL_LIMIT=5 python basic_rag_eval.py
```

Each script appends its results to `results/comparison.json`, so running all iterations builds a full cross-comparison table.

---

## Results

Per-iteration results are written to `results/<iteration>/`:

- `rag_results.json` — raw Q&A pairs with retrieved contexts
- `metrics_<framework>.json` — scores from each evaluation framework
- `summary.json` — all metrics + run config in one file

`results/comparison.json` accumulates metrics and per-question answers across all iterations for side-by-side comparison.

---

## Models

| Role | Model |
|------|-------|
| LLM (generation + eval judge) | `claude-haiku-4-5-20251001` |
| Embeddings (basic_rag) | `BAAI/bge-small-en-v1.5` (local, HuggingFace) |
| Embeddings (all others) | `text-embedding-3-small` (OpenAI) |
| Reranker | `BAAI/bge-reranker-large` (local, sentence-transformers) |
