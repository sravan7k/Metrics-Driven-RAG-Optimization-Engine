"""
HyDE RAG Evaluation Pipeline
Iteration  : hyde_rag
Chunking   : Two-layer Parent-Child with Semantic Boundaries (identical to parent_child_semantic_rag)
             Parent: SemanticChunker(breakpoint_threshold_type="percentile", amount=95)
                     Applied to full document → coarser narrative sections.
             Child : SemanticChunker(breakpoint_threshold_type="percentile", amount=90)
                     Applied to each Parent chunk → finer semantic sub-segments.
             Only child chunks are embedded; parent texts are stored in-memory dict.
Query Transform: Hypothetical Document Embeddings (HyDE) via claude-haiku-4-5-20251001
             Pre-retrieval — generates a single realistic autobiographical paragraph
             answering the user query, then embeds THAT paragraph (not the original query)
             to drive the dense vector search.  The original query is preserved for
             Cross-Encoder re-ranking and final LLM generation.
Index      : LangChain InMemoryVectorStore (OpenAI text-embedding-3-small)
Retrieval  : Four-stage —
               1. HyDE Generation  (claude-haiku-4-5-20251001 → hypothetical doc paragraph)
               2. Hypothetical-Vector Child Sweep  (embed hyp-doc → top_k=25 dense search)
               3. BGE Cross-Encoder rerank of 25 child chunks vs. ORIGINAL query (top_n=4)
               4. Context Swap — for each top-4 child, retrieve its full parent chunk
                  via parent_id metadata; de-duplicate parents before joining.
Reranker   : BAAI/bge-reranker-large  (sentence-transformers CrossEncoder, local)
LLM        : claude-haiku-4-5-20251001  (Anthropic)
Embeddings : text-embedding-3-small  (OpenAI)
Frameworks : RAGAS | DeepEval | LangSmith | Arize Phoenix | TruLens
Dataset    : combined (golden + synthetic) → data/combined_dataset.json

Key distinction from parent_child_semantic_rag:
  A HyDE (Hypothetical Document Embeddings) pre-retrieval layer sits directly on top
  of the Hierarchical Semantic Parent-Child + BGE Cross-Encoder pipeline.  Before any
  vector retrieval occurs, Claude Haiku generates a hypothetical answer paragraph as if
  it were an autobiographical essay excerpt.  This paragraph is embedded with
  text-embedding-3-small and used as the vector query — bridging the query-document
  lexical gap.  The Cross-Encoder still re-ranks retrieved children against the ORIGINAL
  user query; the final LLM generation also uses the ORIGINAL query.  Only the retrieval
  vector is derived from the hypothetical document.

Pipeline:
  User Query
    → HyDE: claude-haiku-4-5-20251001 generates hypothetical autobiographical paragraph
    → Embed hypothetical paragraph with text-embedding-3-small
    → Dense vector search on child chunks using hypothetical embedding (top_k=25)
    → BGE Cross-Encoder re-rank 25 child chunks vs. ORIGINAL user query (top_n=4)
    → Context Swap: parent_id lookup → retrieve full parent chunk for each winner
    → De-duplicate parent chunks (multiple children may share one parent)
    → Combine unique parent texts → LLM generation window (ORIGINAL query)

Results are written to:
  results/hyde_rag/rag_results.json
  results/hyde_rag/metrics_<framework>.json
  results/hyde_rag/summary.json
  results/comparison.json          ← updated across iterations

Prompts are written to:
  prompts/hyde_rag/pipeline_prompts.json   ← RAG chain + HyDE prompts (runtime)
  prompts/hyde_rag/initial_prompt.md       ← static, committed separately
  prompts/hyde_rag/eval_framework_prompts.md
  prompts/hyde_rag/conversation_prompts.md
"""

import os
import json
import time
import random
import traceback
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import anthropic as _anthropic_sdk
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ITERATION                   = "hyde_rag"
PARENT_BREAKPOINT_TYPE      = "percentile"
PARENT_BREAKPOINT_AMOUNT    = 95     # higher threshold → fewer, larger semantic sections
CHILD_BREAKPOINT_TYPE       = "percentile"
CHILD_BREAKPOINT_AMOUNT     = 90     # lower threshold → finer semantic sub-segments
TOP_K_RETRIEVE              = 25     # broad initial child-chunk sweep (HyDE vector)
TOP_N_RERANK                = 4      # top children after BGE reranking (context-swap targets)
EVAL_LIMIT                  = int(os.getenv("EVAL_LIMIT", "10"))
ESSAY_PATH                  = Path("data/paul_graham_essay.txt")
GOLDEN_PATH                 = Path("data/combined_dataset.json")
RESULTS_DIR                 = Path(f"results/{ITERATION}")
PROMPTS_DIR                 = Path(f"prompts/{ITERATION}")
COMPARE_FILE                = Path("results/comparison.json")

ANTHROPIC_KEY               = os.environ["ANTHROPIC_API_KEY"]
OPENAI_KEY                  = os.environ["OPENAI_API_KEY"]
LANGSMITH_KEY               = os.getenv("LANGSMITH_API_KEY")

LLM_MODEL                   = "claude-haiku-4-5-20251001"
EMBED_MODEL                 = "text-embedding-3-small"
RERANKER_MODEL              = "BAAI/bge-reranker-large"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_FILE.parent.mkdir(exist_ok=True)

# ── HyDE system prompt ────────────────────────────────────────────────────────
HYDE_SYSTEM_PROMPT = (
    "You are a document generation assistant. Your task is to write a single, highly "
    "realistic, and cohesive paragraph that answers the following query as if it were an "
    "excerpt from an autobiographical essay by a well-known technology entrepreneur and "
    "essayist. Write with confidence and use plausible narrative details about personal "
    "experiences, formative moments, specific places, people, and intellectual observations "
    "that would naturally appear in such an essay. Match the tone: reflective, direct, and "
    "intellectually honest. Produce exactly one paragraph — no preamble, no title, no "
    "bullet points. This is a HYPOTHETICAL document used strictly for vector similarity "
    "matching and will never be shown to an end user as a factual answer."
)

# ── Cross-Encoder (loaded once at module level) ───────────────────────────────
print(f"Loading Cross-Encoder  {RERANKER_MODEL} …")
from sentence_transformers import CrossEncoder
_cross_encoder = CrossEncoder(RERANKER_MODEL)
print("  Cross-Encoder ready.")

# ── Anthropic client (shared across HyDE generation and eval helpers) ─────────
_anthropic_client = _anthropic_sdk.Anthropic(api_key=ANTHROPIC_KEY)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(filename: str, data) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {path}")


def _save_pipeline_prompts(hyde_system_prompt: str, qa_template: str) -> None:
    payload = {
        "description": "Prompt templates and component configs used in the hyde_rag pipeline.",
        "source": "extracted from hyde_rag_eval.py",
        "hyde_layer": {
            "description": (
                "HyDE (Hypothetical Document Embeddings) pre-retrieval step. "
                "Claude Haiku generates a single realistic autobiographical paragraph "
                "answering the user query. That paragraph is embedded with "
                "text-embedding-3-small and used as the vector query — "
                "the original user query is NOT used for vector retrieval."
            ),
            "model": LLM_MODEL,
            "system_prompt": hyde_system_prompt,
            "max_tokens": 256,
            "output_description": (
                "Single cohesive paragraph used solely as the embedding vector for "
                "dense child-chunk retrieval. Never returned to the end user."
            ),
        },
        "chunker": {
            "description": (
                "Two-layer Parent-Child (Small-to-Large) chunking — both layers use SemanticChunker. "
                "Parent chunks capture full narrative sections (95th-percentile semantic breaks) "
                "and are stored in an in-memory dict. "
                "Child chunks are finer semantic sub-segments (90th-percentile breaks) embedded "
                "into the vector store with parent_id metadata."
            ),
            "parent_splitter": {
                "type": "SemanticChunker",
                "embeddings": f"OpenAIEmbeddings(model='{EMBED_MODEL}')",
                "breakpoint_threshold_type": PARENT_BREAKPOINT_TYPE,
                "breakpoint_threshold_amount": PARENT_BREAKPOINT_AMOUNT,
                "scope": "Applied to full document — produces coarse, narrative-level sections.",
            },
            "child_splitter": {
                "type": "SemanticChunker",
                "embeddings": f"OpenAIEmbeddings(model='{EMBED_MODEL}')",
                "breakpoint_threshold_type": CHILD_BREAKPOINT_TYPE,
                "breakpoint_threshold_amount": CHILD_BREAKPOINT_AMOUNT,
                "scope": "Applied to each Parent chunk — produces fine semantic sub-segments.",
            },
            "index": (
                f"Only child chunks are embedded with OpenAIEmbeddings(model='{EMBED_MODEL}'). "
                "Each child chunk's metadata contains parent_id pointing back to its parent text. "
                "Vector search uses the HyDE hypothetical-doc embedding — not the original query."
            ),
        },
        "reranker": {
            "description": (
                "BGE Cross-Encoder — scores (ORIGINAL_QUERY, child_chunk) pairs. "
                "Crucially, re-ranking uses the ORIGINAL user query — not the HyDE doc — "
                "so the relevance signal measures true alignment with what the user asked."
            ),
            "model": RERANKER_MODEL,
            "top_k_retrieve": TOP_K_RETRIEVE,
            "top_n_rerank": TOP_N_RERANK,
            "rerank_query": "ORIGINAL user query (not the HyDE hypothetical document).",
        },
        "context_swap": {
            "description": (
                "For each top-n child chunk, parent_id metadata is used to look up the full "
                "parent text from the in-memory parent_store dict. Duplicate parent IDs are "
                "skipped. The unique parent texts are joined as the final LLM context window."
            ),
        },
        "templates": {
            "rag_qa_template": {
                "description": (
                    "Main QA prompt — receives the swapped-in parent chunk context "
                    "and the ORIGINAL user query (chat mode via ChatAnthropic)."
                ),
                "template": qa_template,
                "input_variables": ["context", "question"],
                "notes": (
                    "context = unique parent chunk texts joined with '\\n\\n---\\n\\n'. "
                    "question = raw ORIGINAL user query string (never the HyDE document)."
                ),
            },
        },
    }
    path = PROMPTS_DIR / "pipeline_prompts.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"  saved → {path}")


def _update_comparison(metrics: dict, rag_results: list) -> None:
    comparison = (
        json.loads(COMPARE_FILE.read_text()) if COMPARE_FILE.exists()
        else {"iterations": {}, "responses": {}}
    )
    comparison.setdefault("responses", {})
    comparison["last_updated"] = datetime.now(timezone.utc).isoformat()
    comparison["iterations"][ITERATION] = {
        "timestamp":     metrics["timestamp"],
        "config":        metrics["config"],
        "ragas":         metrics.get("ragas", {}),
        "deepeval":      metrics.get("deepeval", {}),
        "langsmith":     metrics.get("langsmith", {}),
        "arize_phoenix": metrics.get("arize_phoenix", {}),
        "trulens":       metrics.get("trulens", {}),
    }
    for r in rag_results:
        q = r["question"]
        if q not in comparison["responses"]:
            comparison["responses"][q] = {"reference": r["reference"], "answers": {}}
        comparison["responses"][q]["answers"][ITERATION] = r["answer"]
    COMPARE_FILE.write_text(json.dumps(comparison, indent=2))
    print(f"  comparison → {COMPARE_FILE}")


def _is_question(text: str) -> bool:
    stripped = text.strip()
    return (
        not stripped.startswith("#")
        and not stripped.startswith("**")
        and not stripped.startswith("##")
        and len(stripped) > 60
    )


def _invoke_with_backoff(chain, inputs: dict, max_retries: int = 6) -> str:
    """Invoke a LangChain chain with exponential backoff on 529 Overloaded errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err_str = str(e)
            if "529" in err_str or "overloaded" in err_str.lower():
                wait = min(2 ** attempt + random.uniform(0.0, 1.0), 60.0)
                print(f"    API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Anthropic API still overloaded after {max_retries} retries.")


def _anthropic_call_with_backoff(client, max_retries: int = 6, **kwargs) -> object:
    """Call client.messages.create(**kwargs) with exponential backoff on 529 errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            if "529" in err_str or "overloaded" in err_str.lower():
                wait = min(2 ** attempt + random.uniform(0.0, 1.0), 60.0)
                print(f"    API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Anthropic API still overloaded after {max_retries} retries.")


# ── Component 1: HyDE Generation Layer (Pre-Retrieval) ───────────────────────
def generate_hypothetical_document(user_query: str) -> str:
    """
    Generate a single realistic autobiographical paragraph as if answering user_query
    from a Paul Graham-style essay.  The paragraph is embedded — not the original query
    — to drive the dense vector search (HyDE technique).  Includes exponential backoff
    to handle 529 Overloaded responses during evaluation batch runs.
    """
    resp = _anthropic_call_with_backoff(
        _anthropic_client,
        model=LLM_MODEL,
        max_tokens=256,
        system=HYDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_query}],
    )
    return resp.content[0].text.strip()


# ── 1. Build Parent-Child Semantic Index ──────────────────────────────────────
def build_parent_child_semantic_index():
    """
    Two-layer semantic ingestion pipeline:

      Parent layer — SemanticChunker(95th percentile) on the full document.
        Produces fewer, larger sections aligned to narrative / topic-level shifts.
        Stored only in parent_store dict (not embedded).

      Child layer  — SemanticChunker(90th percentile) applied to EACH parent chunk.
        Produces finer semantic sub-segments topically coherent within their parent's
        scope.  Only child chunks are embedded into the vector store.
        Each child Document carries metadata['parent_id'] and metadata['child_id'].

    Returns (store, parent_store).
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    print("Building Parent-Child Semantic index …")
    text       = ESSAY_PATH.read_text()
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_KEY)

    # ── Parent layer: coarse semantic sections (95th-percentile threshold) ────
    print(f"  Running parent SemanticChunker (breakpoint={PARENT_BREAKPOINT_AMOUNT}th percentile) …")
    parent_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=PARENT_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=PARENT_BREAKPOINT_AMOUNT,
    )
    parent_docs  = parent_splitter.create_documents([text])
    parent_store = {f"parent_{i}": doc.page_content for i, doc in enumerate(parent_docs)}
    print(
        f"  parent chunks : {len(parent_docs)}  "
        f"(avg {sum(len(t) for t in parent_store.values()) // max(len(parent_docs), 1)} chars)"
    )

    # ── Child layer: fine semantic sub-segments within each parent (90th pct) ─
    print(f"  Running child SemanticChunker (breakpoint={CHILD_BREAKPOINT_AMOUNT}th percentile) on each parent …")
    child_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=CHILD_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=CHILD_BREAKPOINT_AMOUNT,
    )
    child_texts: list[str]  = []
    child_metas: list[dict] = []
    for i, parent_doc in enumerate(parent_docs):
        pid        = f"parent_{i}"
        child_docs = child_splitter.create_documents([parent_doc.page_content])
        for j, child_doc in enumerate(child_docs):
            child_texts.append(child_doc.page_content)
            child_metas.append({"parent_id": pid, "child_id": f"{pid}_child_{j}"})

    print(
        f"  child chunks  : {len(child_texts)}  "
        f"(avg {sum(len(t) for t in child_texts) // max(len(child_texts), 1)} chars)"
    )

    store = InMemoryVectorStore.from_texts(
        child_texts,
        embedding=embeddings,
        metadatas=child_metas,
    )
    print("  vector index built (child chunks only).")
    return store, parent_store


# ── 2. BGE Cross-Encoder Reranker ─────────────────────────────────────────────
def rerank_documents(original_query: str, retrieved_docs: list, top_n: int = 5) -> list:
    """
    Score each (ORIGINAL_QUERY, child_chunk) pair with BAAI/bge-reranker-large.
    The original user query is used here — NOT the HyDE hypothetical document —
    so relevance scores reflect true alignment with what the user actually asked.
    Returns the top_n child Documents sorted by descending cross-encoder score.
    """
    pairs  = [(original_query, doc.page_content) for doc in retrieved_docs]
    scores = _cross_encoder.predict(pairs)

    # ── Log HyDE-retrieval order with cross-encoder scores ────────────────────
    print(f"\n    ── HyDE-retrieval order (top_k={len(retrieved_docs)}, scored vs ORIGINAL query) ──")
    for i, (doc, score) in enumerate(zip(retrieved_docs, scores), 1):
        snippet  = doc.page_content[:80].replace("\n", " ")
        pid      = doc.metadata.get("parent_id", "?")
        child_id = doc.metadata.get("child_id", "?")
        print(f"      [{i:>2}] score={score:+.4f}  child_id={child_id}  parent_id={pid}  │  {snippet}…")

    scored     = sorted(
        zip(scores, range(len(retrieved_docs)), retrieved_docs),
        key=lambda x: x[0], reverse=True,
    )
    top_scored = scored[:top_n]

    # ── Log BGE re-ranked top-n ───────────────────────────────────────────────
    print(f"\n    ── BGE re-ranked top-{top_n} child chunks (vs ORIGINAL query) ──")
    for new_rank, (score, orig_rank, doc) in enumerate(top_scored, 1):
        snippet  = doc.page_content[:80].replace("\n", " ")
        pid      = doc.metadata.get("parent_id", "?")
        child_id = doc.metadata.get("child_id", "?")
        print(
            f"      [{new_rank}] score={score:+.4f}  orig_rank={orig_rank + 1:>2}"
            f"  child_id={child_id}  parent_id={pid}  │  {snippet}…"
        )

    return [doc for _, _, doc in top_scored]


# ── 3. Run RAG Pipeline ───────────────────────────────────────────────────────
def run_rag(store, parent_store: dict) -> list[dict]:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import OpenAIEmbeddings

    print("\nRunning HyDE RAG pipeline …")
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    questions  = data["queries"]
    references = data["responses"]
    qids       = [qid for qid, q in questions.items() if _is_question(q)][:EVAL_LIMIT]

    llm        = ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_KEY)

    qa_template = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely and accurately based only on the provided context."
    )
    _save_pipeline_prompts(HYDE_SYSTEM_PROMPT, qa_template)
    prompt       = ChatPromptTemplate.from_template(qa_template)
    answer_chain = prompt | llm | StrOutputParser()

    results = []
    for i, qid in enumerate(qids, 1):
        q = questions[qid]
        print(f"\n  [{i:>2}/{len(qids)}] {q[:72]}…")

        # ── Component 1: HyDE Generation ─────────────────────────────────────
        print("    [HyDE] Generating hypothetical document …")
        hyp_doc = generate_hypothetical_document(q)
        print(f"    [HyDE] Hypothetical document:\n      \"{hyp_doc[:350].replace(chr(10), ' ')}\"")
        time.sleep(0.5)

        # ── Component 2: HyDE-Driven Vector Search (Stage 1) ─────────────────
        # Embed the hypothetical paragraph — NOT the original query
        hyp_vec          = embeddings.embed_query(hyp_doc)
        broad_child_docs = store.similarity_search_by_vector(hyp_vec, k=TOP_K_RETRIEVE)
        child_ids        = [d.metadata.get("child_id", "?") for d in broad_child_docs]
        print(f"    [HyDE] Retrieved {len(broad_child_docs)} child chunks via hypothetical embedding")
        print(f"    [HyDE] Retrieved child_ids: {child_ids}")

        # ── Component 3: Cross-Encoder Re-rank vs ORIGINAL query (Stage 2) ───
        top_child_docs = rerank_documents(q, broad_child_docs, top_n=TOP_N_RERANK)

        # ── Component 4: Parent Context Swap & Final Inference ────────────────
        seen_pids:    list[str] = []
        parent_texts: list[str] = []
        print(f"\n    ── Parent context dispatched to Claude ──")
        for child_doc in top_child_docs:
            pid = child_doc.metadata.get("parent_id")
            if pid and pid not in seen_pids:
                seen_pids.append(pid)
                ptext = parent_store[pid]
                parent_texts.append(ptext)
                print(f"      {pid}  (len={len(ptext)} chars)")
                print(f"        child text  : {child_doc.page_content[:100].replace(chr(10), ' ')}…")
                print(f"        parent text : {ptext[:200].replace(chr(10), ' ')}…")

        print(f"    unique parent chunks dispatched: {len(parent_texts)}")

        context = "\n\n---\n\n".join(parent_texts)
        answer  = _invoke_with_backoff(answer_chain, {"context": context, "question": q})

        # Brief inter-question pause to stay within API rate limits
        time.sleep(1.5)

        results.append({
            "question":   q,
            "answer":     answer,
            "contexts":   parent_texts,
            "reference":  references.get(qid, ""),
            "hyde_doc":   hyp_doc,
            "child_ids":  child_ids,
            "parent_ids": seen_pids,
        })

    return results


# ── 4. RAGAS ──────────────────────────────────────────────────────────────────
def eval_ragas(rag_results: list[dict]) -> dict:
    print("\n── RAGAS ──")
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import OpenAIEmbeddings

        llm = LangchainLLMWrapper(ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_KEY))

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["reference"] or r["answer"],
            )
            for r in rag_results
        ]
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=emb),
                ContextPrecision(llm=llm),
            ],
        )
        df          = result.to_pandas()
        metric_cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy", "context_precision")]
        scores      = {c: round(float(df[c].mean()), 4) for c in metric_cols}
        print("  scores:", scores)
        return scores
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {"error": err.splitlines()[-1]}


# ── 5. DeepEval ───────────────────────────────────────────────────────────────
def eval_deepeval(rag_results: list[dict]) -> dict:
    print("\n── DeepEval ──")
    try:
        from deepeval import evaluate as deval
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
        )
        from deepeval.test_case import LLMTestCase
        from deepeval.models.base_model import DeepEvalBaseLLM

        class _Claude(DeepEvalBaseLLM):
            def __init__(self):
                self._c = _anthropic_sdk.Anthropic(api_key=ANTHROPIC_KEY)

            def load_model(self):
                return self._c

            def generate(self, prompt: str) -> str:
                r = _anthropic_call_with_backoff(
                    self._c,
                    model=LLM_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                time.sleep(0.5)
                return r.content[0].text

            async def a_generate(self, prompt: str) -> str:
                r = await _anthropic_sdk.AsyncAnthropic(api_key=ANTHROPIC_KEY).messages.create(
                    model=LLM_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text

            def get_model_name(self):
                return LLM_MODEL

        model      = _Claude()
        test_cases = [
            LLMTestCase(
                input=r["question"],
                actual_output=r["answer"],
                retrieval_context=r["contexts"],
                expected_output=r["reference"] or r["answer"],
            )
            for r in rag_results
        ]
        metrics = [
            FaithfulnessMetric(threshold=0.5, model=model, include_reason=False),
            AnswerRelevancyMetric(threshold=0.5, model=model, include_reason=False),
            ContextualPrecisionMetric(threshold=0.5, model=model, include_reason=False),
        ]
        from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
        result = deval(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
        )

        totals, counts = defaultdict(float), defaultdict(int)
        test_results   = getattr(result, "test_results", []) or []
        for tr in test_results:
            for md in getattr(tr, "metrics_data", []) or []:
                if getattr(md, "score", None) is not None:
                    totals[md.name] += md.score
                    counts[md.name] += 1

        scores = {k: round(totals[k] / counts[k], 4) for k in totals if counts[k]}
        print("  scores:", scores)
        return scores
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {"error": err.splitlines()[-1]}


# ── 6. LangSmith ──────────────────────────────────────────────────────────────
def eval_langsmith(rag_results: list[dict]) -> dict:
    """
    Uses LangChain string evaluators locally.
    If LANGSMITH_API_KEY is set, traces are also uploaded to LangSmith.
    Small sleeps between calls prevent 529 overload errors.
    """
    print("\n── LangSmith ──")
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_classic.evaluation import load_evaluator

        if LANGSMITH_KEY:
            os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_KEY)
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_PROJECT", f"rag-eval-{ITERATION}")

        llm     = ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY)
        qa_eval = load_evaluator("qa", llm=llm)
        rl_eval = load_evaluator(
            "criteria",
            criteria={"relevance": "Is the response directly relevant to and addresses the question asked?"},
            llm=llm,
        )

        qa_scores, rl_scores = [], []
        for r in rag_results:
            qa_scores.append(
                qa_eval.evaluate_strings(
                    prediction=r["answer"],
                    input=r["question"],
                    reference=r["reference"] or r["answer"],
                ).get("score", 0)
            )
            time.sleep(1.0)
            rl_scores.append(
                rl_eval.evaluate_strings(
                    prediction=r["answer"],
                    input=r["question"],
                ).get("score", 0)
            )
            time.sleep(1.0)

        note   = "(tracing enabled)" if LANGSMITH_KEY else "(no LANGSMITH_API_KEY – local only)"
        scores = {
            "correctness": round(sum(qa_scores) / len(qa_scores), 4),
            "relevance":   round(sum(rl_scores) / len(rl_scores), 4),
        }
        print(f"  scores: {scores}  {note}")
        return scores
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {"error": err.splitlines()[-1]}


# ── 7. Arize Phoenix ──────────────────────────────────────────────────────────
def eval_arize_phoenix(rag_results: list[dict]) -> dict:
    print("\n── Arize Phoenix ──")
    try:
        import pandas as pd
        from phoenix.evals.metrics import FaithfulnessEvaluator, CorrectnessEvaluator
        from phoenix.evals.llm import LLM
        from phoenix.evals import evaluate_dataframe

        llm = LLM(
            provider="anthropic",
            model=LLM_MODEL,
            sync_client_kwargs={"api_key": ANTHROPIC_KEY},
        )
        df = pd.DataFrame([
            {
                "input":     r["question"],
                "output":    r["answer"],
                "reference": r["reference"] or r["answer"],
                "context":   "\n\n---\n\n".join(r["contexts"]),
            }
            for r in rag_results
        ])
        result_df = evaluate_dataframe(
            df,
            evaluators=[FaithfulnessEvaluator(llm=llm), CorrectnessEvaluator(llm=llm)],
        )

        def _avg_score(col: str):
            if col not in result_df.columns:
                return None
            vals = [
                v.get("score")
                for v in result_df[col]
                if isinstance(v, dict) and v.get("score") is not None
            ]
            return round(sum(vals) / len(vals), 4) if vals else None

        scores = {
            "faithfulness": _avg_score("faithfulness_score"),
            "correctness":  _avg_score("correctness_score"),
        }
        print("  scores:", scores)
        return scores
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {"error": err.splitlines()[-1]}


# ── 8. TruLens ────────────────────────────────────────────────────────────────
def eval_trulens(rag_results: list[dict]) -> dict:
    """
    TruLens 2.x Anthropic support via the LiteLLM provider.
    Small sleeps between calls prevent rate-limit errors.
    """
    print("\n── TruLens ──")
    try:
        from trulens.core import TruSession
        from trulens.providers.litellm import LiteLLM

        os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
        TruSession()
        provider = LiteLLM(model_engine=f"anthropic/{LLM_MODEL}")

        def _score(fn, *args):
            try:
                v = fn(*args)
                return float(v[0]) if isinstance(v, tuple) else float(v)
            except Exception:
                return None

        ans_rel, ctx_rel, ground = [], [], []
        for r in rag_results:
            ctx = "\n\n---\n\n".join(r["contexts"])
            s   = _score(provider.relevance, r["question"], r["answer"])
            if s is not None:
                ans_rel.append(s)
            time.sleep(0.5)
            s = _score(provider.context_relevance, r["question"], ctx)
            if s is not None:
                ctx_rel.append(s)
            time.sleep(0.5)
            s = _score(provider.groundedness_measure_with_cot_reasons, ctx, r["answer"])
            if s is not None:
                ground.append(s)
            time.sleep(0.5)

        def _avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else None

        scores = {
            "answer_relevance":  _avg(ans_rel),
            "context_relevance": _avg(ctx_rel),
            "groundedness":      _avg(ground),
        }
        print("  scores:", scores)
        return scores
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {"error": err.splitlines()[-1]}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*64}")
    print(f"  HyDE RAG Evaluation  |  samples={EVAL_LIMIT}")
    print(f"  pre-retrieval: HyDE — claude-haiku generates hypothetical doc → embed")
    print(f"  parent: SemanticChunker(breakpoint={PARENT_BREAKPOINT_AMOUNT}th percentile)  [full document]")
    print(f"  child:  SemanticChunker(breakpoint={CHILD_BREAKPOINT_AMOUNT}th percentile)  [per parent chunk]")
    print(f"  embed={EMBED_MODEL}  top_k_retrieve={TOP_K_RETRIEVE}  top_n_rerank={TOP_N_RERANK}")
    print(f"  reranker={RERANKER_MODEL}  (scored vs ORIGINAL query)")
    print(f"{'='*64}\n")

    store, parent_store = build_parent_child_semantic_index()
    rag_results         = run_rag(store, parent_store)
    _save("rag_results.json", rag_results)

    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "iteration":                ITERATION,
            "chunker":                  "ParentChildSemanticChunker",
            "parent_breakpoint_type":   PARENT_BREAKPOINT_TYPE,
            "parent_breakpoint_amount": PARENT_BREAKPOINT_AMOUNT,
            "child_breakpoint_type":    CHILD_BREAKPOINT_TYPE,
            "child_breakpoint_amount":  CHILD_BREAKPOINT_AMOUNT,
            "embed_model":              EMBED_MODEL,
            "reranker_model":           RERANKER_MODEL,
            "top_k_retrieve":           TOP_K_RETRIEVE,
            "top_n_rerank":             TOP_N_RERANK,
            "eval_samples":             EVAL_LIMIT,
            "llm_model":                LLM_MODEL,
            "pre_retrieval_transform":  "HyDE (Hypothetical Document Embeddings)",
            "hyde_model":               LLM_MODEL,
            "hyde_max_tokens":          256,
            "rerank_query":             "ORIGINAL user query",
        },
        "ragas":         eval_ragas(rag_results),
        "deepeval":      eval_deepeval(rag_results),
        "langsmith":     eval_langsmith(rag_results),
        "arize_phoenix": eval_arize_phoenix(rag_results),
        "trulens":       eval_trulens(rag_results),
    }

    print("\n\nSaving results …")
    for fw in ("ragas", "deepeval", "langsmith", "arize_phoenix", "trulens"):
        _save(f"metrics_{fw}.json", all_metrics[fw])
    _save("summary.json", all_metrics)
    _update_comparison(all_metrics, rag_results)

    print(f"\n{'='*64}")
    print("  Final Metrics Summary")
    print(f"{'='*64}")
    for fw in ("ragas", "deepeval", "langsmith", "arize_phoenix", "trulens"):
        m = all_metrics[fw]
        print(f"\n  [{fw}]")
        if isinstance(m, dict):
            for k, v in m.items():
                print(f"    {k:<28} {v}")
        else:
            print(f"    {m}")


if __name__ == "__main__":
    main()
