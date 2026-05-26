"""
Sub-Query Semantic RAG Evaluation Pipeline
Iteration  : sub_query_semantic_rag
Chunking   : Single-layer SemanticChunker (90th percentile) on the full document.
             Identical chunking strategy to reranker_rag — flat semantic segments,
             no parent-child hierarchy.
Query Transform: Sub-Query Expansion via claude-haiku-4-5-20251001 + Anthropic Tool Use
             Pydantic-validated JSON guarantees a list of 2–3 focused sub-queries.
Index      : LangChain InMemoryVectorStore (OpenAI text-embedding-3-small)
Retrieval  : Three-stage —
               1. Sub-Query Expansion  (2–3 sub-queries via Haiku forced tool_choice)
               2. Multi-Query Semantic Sweep  (top_k=15 per query; original + sub-queries)
                  → pool all hits → strict de-dup by chunk_id metadata
               3. BGE Cross-Encoder rerank of all de-duplicated chunks  (top_n=4)
                  → top-4 semantic chunks passed DIRECTLY to LLM (no parent swap)
Reranker   : BAAI/bge-reranker-large  (sentence-transformers CrossEncoder, local)
LLM        : claude-haiku-4-5-20251001  (Anthropic)
Embeddings : text-embedding-3-small  (OpenAI)
Frameworks : RAGAS | DeepEval | LangSmith | Arize Phoenix | TruLens
Dataset    : combined (golden + synthetic) → data/combined_dataset.json

Key distinction from sub_query_rag:
  The parent-child context-swap layer has been removed entirely. After Sub-Query
  Expansion and multi-query retrieval, de-duplicated semantic chunks are scored by
  the BGE Cross-Encoder against the ORIGINAL query. The top-4 highest-scoring
  SEMANTIC CHUNKS themselves (not their parent blocks) are passed directly into the
  LLM context window. This isolates the contribution of Sub-Query Expansion on top of
  a flat semantic chunking baseline, removing the confounding variable of parent-context
  inflation.

Pipeline:
  User Query
    → Sub-Query Expansion  (Haiku tool_choice → 2–3 focused sub-queries)
    → Multi-query dense vector search on semantic chunks  (top_k=15 per query)
    → Pool all hits → strict de-dup by chunk_id metadata
    → BAAI/bge-reranker-large CrossEncoder  (score all de-duped chunks vs. ORIGINAL query)
    → Sort descending, slice top_n=4 winning semantic chunks
    → Combine top-4 chunk texts → LLM generation window  (NO parent swap)

Results are written to:
  results/sub_query_semantic_rag/rag_results.json
  results/sub_query_semantic_rag/metrics_<framework>.json
  results/sub_query_semantic_rag/summary.json
  results/comparison.json          ← updated across iterations

Prompts are written to:
  prompts/sub_query_semantic_rag/pipeline_prompts.json   ← RAG chain + sub-query prompts (runtime)
  prompts/sub_query_semantic_rag/initial_prompt.md       ← static, committed separately
  prompts/sub_query_semantic_rag/eval_framework_prompts.md
  prompts/sub_query_semantic_rag/conversation_prompts.md
"""

import os
import json
import time
import random
import traceback
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ITERATION               = "sub_query_semantic_rag"
BREAKPOINT_TYPE         = "percentile"
BREAKPOINT_AMOUNT       = 90     # 90th-percentile semantic chunking (flat, single layer)
TOP_K_RETRIEVE          = 15     # semantic chunks retrieved per query (original + each sub-query)
TOP_N_RERANK            = 4      # top chunks after BGE reranking (passed directly to LLM)
EVAL_LIMIT              = int(os.getenv("EVAL_LIMIT", "10"))
ESSAY_PATH              = Path("data/paul_graham_essay.txt")
GOLDEN_PATH             = Path("data/combined_dataset.json")
RESULTS_DIR             = Path(f"results/{ITERATION}")
PROMPTS_DIR             = Path(f"prompts/{ITERATION}")
COMPARE_FILE            = Path("results/comparison.json")

ANTHROPIC_KEY           = os.environ["ANTHROPIC_API_KEY"]
OPENAI_KEY              = os.environ["OPENAI_API_KEY"]
LANGSMITH_KEY           = os.getenv("LANGSMITH_API_KEY")

LLM_MODEL               = "claude-haiku-4-5-20251001"
EMBED_MODEL             = "text-embedding-3-small"
RERANKER_MODEL          = "BAAI/bge-reranker-large"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_FILE.parent.mkdir(exist_ok=True)


# ── Cross-Encoder (loaded once at module level) ───────────────────────────────
print(f"Loading Cross-Encoder  {RERANKER_MODEL} …")
from sentence_transformers import CrossEncoder
_cross_encoder = CrossEncoder(RERANKER_MODEL)
print("  Cross-Encoder ready.")


# ── Component 1: Sub-Query Expansion (Pydantic model + Anthropic tool schema) ─
class SubQueryList(BaseModel):
    sub_queries: list[str]


SUB_QUERY_SYSTEM_PROMPT = (
    "You are an expert query decomposer for a Retrieval-Augmented Generation (RAG) system "
    "specialised on the essays of Paul Graham.\n\n"
    "Your task: analyse the user query and break it into 2–3 distinct, hyper-focused "
    "sub-queries that together cover its full information need.\n\n"
    "Rules\n"
    "-----\n"
    "1. Comparative queries (e.g. \"Compare Paul Graham's experience at Interleaf vs Viaweb\") "
    "MUST be split into one focused sub-query per subject:\n"
    "   example output: [\"Paul Graham Interleaf work experience\", "
    "\"Paul Graham Viaweb startup experience\"]\n"
    "2. Simple or narrow queries: return the original query plus ONE semantically distinct "
    "alternative phrasing — never repeat the same sentence verbatim.\n"
    "3. Each sub-query must be self-contained, specific, and optimised for dense vector "
    "retrieval (cosine similarity in embedding space).\n"
    "4. No two sub-queries may be semantically identical."
)

_SUB_QUERY_TOOL = {
    "name": "generate_sub_queries",
    "description": (
        "Decompose the user query into 2–3 hyper-focused sub-queries optimised for "
        "dense vector retrieval against a Paul Graham essay corpus. "
        "Each entry must be semantically distinct from the others."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of 2–3 focused sub-queries derived from the original user query. "
                    "Comparative queries are split per subject. "
                    "Simple queries yield the original plus one semantically distinct paraphrase."
                ),
                "minItems": 2,
                "maxItems": 3,
            }
        },
        "required": ["sub_queries"],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(filename: str, data) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {path}")


def _save_pipeline_prompts(qa_template: str) -> None:
    payload = {
        "description": "Prompts and configuration used in the sub_query_semantic_rag pipeline.",
        "source": "extracted from sub_query_semantic_rag_eval.py",
        "sub_query_expansion": {
            "description": (
                "Pre-retrieval query decomposition layer. Claude Haiku is invoked with a forced "
                "tool_choice (Anthropic Tool Use) so the model MUST return structured JSON — "
                "the same guarantee as OpenAI Structured Outputs, implemented via Pydantic "
                "validation of the tool_use block. Each user query is expanded into 2–3 "
                "semantically distinct sub-queries before any vector retrieval begins."
            ),
            "model": LLM_MODEL,
            "system_prompt": SUB_QUERY_SYSTEM_PROMPT,
            "tool": _SUB_QUERY_TOOL,
            "tool_choice": {"type": "tool", "name": "generate_sub_queries"},
            "output_model": "SubQueryList(sub_queries: list[str])",
            "sub_query_count": "2–3 distinct sub-queries per user query",
            "backoff": "exponential backoff on 529 Overloaded (shared with all Anthropic calls)",
        },
        "chunker": {
            "description": (
                "Single-layer flat semantic chunking — SemanticChunker(90th percentile). "
                "Applied directly to the full document to produce variable-length, "
                "topically coherent segments. No parent-child hierarchy is used. "
                "Each chunk is assigned a unique chunk_id in metadata for de-duplication."
            ),
            "type": "SemanticChunker",
            "embeddings": f"OpenAIEmbeddings(model='{EMBED_MODEL}')",
            "breakpoint_threshold_type": BREAKPOINT_TYPE,
            "breakpoint_threshold_amount": BREAKPOINT_AMOUNT,
            "scope": "Applied to full document — single layer, no parent tracking.",
            "index": (
                f"All semantic chunks embedded with OpenAIEmbeddings(model='{EMBED_MODEL}'). "
                "Each chunk's metadata contains a unique chunk_id."
            ),
        },
        "retrieval": {
            "strategy": (
                "Multi-query vector search: original query + each sub-query, top_k=15 each. "
                "All hits are pooled then strictly de-duplicated by chunk_id metadata "
                "to eliminate overlap from semantically overlapping sub-queries."
            ),
            "top_k_per_query": TOP_K_RETRIEVE,
            "queries_per_question": "1 original + 2–3 sub-queries = 3–4 total searches",
        },
        "reranker": {
            "description": (
                "BGE Cross-Encoder — scores all de-duplicated (query, semantic_chunk) pairs against "
                "the ORIGINAL user query only (not sub-queries). "
                "Top-n scoring semantic chunks are passed DIRECTLY into the LLM context window. "
                "No parent-context swap occurs."
            ),
            "model": RERANKER_MODEL,
            "top_k_retrieve": TOP_K_RETRIEVE,
            "top_n_rerank": TOP_N_RERANK,
            "rerank_against": "original user query only",
            "context_passed_to_llm": "top-n semantic chunks directly (no parent swap)",
        },
        "templates": {
            "rag_qa_template": {
                "description": (
                    "Main QA prompt — receives the top-n BGE-reranked semantic chunks "
                    "and the original user query (chat mode via ChatAnthropic)."
                ),
                "template": qa_template,
                "input_variables": ["context", "question"],
                "notes": (
                    "context = top-n semantic chunk texts joined with '\\n\\n---\\n\\n'. "
                    "question = raw original user query string (not sub-queries)."
                ),
            }
        },
    }
    path = PROMPTS_DIR / "pipeline_prompts.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"  saved → {path}")


def _update_comparison(metrics: dict, rag_results: list[dict]) -> None:
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


def _invoke_with_backoff(chain, inputs: dict, max_retries: int = 10) -> str:
    """Invoke a LangChain chain with exponential backoff on 529 Overloaded errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err_str = str(e)
            if "529" in err_str or "overloaded" in err_str.lower():
                wait = min(2 ** attempt + random.uniform(0.0, 1.0), 120.0)
                print(f"    API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Anthropic API still overloaded after {max_retries} retries.")


def _anthropic_call_with_backoff(client, max_retries: int = 10, **kwargs) -> object:
    """Call client.messages.create(**kwargs) with exponential backoff on 529 errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            if "529" in err_str or "overloaded" in err_str.lower():
                wait = min(2 ** attempt + random.uniform(0.0, 1.0), 120.0)
                print(f"    API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Anthropic API still overloaded after {max_retries} retries.")


def generate_sub_queries(user_query: str) -> list[str]:
    """
    Decompose user_query into 2–3 hyper-focused sub-queries using Claude Haiku.

    Mechanism: Anthropic Tool Use with tool_choice forced to 'generate_sub_queries'
    guarantees the model returns structured JSON (same guarantee as OpenAI Structured
    Outputs). The tool input is validated with Pydantic (SubQueryList) before returning.
    Exponential backoff protects against Anthropic 529 Overloaded errors.

    Example — comparative query:
      Input : "Compare Paul Graham's experience at Interleaf vs Viaweb"
      Output: ["Paul Graham Interleaf work experience",
               "Paul Graham Viaweb startup experience"]

    Example — simple query:
      Input : "What did Paul Graham learn from painting?"
      Output: ["lessons Paul Graham learned from painting",
               "Paul Graham painting influence on his thinking"]
    """
    from anthropic import Anthropic as _Anth
    client   = _Anth(api_key=ANTHROPIC_KEY)
    response = _anthropic_call_with_backoff(
        client,
        model=LLM_MODEL,
        max_tokens=512,
        system=SUB_QUERY_SYSTEM_PROMPT,
        tools=[_SUB_QUERY_TOOL],
        tool_choice={"type": "tool", "name": "generate_sub_queries"},
        messages=[{"role": "user", "content": user_query}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return SubQueryList(**tool_block.input).sub_queries


# ── 1. Build Flat Semantic Index ──────────────────────────────────────────────
def build_semantic_index():
    """
    Single-layer semantic ingestion pipeline.

    SemanticChunker(90th percentile) is applied directly to the full document,
    producing flat, variable-length semantic chunks with no hierarchy.
    Each chunk is embedded into the vector store with a unique chunk_id in metadata
    used for strict de-duplication in the multi-query retrieval step.

    Returns the InMemoryVectorStore.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    print("Building Flat Semantic index …")
    text       = ESSAY_PATH.read_text()
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_KEY)

    print(f"  Running SemanticChunker (breakpoint={BREAKPOINT_AMOUNT}th percentile) …")
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=BREAKPOINT_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_AMOUNT,
    )
    docs = splitter.create_documents([text])

    chunk_texts = [doc.page_content for doc in docs]
    chunk_metas = [{"chunk_id": f"chunk_{i}"} for i in range(len(docs))]

    avg_len = sum(len(t) for t in chunk_texts) // max(len(chunk_texts), 1)
    print(f"  semantic chunks : {len(chunk_texts)}  (avg {avg_len} chars)")

    store = InMemoryVectorStore.from_texts(
        chunk_texts,
        embedding=embeddings,
        metadatas=chunk_metas,
    )
    print("  vector index built.")
    return store


# ── 2. BGE Cross-Encoder Reranker ─────────────────────────────────────────────
def rerank_documents(query: str, retrieved_docs: list, top_n: int = 5) -> list:
    """
    Score each (original_query, semantic_chunk) pair with BAAI/bge-reranker-large.
    Always reranks against the ORIGINAL user query — not the sub-queries — so the
    re-ranking signal remains aligned to the user's actual intent.

    Logs:
      - All de-duplicated chunks with their cross-encoder scores.
      - The top-n winners with original pool index shown.
    Returns top_n Documents sorted by descending relevance score.
    """
    pairs  = [(query, doc.page_content) for doc in retrieved_docs]
    scores = _cross_encoder.predict(pairs)

    # ── Log all de-duplicated chunks with cross-encoder scores ────────────────
    print(f"\n    ── BGE Cross-Encoder: scoring {len(retrieved_docs)} de-duplicated chunks (vs. original query) ──")
    for i, (doc, score) in enumerate(zip(retrieved_docs, scores), 1):
        snippet  = doc.page_content[:80].replace("\n", " ")
        chunk_id = doc.metadata.get("chunk_id", "?")
        print(f"      [{i:>2}] score={score:+.4f}  chunk_id={chunk_id}  │  {snippet}…")

    # ── Sort by cross-encoder score descending ────────────────────────────────
    scored     = sorted(
        zip(scores, range(len(retrieved_docs)), retrieved_docs),
        key=lambda x: x[0], reverse=True,
    )
    top_scored = scored[:top_n]

    # ── Log BGE re-ranked top-n chunks ────────────────────────────────────────
    print(f"\n    ── BGE re-ranked top-{top_n} semantic chunks ──")
    for new_rank, (score, orig_idx, doc) in enumerate(top_scored, 1):
        snippet  = doc.page_content[:80].replace("\n", " ")
        chunk_id = doc.metadata.get("chunk_id", "?")
        print(
            f"      [{new_rank}] score={score:+.4f}  pool_idx={orig_idx + 1:>2}"
            f"  chunk_id={chunk_id}  │  {snippet}…"
        )

    return [doc for _, _, doc in top_scored]


# ── 3. Run RAG Pipeline ───────────────────────────────────────────────────────
def run_rag(store) -> list[dict]:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print("\nRunning Sub-Query Semantic RAG pipeline …")
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    questions  = data["queries"]
    references = data["responses"]
    qids       = [qid for qid, q in questions.items() if _is_question(q)][:EVAL_LIMIT]

    # Retriever returns top_k semantic chunks per query execution
    retriever = store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
    llm       = ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY)

    qa_template = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely and accurately based only on the provided context."
    )
    _save_pipeline_prompts(qa_template)
    prompt       = ChatPromptTemplate.from_template(qa_template)
    answer_chain = prompt | llm | StrOutputParser()

    results = []
    for i, qid in enumerate(qids, 1):
        q = questions[qid]
        print(f"\n  [{i:>2}/{len(qids)}] {q[:72]}…")
        print(f"    Original query : {q}")

        # ── Stage 1: Sub-Query Expansion ──────────────────────────────────────
        sub_queries = generate_sub_queries(q)
        print(f"    Sub-queries generated ({len(sub_queries)}):")
        for j, sq in enumerate(sub_queries, 1):
            print(f"      [{j}] {sq}")

        # ── Stage 2: Multi-Query Vector Search (top_k=15 per query) ──────────
        all_queries              = [q] + sub_queries
        raw_hits: list           = []
        per_query_counts: list[int] = []
        for sq in all_queries:
            hits = retriever.invoke(sq)
            per_query_counts.append(len(hits))
            raw_hits.extend(hits)

        print(f"    Semantic chunks retrieved:")
        print(f"      original query  : {per_query_counts[0]}")
        for j, cnt in enumerate(per_query_counts[1:], 1):
            sq_preview = sub_queries[j - 1][:55]
            print(f"      sub-query [{j}]   : {cnt}  ← {sq_preview}…")
        print(f"      total raw       : {len(raw_hits)}")

        # ── Strict de-duplication by chunk_id ─────────────────────────────────
        seen_cids: set[str]      = set()
        deduped_docs: list       = []
        for doc in raw_hits:
            cid = doc.metadata.get("chunk_id", str(id(doc)))
            if cid not in seen_cids:
                seen_cids.add(cid)
                deduped_docs.append(doc)
        print(f"      after dedup     : {len(deduped_docs)}")

        # ── Stage 3: BGE Cross-Encoder rerank vs. ORIGINAL query ──────────────
        top_docs = rerank_documents(q, deduped_docs, top_n=TOP_N_RERANK)

        # ── Stage 4: LLM generation on top-n semantic chunks directly ─────────
        # No parent swap — semantic chunks are passed straight to the LLM.
        chunk_texts = [doc.page_content for doc in top_docs]
        print(f"\n    ── Top-{TOP_N_RERANK} semantic chunks passed to LLM (no parent swap) ──")
        for idx, text in enumerate(chunk_texts, 1):
            chunk_id = top_docs[idx - 1].metadata.get("chunk_id", "?")
            print(f"      [{idx}] {chunk_id}  (len={len(text)} chars)  │  {text[:100].replace(chr(10), ' ')}…")

        context = "\n\n---\n\n".join(chunk_texts)
        answer  = _invoke_with_backoff(answer_chain, {"context": context, "question": q})

        # Inter-question pause to stay within API rate limits
        time.sleep(5.0)

        results.append({
            "question":    q,
            "answer":      answer,
            "contexts":    chunk_texts,    # top-n semantic chunks used as eval context
            "reference":   references.get(qid, ""),
            "sub_queries": sub_queries,    # captured for traceability
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
        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy", "context_precision")]
        scores = {c: round(float(df[c].mean()), 4) for c in metric_cols}
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
                from anthropic import Anthropic as AC
                self._c = AC(api_key=ANTHROPIC_KEY)

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
                from anthropic import AsyncAnthropic
                r = await AsyncAnthropic(api_key=ANTHROPIC_KEY).messages.create(
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
        test_results = getattr(result, "test_results", []) or []
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

        def _avg_score(col: str) -> float | None:
            if col not in result_df.columns:
                return None
            vals = [v.get("score") for v in result_df[col] if isinstance(v, dict) and v.get("score") is not None]
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

        def _score(fn, *args) -> float | None:
            try:
                v = fn(*args)
                return float(v[0]) if isinstance(v, tuple) else float(v)
            except Exception:
                return None

        ans_rel, ctx_rel, ground = [], [], []
        for r in rag_results:
            ctx = "\n\n---\n\n".join(r["contexts"])
            s = _score(provider.relevance, r["question"], r["answer"])
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
    print(f"  Sub-Query Semantic RAG Evaluation  |  samples={EVAL_LIMIT}")
    print(f"  chunker : SemanticChunker({BREAKPOINT_AMOUNT}th pct)  [single layer, no parent-child]")
    print(f"  + Sub-Query Expansion: Haiku tool_choice → 2–3 focused sub-queries")
    print(f"  embed={EMBED_MODEL}  top_k_per_query={TOP_K_RETRIEVE}  top_n_rerank={TOP_N_RERANK}")
    print(f"  reranker={RERANKER_MODEL}  context=top-n semantic chunks (no parent swap)")
    print(f"{'='*64}\n")

    store       = build_semantic_index()
    rag_results = run_rag(store)
    _save("rag_results.json", rag_results)

    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "iteration":              ITERATION,
            "chunker":                "SemanticChunker",
            "query_transformation":   "SubQueryExpansion",
            "sub_query_model":        LLM_MODEL,
            "breakpoint_type":        BREAKPOINT_TYPE,
            "breakpoint_amount":      BREAKPOINT_AMOUNT,
            "embed_model":            EMBED_MODEL,
            "reranker_model":         RERANKER_MODEL,
            "top_k_retrieve_per_query": TOP_K_RETRIEVE,
            "top_n_rerank":           TOP_N_RERANK,
            "context_strategy":       "direct_semantic_chunks",
            "parent_child":           False,
            "eval_samples":           EVAL_LIMIT,
            "llm_model":              LLM_MODEL,
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
