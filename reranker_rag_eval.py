"""
Reranker RAG Evaluation Pipeline
Iteration  : reranker_rag
Chunking   : SemanticChunker (langchain-experimental)
             OpenAIEmbeddings(model="text-embedding-3-small")
             breakpoint_threshold_type="percentile"  threshold=90
Index      : LangChain InMemoryVectorStore (OpenAI text-embedding-3-small)
Retrieval  : Two-stage — dense vector search (top_k=25) → BGE Cross-Encoder rerank → top_n=3
Reranker   : BAAI/bge-reranker-large  (sentence-transformers CrossEncoder, local)
LLM        : claude-haiku-4-5-20251001  (Anthropic)
Frameworks : RAGAS | DeepEval | LangSmith | Arize Phoenix | TruLens
Dataset    : combined (golden + synthetic) → data/combined_dataset.json

Pipeline:
  User Query
    → InMemoryVectorStore dense search  (top_k=25)
    → BAAI/bge-reranker-large CrossEncoder  (score all 25 pairs)
    → Sort descending, slice top_n=3
    → LLM context window

Results are written to:
  results/reranker_rag/rag_results.json
  results/reranker_rag/metrics_<framework>.json
  results/reranker_rag/summary.json
  results/comparison.json          ← updated across iterations

Prompts are written to:
  prompts/reranker_rag/pipeline_prompts.json   ← RAG chain template (runtime)
  prompts/reranker_rag/initial_prompt.md       ← static, committed separately
  prompts/reranker_rag/eval_framework_prompts.md
  prompts/reranker_rag/conversation_prompts.md
"""

import os
import json
import traceback
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ITERATION               = "reranker_rag"
BREAKPOINT_TYPE         = "percentile"
BREAKPOINT_AMOUNT       = 90
TOP_K_RETRIEVE          = 25        # broad initial vector sweep
TOP_N_RERANK            = 3         # sliced final context after BGE reranking
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(filename: str, data: dict) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {path}")


def _save_pipeline_prompts(qa_template: str) -> None:
    payload = {
        "description": "LangChain prompt templates used in the reranker_rag pipeline.",
        "source": "extracted from reranker_rag_eval.py",
        "chunker": {
            "description": (
                "SemanticChunker does not use an LLM prompt. "
                "It computes cosine-similarity distances between sentence embeddings "
                "and splits at distances above the breakpoint threshold."
            ),
            "model": f"OpenAIEmbeddings(model='{EMBED_MODEL}')",
            "breakpoint_threshold_type": BREAKPOINT_TYPE,
            "breakpoint_threshold_amount": BREAKPOINT_AMOUNT,
        },
        "reranker": {
            "description": (
                "BGE Cross-Encoder reranker — no LLM prompt. "
                "Pairs (query, doc_text) are passed through the CrossEncoder to produce "
                "a bidirectional semantic relevance score. Results are sorted descending "
                f"and sliced to top_n={TOP_N_RERANK}."
            ),
            "model": RERANKER_MODEL,
            "top_k_retrieve": TOP_K_RETRIEVE,
            "top_n_rerank": TOP_N_RERANK,
        },
        "templates": {
            "rag_qa_template": {
                "description": (
                    "Main prompt for answering a question given the top-3 BGE-reranked context chunks "
                    "(chat mode via ChatAnthropic)."
                ),
                "template": qa_template,
                "input_variables": ["context", "question"],
                "notes": (
                    "context is formed by joining each reranked Document's page_content "
                    "with '\\n\\n---\\n\\n'. "
                    "question is the raw user query."
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


# ── 1. SemanticChunker ────────────────────────────────────────────────────────
def build_semantic_chunks(text: str) -> list[str]:
    """
    Split text using cosine-similarity breakpoints between sentences.
    Breakpoints above the 90th-percentile of similarity distances trigger
    a new chunk boundary, producing variable-length topic-coherent segments.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_KEY)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=BREAKPOINT_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_AMOUNT,
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]


# ── 2. Build InMemoryVectorStore ──────────────────────────────────────────────
def build_index():
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    print("Building index …")
    text   = ESSAY_PATH.read_text()
    chunks = build_semantic_chunks(text)
    print(f"  semantic chunks : {len(chunks)}")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_KEY)
    store      = InMemoryVectorStore.from_texts(chunks, embedding=embeddings)
    return store


# ── 3. BGE Cross-Encoder Reranker ─────────────────────────────────────────────
# Loaded once at module level to avoid reloading per query.
print(f"Loading Cross-Encoder  {RERANKER_MODEL} …")
from sentence_transformers import CrossEncoder
_cross_encoder = CrossEncoder(RERANKER_MODEL)
print("  Cross-Encoder ready.")


def rerank_documents(query: str, retrieved_docs: list, top_n: int = 5) -> list:
    """
    Score each (query, doc) pair with BAAI/bge-reranker-large and return
    the top_n documents sorted by descending relevance score.

    Logs the original retrieval order vs. the BGE re-ranked order so the
    impact of reranking is visible in the console output.
    """
    pairs  = [(query, doc.page_content) for doc in retrieved_docs]
    scores = _cross_encoder.predict(pairs)      # shape: (len(retrieved_docs),)

    # ── Log original order with cross-encoder scores ──────────────────────────
    print(f"\n    ── Original vector-retrieval order (top_k={len(retrieved_docs)}) ──")
    for i, (doc, score) in enumerate(zip(retrieved_docs, scores), 1):
        snippet = doc.page_content[:80].replace("\n", " ")
        print(f"      [{i:>2}] score={score:+.4f}  │  {snippet}…")

    # ── Sort by cross-encoder score descending ────────────────────────────────
    scored = sorted(zip(scores, range(len(retrieved_docs)), retrieved_docs),
                    key=lambda x: x[0], reverse=True)

    top_scored = scored[:top_n]

    # ── Log re-ranked order with original rank shown ───────────────────────────
    print(f"\n    ── BGE re-ranked order (top_n={top_n}) ──")
    for new_rank, (score, orig_rank, doc) in enumerate(top_scored, 1):
        snippet = doc.page_content[:80].replace("\n", " ")
        print(f"      [{new_rank}] score={score:+.4f}  orig_rank={orig_rank + 1:>2}  │  {snippet}…")

    return [doc for _, _, doc in top_scored]


# ── 4. Run RAG Pipeline ───────────────────────────────────────────────────────
def run_rag(store) -> list[dict]:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print("\nRunning RAG pipeline …")
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    questions  = data["queries"]
    references = data["responses"]
    qids       = [qid for qid, q in questions.items() if _is_question(q)][:EVAL_LIMIT]

    # Broad retriever — fetch 25 candidates for the reranker to score
    retriever = store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
    llm       = ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY)

    qa_template = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely and accurately based only on the provided context."
    )
    _save_pipeline_prompts(qa_template)
    prompt = ChatPromptTemplate.from_template(qa_template)
    answer_chain = prompt | llm | StrOutputParser()

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    results = []
    for i, qid in enumerate(qids, 1):
        q = questions[qid]
        print(f"\n  [{i:>2}/{len(qids)}] {q[:72]}…")

        # Stage 1: broad dense retrieval
        broad_docs = retriever.invoke(q)
        print(f"    vector search retrieved {len(broad_docs)} docs")

        # Stage 2: BGE cross-encoder reranking → top 3
        reranked_docs = rerank_documents(q, broad_docs, top_n=TOP_N_RERANK)

        # Stage 3: LLM generation on top-3 reranked context
        context = format_docs(reranked_docs)
        answer  = answer_chain.invoke({"context": context, "question": q})

        results.append({
            "question":  q,
            "answer":    answer,
            "contexts":  [d.page_content for d in reranked_docs],
            "reference": references.get(qid, ""),
        })

    return results


# ── 5. RAGAS ──────────────────────────────────────────────────────────────────
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


# ── 6. DeepEval ───────────────────────────────────────────────────────────────
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

            def load_model(self): return self._c

            def generate(self, prompt: str) -> str:
                r = self._c.messages.create(
                    model=LLM_MODEL, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text

            async def a_generate(self, prompt: str) -> str:
                from anthropic import AsyncAnthropic
                r = await AsyncAnthropic(api_key=ANTHROPIC_KEY).messages.create(
                    model=LLM_MODEL, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text

            def get_model_name(self): return LLM_MODEL

        model = _Claude()
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


# ── 7. LangSmith ──────────────────────────────────────────────────────────────
def eval_langsmith(rag_results: list[dict]) -> dict:
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
            rl_scores.append(
                rl_eval.evaluate_strings(
                    prediction=r["answer"],
                    input=r["question"],
                ).get("score", 0)
            )

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


# ── 8. Arize Phoenix ──────────────────────────────────────────────────────────
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
            scores = [v.get("score") for v in result_df[col] if isinstance(v, dict) and v.get("score") is not None]
            return round(sum(scores) / len(scores), 4) if scores else None

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


# ── 9. TruLens ────────────────────────────────────────────────────────────────
def eval_trulens(rag_results: list[dict]) -> dict:
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
            if s is not None: ans_rel.append(s)
            s = _score(provider.context_relevance, r["question"], ctx)
            if s is not None: ctx_rel.append(s)
            s = _score(provider.groundedness_measure_with_cot_reasons, ctx, r["answer"])
            if s is not None: ground.append(s)

        def _avg(lst): return round(sum(lst) / len(lst), 4) if lst else None

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
    print(f"  Reranker RAG Evaluation  |  samples={EVAL_LIMIT}")
    print(f"  chunker=SemanticChunker  breakpoint={BREAKPOINT_TYPE}@{BREAKPOINT_AMOUNT}")
    print(f"  embed={EMBED_MODEL}  top_k_retrieve={TOP_K_RETRIEVE}  top_n_rerank={TOP_N_RERANK}")
    print(f"  reranker={RERANKER_MODEL}")
    print(f"{'='*64}\n")

    store       = build_index()
    rag_results = run_rag(store)
    _save("rag_results.json", rag_results)

    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "iteration":          ITERATION,
            "chunker":            "SemanticChunker",
            "breakpoint_type":    BREAKPOINT_TYPE,
            "breakpoint_amount":  BREAKPOINT_AMOUNT,
            "embed_model":        EMBED_MODEL,
            "reranker_model":     RERANKER_MODEL,
            "top_k_retrieve":     TOP_K_RETRIEVE,
            "top_n_rerank":       TOP_N_RERANK,
            "eval_samples":       EVAL_LIMIT,
            "llm_model":          LLM_MODEL,
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
