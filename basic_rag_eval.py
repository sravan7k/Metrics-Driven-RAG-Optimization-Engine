"""
Basic RAG Evaluation Pipeline
Iteration  : basic_rag
Chunking   : fixed-size character (size=500, overlap=50)
Index      : LlamaIndex in-memory VectorStoreIndex
LLM        : claude-haiku-4-5-20251001  (Anthropic)
Embeddings : BAAI/bge-small-en-v1.5    (HuggingFace, local)
Frameworks : RAGAS | DeepEval | LangSmith | Arize Phoenix | TruLens
Dataset    : combined (golden + synthetic) → data/combined_dataset.json

Results are written to:
  results/basic_rag/rag_results.json
  results/basic_rag/metrics_<framework>.json
  results/basic_rag/summary.json
  results/comparison.json          ← updated across iterations
"""

import os
import json
import traceback
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
ITERATION     = "basic_rag"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 3
EVAL_LIMIT    = int(os.getenv("EVAL_LIMIT", "10"))
ESSAY_PATH    = Path("data/paul_graham_essay.txt")
GOLDEN_PATH   = Path("data/combined_dataset.json")
RESULTS_DIR   = Path(f"results/{ITERATION}")
COMPARE_FILE  = Path("results/comparison.json")

ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
LANGSMITH_KEY = os.getenv("LANGSMITH_API_KEY")   # optional – tracing only if set

LLM_MODEL   = "claude-haiku-4-5-20251001"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_FILE.parent.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(filename: str, data: dict) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
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

    # Merge per-question responses so each question accumulates answers from all iterations.
    for r in rag_results:
        q = r["question"]
        if q not in comparison["responses"]:
            comparison["responses"][q] = {"reference": r["reference"], "answers": {}}
        comparison["responses"][q]["answers"][ITERATION] = r["answer"]

    COMPARE_FILE.write_text(json.dumps(comparison, indent=2))
    print(f"  comparison → {COMPARE_FILE}")


# ── 1. Character Chunking ─────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """Pure character-based chunking with overlap."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── 2. Build LlamaIndex In-Memory Vector Index ───────────────────────────────
def build_index():
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.anthropic import Anthropic

    print("Building index …")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Anthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY)

    chunks = chunk_text(ESSAY_PATH.read_text())
    print(f"  chunks : {len(chunks)}")
    nodes = [TextNode(text=c, id_=f"chunk_{i}") for i, c in enumerate(chunks)]
    index = VectorStoreIndex(nodes, show_progress=True)
    return index


# ── Helpers (dataset) ─────────────────────────────────────────────────────────
def _is_question(text: str) -> bool:
    """Return True only for real questions; skip markdown headers and label lines."""
    stripped = text.strip()
    return (
        not stripped.startswith("#")
        and not stripped.startswith("**")
        and not stripped.startswith("##")
        and len(stripped) > 60
    )


# ── 3. Run RAG Pipeline ───────────────────────────────────────────────────────
def run_rag(index) -> list[dict]:
    print("\nRunning RAG pipeline …")
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    questions  = data["queries"]
    references = data["responses"]
    qids       = [qid for qid, q in questions.items() if _is_question(q)][:EVAL_LIMIT]
    qe         = index.as_query_engine(similarity_top_k=TOP_K)

    results = []
    for i, qid in enumerate(qids, 1):
        q = questions[qid]
        print(f"  [{i:>2}/{len(qids)}] {q[:72]}…")
        resp = qe.query(q)
        results.append({
            "question":  q,
            "answer":    str(resp),
            "contexts":  [n.node.get_content() for n in resp.source_nodes],
            "reference": references.get(qid, ""),
        })
    return results


# ── 4. RAGAS ─────────────────────────────────────────────────────────────────
def eval_ragas(rag_results: list[dict]) -> dict:
    print("\n── RAGAS ──")
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_anthropic import ChatAnthropic
        from langchain_huggingface import HuggingFaceEmbeddings

        llm = LangchainLLMWrapper(ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY))
        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))

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


# ── 6. LangSmith ─────────────────────────────────────────────────────────────
def eval_langsmith(rag_results: list[dict]) -> dict:
    """
    Uses LangChain string evaluators locally.
    If LANGSMITH_API_KEY is set, traces are also uploaded to LangSmith.
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
            scores = [v.get("score") for v in result_df[col] if isinstance(v, dict) and v.get("score") is not None]
            return round(sum(scores) / len(scores), 4) if scores else None

        scores = {
            "faithfulness":  _avg_score("faithfulness_score"),
            "correctness":   _avg_score("correctness_score"),
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
    TruLens 2.x Anthropic support via the LiteLLM provider
    (trulens-providers-litellm bridges litellm → Anthropic).
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
    print(f"  Basic RAG Evaluation  |  samples={EVAL_LIMIT}")
    print(f"  chunk_size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}  top_k={TOP_K}")
    print(f"{'='*64}\n")

    index       = build_index()
    rag_results = run_rag(index)
    _save("rag_results.json", rag_results)

    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "iteration":     ITERATION,
            "chunk_size":    CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k":         TOP_K,
            "eval_samples":  EVAL_LIMIT,
            "llm_model":     LLM_MODEL,
            "embed_model":   EMBED_MODEL,
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
