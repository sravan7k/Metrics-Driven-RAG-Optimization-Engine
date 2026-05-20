"""
Semantic RAG Evaluation Pipeline
Iteration  : semantic_rag
Chunking   : SemanticChunker (langchain-experimental)
             OpenAIEmbeddings(model="text-embedding-3-small")
             breakpoint_threshold_type="percentile"  threshold=90
Index      : LangChain InMemoryVectorStore (OpenAI text-embedding-3-small)
LLM        : claude-haiku-4-5-20251001  (Anthropic)
Frameworks : RAGAS | DeepEval | LangSmith | Arize Phoenix | TruLens
Dataset    : combined (golden + synthetic) → data/combined_dataset.json

Results are written to:
  results/semantic_rag/rag_results.json
  results/semantic_rag/metrics_<framework>.json
  results/semantic_rag/summary.json
  results/comparison.json          ← updated across iterations

Prompts are written to:
  prompts/semantic_rag/pipeline_prompts.json   ← RAG chain template (runtime)
  prompts/semantic_rag/initial_prompt.md       ← static, committed separately
  prompts/semantic_rag/eval_framework_prompts.md
  prompts/semantic_rag/conversation_prompts.md
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
ITERATION               = "semantic_rag"
BREAKPOINT_TYPE         = "percentile"      # SemanticChunker breakpoint strategy
BREAKPOINT_AMOUNT       = 90                # 90th-percentile threshold
TOP_K                   = 3
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
EMBED_MODEL             = "text-embedding-3-small"      # OpenAI embedding

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_FILE.parent.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(filename: str, data: dict) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {path}")


def _save_pipeline_prompts(qa_template: str) -> None:
    """Write the runtime RAG chain template to prompts/semantic_rag/pipeline_prompts.json."""
    payload = {
        "description": "LangChain prompt templates used in the semantic_rag pipeline.",
        "source": "extracted from semantic_rag_eval.py — ChatPromptTemplate passed to the LCEL chain",
        "chunker": {
            "description": (
                "SemanticChunker does not use an LLM prompt. "
                "It computes cosine-similarity distances between sentence embeddings "
                "and splits at distances above the breakpoint threshold. "
                "No prompt template applies."
            ),
            "model": f"OpenAIEmbeddings(model='{EMBED_MODEL}')",
            "breakpoint_threshold_type": BREAKPOINT_TYPE,
            "breakpoint_threshold_amount": BREAKPOINT_AMOUNT,
        },
        "templates": {
            "rag_qa_template": {
                "description": (
                    "Main prompt for answering a question given semantically-retrieved context chunks "
                    "(chat mode via ChatAnthropic)."
                ),
                "template": qa_template,
                "input_variables": ["context", "question"],
                "notes": (
                    "context is formed by joining each retrieved Document's page_content "
                    "with '\\n\\n---\\n\\n'. "
                    "question is passed through RunnablePassthrough."
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


# ── 1. SemanticChunker ────────────────────────────────────────────────────────
def build_semantic_chunks(text: str) -> list[str]:
    """
    Split text using cosine-similarity breakpoints between sentences.
    Breakpoints above the 90th percentile of similarity distances trigger
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


# ── 2. Build LangChain InMemoryVectorStore ────────────────────────────────────
def build_index():
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    print("Building index …")
    text    = ESSAY_PATH.read_text()
    chunks  = build_semantic_chunks(text)
    print(f"  semantic chunks : {len(chunks)}")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_KEY)
    store      = InMemoryVectorStore.from_texts(chunks, embedding=embeddings)
    return store


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
def run_rag(store) -> list[dict]:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    print("\nRunning RAG pipeline …")
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    questions  = data["queries"]
    references = data["responses"]
    qids       = [qid for qid, q in questions.items() if _is_question(q)][:EVAL_LIMIT]

    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    llm       = ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_KEY)

    qa_template = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely and accurately based only on the provided context."
    )
    _save_pipeline_prompts(qa_template)
    prompt = ChatPromptTemplate.from_template(qa_template)

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    results = []
    for i, qid in enumerate(qids, 1):
        q = questions[qid]
        print(f"  [{i:>2}/{len(qids)}] {q[:72]}…")
        retrieved_docs = retriever.invoke(q)
        answer         = chain.invoke(q)
        results.append({
            "question":  q,
            "answer":    answer,
            "contexts":  [d.page_content for d in retrieved_docs],
            "reference": references.get(qid, ""),
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


# ── 6. LangSmith ──────────────────────────────────────────────────────────────
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
    TruLens 2.x via the LiteLLM provider bridging to Anthropic.
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
    print(f"  Semantic RAG Evaluation  |  samples={EVAL_LIMIT}")
    print(f"  chunker=SemanticChunker  breakpoint={BREAKPOINT_TYPE}@{BREAKPOINT_AMOUNT}")
    print(f"  embed={EMBED_MODEL}  top_k={TOP_K}")
    print(f"{'='*64}\n")

    store       = build_index()
    rag_results = run_rag(store)
    _save("rag_results.json", rag_results)

    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "iteration":              ITERATION,
            "chunker":                "SemanticChunker",
            "breakpoint_type":        BREAKPOINT_TYPE,
            "breakpoint_amount":      BREAKPOINT_AMOUNT,
            "embed_model":            EMBED_MODEL,
            "top_k":                  TOP_K,
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
