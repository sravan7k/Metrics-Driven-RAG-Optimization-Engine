"""
Re-run ALL evaluation frameworks for sub_query_rag only.
Loads existing rag_results.json — skips RAG pipeline re-execution.
Overwrites metrics_*.json, summary.json, and patches comparison.json
only for the sub_query_rag iteration.
"""

import os, json, time, traceback
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

ITERATION     = "sub_query_rag"
RESULTS_DIR   = Path(f"results/{ITERATION}")
COMPARE_FILE  = Path("results/comparison.json")
LLM_MODEL     = "claude-haiku-4-5-20251001"
EMBED_MODEL   = "text-embedding-3-small"
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_KEY    = os.environ["OPENAI_API_KEY"]
LANGSMITH_KEY = os.getenv("LANGSMITH_API_KEY")

rag_results: list[dict] = json.loads((RESULTS_DIR / "rag_results.json").read_text())
print(f"Loaded {len(rag_results)} results from {RESULTS_DIR / 'rag_results.json'}")


# ── Backoff helper ────────────────────────────────────────────────────────────
def _anthropic_call_with_backoff(client, max_retries: int = 6, **kwargs):
    import random
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


# ── 1. RAGAS ──────────────────────────────────────────────────────────────────
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


# ── 2. DeepEval ───────────────────────────────────────────────────────────────
def eval_deepeval(rag_results: list[dict]) -> dict:
    print("\n── DeepEval ──")
    try:
        from deepeval import evaluate as deval
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
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
                    self._c, model=LLM_MODEL, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                time.sleep(0.5)
                return r.content[0].text

            async def a_generate(self, prompt: str) -> str:
                from anthropic import AsyncAnthropic
                r = await AsyncAnthropic(api_key=ANTHROPIC_KEY).messages.create(
                    model=LLM_MODEL, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text

            def get_model_name(self):
                return LLM_MODEL

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
        for tr in getattr(result, "test_results", []) or []:
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


# ── 3. LangSmith ──────────────────────────────────────────────────────────────
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
                    prediction=r["answer"], input=r["question"],
                    reference=r["reference"] or r["answer"],
                ).get("score", 0)
            )
            time.sleep(1.0)
            rl_scores.append(
                rl_eval.evaluate_strings(
                    prediction=r["answer"], input=r["question"],
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


# ── 4. Arize Phoenix ──────────────────────────────────────────────────────────
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


# ── 5. TruLens ────────────────────────────────────────────────────────────────
def eval_trulens(rag_results: list[dict]) -> dict:
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
        for i, r in enumerate(rag_results, 1):
            print(f"  [{i}/{len(rag_results)}] scoring …")
            ctx = "\n\n---\n\n".join(r["contexts"])
            s = _score(provider.relevance, r["question"], r["answer"])
            if s is not None:
                ans_rel.append(s)
            time.sleep(1.0)
            s = _score(provider.context_relevance, r["question"], ctx)
            if s is not None:
                ctx_rel.append(s)
            time.sleep(1.0)
            s = _score(provider.groundedness_measure_with_cot_reasons, ctx, r["answer"])
            if s is not None:
                ground.append(s)
            time.sleep(1.0)

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


# ── Save / patch helpers ──────────────────────────────────────────────────────
def _save(filename: str, data) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {path}")


def _patch_summary_and_comparison(all_metrics: dict) -> None:
    # Reload existing summary to preserve config / timestamp, then overwrite metrics
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    summary["timestamp"]     = all_metrics["timestamp"]
    summary["ragas"]         = all_metrics["ragas"]
    summary["deepeval"]      = all_metrics["deepeval"]
    summary["langsmith"]     = all_metrics["langsmith"]
    summary["arize_phoenix"] = all_metrics["arize_phoenix"]
    summary["trulens"]       = all_metrics["trulens"]
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  patched → {RESULTS_DIR / 'summary.json'}")

    comparison = json.loads(COMPARE_FILE.read_text())
    comparison["last_updated"] = all_metrics["timestamp"]
    comparison["iterations"][ITERATION].update({
        "timestamp":     all_metrics["timestamp"],
        "ragas":         all_metrics["ragas"],
        "deepeval":      all_metrics["deepeval"],
        "langsmith":     all_metrics["langsmith"],
        "arize_phoenix": all_metrics["arize_phoenix"],
        "trulens":       all_metrics["trulens"],
    })
    COMPARE_FILE.write_text(json.dumps(comparison, indent=2))
    print(f"  patched → {COMPARE_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*64}")
    print(f"  sub_query_rag — All Frameworks Re-evaluation")
    print(f"  samples = {len(rag_results)}  (rag_results.json reused, no pipeline re-run)")
    print(f"{'='*64}\n")

    all_metrics = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "ragas":         eval_ragas(rag_results),
        "deepeval":      eval_deepeval(rag_results),
        "langsmith":     eval_langsmith(rag_results),
        "arize_phoenix": eval_arize_phoenix(rag_results),
        "trulens":       eval_trulens(rag_results),
    }

    print("\nSaving …")
    for fw in ("ragas", "deepeval", "langsmith", "arize_phoenix", "trulens"):
        _save(f"metrics_{fw}.json", all_metrics[fw])
    _patch_summary_and_comparison(all_metrics)

    print(f"\n{'='*64}")
    print("  Final Scores")
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
