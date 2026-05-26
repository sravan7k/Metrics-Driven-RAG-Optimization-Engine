"""
Re-run Arize Phoenix + TruLens evaluations for sub_query_rag only.
Loads existing rag_results.json — no RAG pipeline re-execution.
Patches metrics_arize_phoenix.json, metrics_trulens.json, summary.json,
and comparison.json in-place.
"""

import os, json, time, traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ITERATION     = "sub_query_rag"
RESULTS_DIR   = Path(f"results/{ITERATION}")
COMPARE_FILE  = Path("results/comparison.json")
LLM_MODEL     = "claude-haiku-4-5-20251001"
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

rag_results: list[dict] = json.loads((RESULTS_DIR / "rag_results.json").read_text())
print(f"Loaded {len(rag_results)} results from {RESULTS_DIR / 'rag_results.json'}")


# ── Arize Phoenix ─────────────────────────────────────────────────────────────
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


# ── TruLens ───────────────────────────────────────────────────────────────────
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


# ── Patch helpers ─────────────────────────────────────────────────────────────
def _save(filename: str, data) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {path}")


def _patch_summary(ap_scores: dict, tl_scores: dict) -> None:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    summary["arize_phoenix"] = ap_scores
    summary["trulens"]       = tl_scores
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  patched → {RESULTS_DIR / 'summary.json'}")


def _patch_comparison(ap_scores: dict, tl_scores: dict) -> None:
    comparison = json.loads(COMPARE_FILE.read_text())
    comparison["iterations"][ITERATION]["arize_phoenix"] = ap_scores
    comparison["iterations"][ITERATION]["trulens"]       = tl_scores
    COMPARE_FILE.write_text(json.dumps(comparison, indent=2))
    print(f"  patched → {COMPARE_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  sub_query_rag — Arize Phoenix + TruLens re-evaluation")
    print(f"  samples = {len(rag_results)}")
    print(f"{'='*60}\n")

    ap_scores = eval_arize_phoenix(rag_results)
    tl_scores = eval_trulens(rag_results)

    print("\nSaving …")
    _save("metrics_arize_phoenix.json", ap_scores)
    _save("metrics_trulens.json",       tl_scores)
    _patch_summary(ap_scores, tl_scores)
    _patch_comparison(ap_scores, tl_scores)

    print(f"\n{'='*60}")
    print("  Final Scores")
    print(f"{'='*60}")
    print(f"\n  [arize_phoenix]")
    for k, v in ap_scores.items():
        print(f"    {k:<24} {v}")
    print(f"\n  [trulens]")
    for k, v in tl_scores.items():
        print(f"    {k:<24} {v}")


if __name__ == "__main__":
    main()
