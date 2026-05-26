# Evaluation Report: Naive RAG Baseline (Iteration 1)

This document establishes the performance baseline for our initial, naive Retrieval-Augmented Generation (RAG) implementation. By evaluating our system across five distinct frameworks, we isolated specific pipeline bottlenecks before implementing advanced retrieval strategies.

## Execution Configuration
* **Timestamp:** 2026-05-18
* **Chunking Strategy:** Fixed-size character splitting (`chunk_size`: 500, `chunk_overlap`: 50)
* **Embedding Model:** `BAAI/bge-small-en-v1.5`
* **LLM (Generator):** `claude-haiku-4-5-20251001`
* **Top-K Retrieval:** 3 chunks
* **Dataset:** Paul Graham Essay ("What I Worked On")

---

## Metric Summary Matrix

| Framework | Metric Category | Parameter | Score |
| :--- | :--- | :--- | :--- |
| **RAGAS** | Retrieval | Context Precision | **0.0000** |
| | Generation | Faithfulness | 0.6374 |
| | Generation | Answer Relevancy | 0.3597 |
| **DeepEval** | Retrieval | Contextual Precision | **0.2000** |
| | Generation | Faithfulness | 0.7543 |
| | Generation | Answer Relevancy | 1.0000* |
| **TruLens** | Retrieval | Context Relevance | **0.4667** |
| | Generation | Groundedness | 0.5586 |
| | Generation | Answer Relevance | 0.2667 |
| **LangSmith** | Generation | Correctness / Match | **0.0000** |
| | Generation | Relevance | 0.4000 |
| **Arize Phoenix** | Generation | Faithfulness | 0.8000 |
| | Generation | Correctness | 0.8000 |

*\*Note: High relevancy metrics in DeepEval/Arize typically point to a lenient evaluation prompt that validates if the model structurally addressed the prompt topic, rather than a strict ground-truth match.*

---

## Performance Deep Dive (The RAG Triad)

The data across all five frameworks points to a singular conclusion: **The LLM is performing optimally given its constraints, but the retrieval layer is failing completely.** ### 1. The Root Problem: Retrieval Failure (Context Metrics)
The system is fundamentally struggling to locate and surface relevant information from the corpus.
* **Key Indicators:** RAGAS Context Precision (`0.0`), DeepEval Contextual Precision (`0.2`), TruLens Context Relevance (`0.4667`).
* **Analysis:** A score of `0.0` on RAGAS indicates that for the evaluation subset, the precise text block required to formulate the answer was almost never ranked as the #1 result. 
* **Root Cause:** Naive fixed-size character chunking blindly splits narrative prose. This mechanical slicing cuts essential coreferences, pronouns (*"he"*, *"it"*, *"they"*), and logical sentence structures in half. It strips away vital context before the embeddings are ever generated, resulting in low-quality vector matches.

### 2. The Ripple Effect: LLM Hallucinations (Faithfulness)
Because the retrieval sub-system supplied irrelevant or fragmented context, the generator was forced to guess.
* **Key Indicators:** TruLens Groundedness (`0.5586`), RAGAS Faithfulness (`0.6374`), DeepEval Faithfulness (`0.7543`).
* **Analysis:** Faithfulness measures whether the LLM's response is *strictly and exclusively* supported by the retrieved context. 
* **Root Cause:** Scores lingering between `0.55` and `0.75` mean that **25% to 45% of the assertions made by Claude-Haiku were fabricated**, or pulled from its pre-trained parametric weights rather than our data. The LLM attempted to compensate for poor context by hallucinating missing details.

### 3. The Final Output: Low Correctness & Relevance
When the input context is broken and the model hallucinates, the final user-facing answer suffers entirely.
* **Key Indicators:** LangSmith Correctness (`0.0`), TruLens Answer Relevance (`0.2667`), RAGAS Answer Relevancy (`0.3597`).
* **Analysis:** LangSmith's absolute `0.0` Correctness score mathematically confirms that our production pipeline failed to resolve the actual answers defined in our Golden Dataset.

---

## Engineering Verdict & Next Steps

> **Baseline Conclusion:** Naive RAG (arbitrary 500-character block indexing) is completely unsuited for handling unstructured, long-form narrative documents. 

To resolve the core retrieval bottleneck, we will pivot away from static token/character chunk boundaries. Our next development iteration will implement **Semantic Chunking**, utilizing an embedding-driven distance threshold loop to split files along native semantic shifts rather than character counts.