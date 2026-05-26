# RAG Evaluation Report: Iteration `semantic_rag`
**Timestamp:** 2026-05-20T04:27:55Z  
**Architecture Status:** Semantic Chunking Enabled (Baseline Character Chunking Deprecated)

---

The implementation of the **Semantic Chunker RAG system** (`semantic_rag`) shows a significant and comprehensive improvement over your baseline fixed-size chunking system (`basic_rag`). By moving from a rigid 500-character chunk size to semantic-based boundaries (using a 90th percentile breakpoint) and upgrading to the `text-embedding-3-small` model, you have successfully resolved major context fragmentation issues.

---

## 1. Quantitative Evaluation Metrics Breakdown

Across nearly every evaluation framework (`ragas`, `deepeval`, `langsmith`, `trulens`, and `arize_phoenix`), the semantic RAG pipeline outperforms the basic configuration:

| Framework & Metric | Basic RAG (`basic_rag`) | Semantic RAG (`semantic_rag`) | Analysis & Impact |
| :--- | :---: | :---: | :--- |
| **RAGAS** | | | |
| *Context Precision* | 0.1500 | 0.6000 | **+400% Improvement:** Semantic chunks keep sentences that belong together intact, drastically reducing irrelevant noise in retrieved context. |
| *Answer Relevancy* | 0.4836 | 0.7461 | Significant boost; the model answers the core prompt instead of complaining about partial data. |
| *Faithfulness* | 0.9037 | 0.9177 | Remains consistently high. |
| **DeepEval** | | | |
| *Contextual Precision*| 0.5417 | 0.9000 | Confirms that semantic boundaries align closely with the information requested by the user. |
| *Answer Relevancy* | 0.9202 | 0.9418 | Highly relevant generations. |
| **LangSmith** | | | |
| *Correctness* | 0.2000 | 0.9000 | **Massive leap:** Basic chunking was losing factual links, leading to an extremely low correctness score which is now solved. |
| **Arize Phoenix** | | | |
| *Correctness* | 0.2000 | 0.7000 | Strong validation of factuality improvements. |
| **TruLens** | | | |
| *Answer Relevance* | 0.6000 | 0.9333 | Better alignment with user intent. |
| *Context Relevance* | 0.8333 | 0.9333 | Better semantic matching during retrieval. |

---

## 2. Qualitative Analysis: Why Semantic Chunking Solved the Problem

Looking at the underlying responses helps explain exactly why your metrics skyrocketed:

### A. Elimination of "Fragmented Context" Failures
In `basic_rag`, fixed character boundaries chopped up paragraphs arbitrarily. Because of this, the LLM frequently refused to answer or failed to find facts because sentences were severed mid-thought.
* **Example (Accademia's "Arrangement"):** When asked about the student-faculty setup at the art academy, `basic_rag` gave up, stating: *"The text is incomplete and does not contain information... The passage cuts off mid-sentence."*
* `semantic_rag` successfully kept the entire topic unit intact, delivering a flawless explanation of the mutual non-engagement arrangement.

### B. Fixing Hallucinations and Context Overlap Errors
When character limits force chunks to blend unrelated ideas together, basic RAG pipelines often mix up contexts.
* **Example (IBM 1401 Limitations):** When asked about Paul Graham's 9th-grade experience with the IBM 1401, `basic_rag` mixed up details and erroneously retrieved constraints from a completely different machine (the TRS-80 word processor memory limits).
* `semantic_rag` accurately retrieved only the details pertaining specifically to the IBM 1401 (lack of punch card data and math knowledge).

### C. Comprehensive & Nuanced Answers
For multi-part conceptual questions (such as why Graham switched from Philosophy to AI), `basic_rag` only caught a single superficial detail (*"he found courses boring"*), whereas `semantic_rag` captured the structural arguments (other fields dominating the space of ideas, and philosophy being relegated only to edge cases).

---

## 3. Areas for Continued Optimization

While the `semantic_rag` setup is excellent, you can squeeze out even more performance by looking at the remaining gaps:

* **Missed Sub-Inspirations:** For the question regarding the two specific inspirations that motivated Graham's work in AI (Heinlein's novel and the PBS Winograd documentary), both systems missed the exact call-out in their final output, or stated that the text didn't explicitly name them. This indicates that even with semantic chunking, if those two details are separated from the surrounding "Lisp/AI" discussion, your `top_k: 3` retrieval constraint might be leaving that specific chunk behind. Try increasing `top_k` to `4` or `5` or testing a small re-ranker (like Cohere) to see if it pulls that missing node into view.
* **Arize Phoenix Faithfulness Drop:** You'll notice a very minor decrease in Arize Phoenix Faithfulness ($1.0 \rightarrow 0.9$). This happens because `semantic_rag` writes much lengthier, highly descriptive answers rather than short, safe defensive answers. Longer responses naturally introduce more evaluation surface area for automated judges.

---

## Verdict

The change to **`SemanticChunker`** was an unqualified success. It completely turned the system around from a fundamentally broken prototype (20% correctness) to a highly precise, production-grade RAG pipeline (70%–90% correctness).