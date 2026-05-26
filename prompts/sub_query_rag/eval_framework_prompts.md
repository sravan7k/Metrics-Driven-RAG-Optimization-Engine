# Evaluation Framework Prompts — sub_query_rag

Each framework below sends its own prompts to the LLM when computing scores.
These are the internal templates each framework uses (paraphrased from source).

> **Key distinction from parent_child_semantic_rag:** A Sub-Query Expansion layer sits
> directly on top of the Hierarchical Semantic Parent-Child + BGE Cross-Encoder pipeline.
>
> Before any vector retrieval begins, Claude Haiku decomposes the user query into 2–3
> hyper-focused sub-queries using **Anthropic Tool Use** with a forced `tool_choice`
> (same guarantee as OpenAI Structured Outputs, validated with Pydantic). Dense vector
> search runs independently for the **original query AND each sub-query** (top_k=15 each).
> All child hits are pooled and strictly de-duplicated by `child_id` before being scored
> by the BGE Cross-Encoder against the **original query only**. This expands semantic
> surface coverage while keeping the re-ranking signal clean.
>
> Evaluation frameworks receive the swapped-in **parent chunk texts** as the retrieved
> contexts — the same de-duplicated parent chunks that were fed to the LLM. Because
> sub-query expansion broadens the retrieval surface, the parent chunks entering the
> context window often cover more of the user's information need than a single-query sweep.

---

## Component 1: Sub-Query Expansion (new in this iteration)

### System Prompt (sent to Claude Haiku via Anthropic Tool Use)
```
You are an expert query decomposer for a Retrieval-Augmented Generation (RAG) system
specialised on the essays of Paul Graham.

Your task: analyse the user query and break it into 2–3 distinct, hyper-focused
sub-queries that together cover its full information need.

Rules
-----
1. Comparative queries (e.g. "Compare Paul Graham's experience at Interleaf vs Viaweb")
   MUST be split into one focused sub-query per subject:
   example output: ["Paul Graham Interleaf work experience",
                    "Paul Graham Viaweb startup experience"]
2. Simple or narrow queries: return the original query plus ONE semantically distinct
   alternative phrasing — never repeat the same sentence verbatim.
3. Each sub-query must be self-contained, specific, and optimised for dense vector
   retrieval (cosine similarity in embedding space).
4. No two sub-queries may be semantically identical.
```

### Tool Schema (Pydantic-validated via Anthropic Tool Use)
```json
{
  "name": "generate_sub_queries",
  "input_schema": {
    "type": "object",
    "properties": {
      "sub_queries": {
        "type": "array",
        "items": { "type": "string" },
        "minItems": 2,
        "maxItems": 3
      }
    },
    "required": ["sub_queries"]
  }
}
```

The model is forced to call this tool via `tool_choice={"type": "tool", "name": "generate_sub_queries"}`,
guaranteeing that the response is always valid JSON. The `tool_use` block's `.input` dict
is passed directly to `SubQueryList(**tool_block.input)` for Pydantic validation.

---

## RAGAS

### Faithfulness
Asks the LLM to break the answer into atomic statements, then verify each statement
against the retrieved contexts (semantic parent chunks in this pipeline).

**Step 1 — decompose answer into statements:**
```
Given a question and answer, create one or more statements from each sentence in the given answer.
question: {question}
answer: {answer}
```

**Step 2 — verify each statement against context:**
```
Consider the given context and following statements, then determine whether they are supported by the information present in the context.
Provide a brief explanation for each statement before arriving at the verdict (Yes/No).
context: {context}
statements: {statements}
```

### Answer Relevancy
Generates N reverse questions from the answer and measures cosine similarity between them and the original question.
```
Generate {n} questions for the given answer.
answer: {answer}
```

### Context Precision
Checks whether each retrieved chunk was actually useful for producing the answer.
```
Given question, answer and context verify if the context was useful in arriving at the given answer.
question: {question}
context: {context}
answer: {answer}
```

---

## DeepEval

### FaithfulnessMetric
Extracts claims from the answer, then verifies each claim against retrieval contexts (semantic parent chunks).

**Step 1 — extract claims:**
```
Based on the given text, please generate a comprehensive list of FACTUAL CLAIMS that can inferred from the provided text.
===== TEXT =====
{actual_output}
```

**Step 2 — verify claims:**
```
Based on the given list of truths, determine whether the following claim is truthful.
===== LIST OF TRUTHS =====
{retrieval_context}
===== CLAIM =====
{claim}
```

### AnswerRelevancyMetric
Generates statements from the answer, then scores how many are relevant to the input.
```
Based on the input, generate a list of statements that were made in the actual output.
===== INPUT =====
{input}
===== ACTUAL OUTPUT =====
{actual_output}
```

### ContextualPrecisionMetric
Verifies that retrieved nodes at the top of the ranking are more useful than those lower down.
```
Given the input and expected output, rank the nodes from the retrieval context in order of relevance.
input: {input}
expected output: {expected_output}
retrieval context: {retrieval_context}
```

---

## LangSmith (LangChain evaluators, local)

### QA Evaluator (correctness)
Checks whether the prediction matches the reference answer.
```
You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer.
You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Grade the student answers being lenient with edge cases.
QUESTION: {query}
CONTEXT: {context}
STUDENT ANSWER: {result}
```

### Criteria Evaluator (relevance)
```
Evaluate the following submission with respect to the following criteria:
{criteria}

Only answer with a score of 1 (meets criteria) or 0 (does not meet criteria).

<submission>
{input}
{output}
</submission>
```

---

## Arize Phoenix

### FaithfulnessEvaluator
```
In this task, you will be presented with a query, a reference text and an answer.
The answer is generated to the question based on the reference text.
Your task is to determine if the answer provided is faithful to the reference text by following
these guidelines:
- Faithfulness: The answer should only contain information that is present in the reference text.
Score: (1 for faithful, 0 for not faithful)
```

### CorrectnessEvaluator
```
Compare the following answer to the reference and rate it on correctness on a scale from 0.0 to 1.0.
[Question]: {input}
[Reference]: {reference}
[Answer]: {output}
```

---

## TruLens (via LiteLLM → Anthropic)

### Relevance (answer relevance)
```
You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given PROMPT as a score from 0 to 10 where 10 is the most relevant.
PROMPT: {prompt}
RESPONSE: {response}
RELEVANCE:
```

### Context Relevance
```
You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given QUESTION as a score from 0 to 10 where 10 is the most relevant.
QUESTION: {question}
CONTEXT: {context}
RELEVANCE:
```

### Groundedness (with chain-of-thought)
```
You are a FAITHFULNESS grader. Assess whether the STATEMENT is supported by the CONTEXT.
For each sentence in the statement, reason step-by-step whether the context provides evidence for it.
CONTEXT: {context}
STATEMENT: {statement}
FAITHFULNESS:
```

---

## BGE Cross-Encoder (not an LLM prompt — internal scoring)

The `BAAI/bge-reranker-large` model does **not** use a natural-language prompt template.
It operates as a classification head on top of a bi-encoder transformer:
- Input: concatenated `[CLS] query [SEP] child_chunk_text [SEP]` token sequence
- Output: a single scalar relevance logit (higher = more relevant)
- No prompt engineering applies; performance comes from the model's pre-training on
  MS MARCO and BEIR-style retrieval tasks.
- The full de-duplicated pool of child chunks (from all sub-queries) is scored against
  the **ORIGINAL user query** only. This ensures the re-ranker re-aligns the expanded
  retrieval set to the user's actual intent before the Context Swap.
- The top-4 scoring child chunks trigger the **Context Swap** — their `parent_id`
  metadata is used to retrieve the full semantic parent chunk text from the in-memory store.

---

## Sub-Query Expansion Pipeline (pipeline-level, not an LLM prompt)

```python
# Stage 1 — Sub-Query Expansion
sub_queries = generate_sub_queries(user_query)
# e.g. ["Paul Graham Interleaf experience", "Paul Graham Viaweb startup"]

# Stage 2 — Multi-Query Vector Search
all_queries = [user_query] + sub_queries
raw_hits = []
for q in all_queries:
    raw_hits.extend(retriever.invoke(q))   # top_k=15 per query

# Stage 3 — Strict dedup by child_id
seen_cids = set()
deduped_child_docs = []
for doc in raw_hits:
    cid = doc.metadata["child_id"]
    if cid not in seen_cids:
        seen_cids.add(cid)
        deduped_child_docs.append(doc)

# Stage 4 — BGE Cross-Encoder rerank vs. ORIGINAL query
top_child_docs = rerank_documents(user_query, deduped_child_docs, top_n=4)

# Stage 5 — Context Swap: parent_id lookup → de-duplicate parents
seen_pids = []
parent_texts = []
for child_doc in top_child_docs:            # top-4 after BGE rerank
    pid = child_doc.metadata["parent_id"]
    if pid not in seen_pids:
        parent_texts.append(parent_store[pid])   # full semantic parent block
        seen_pids.append(pid)

context = "\n\n---\n\n".join(parent_texts)       # de-duplicated, ordered
answer  = llm.invoke({"context": context, "question": user_query})
```

Sub-queries act as **semantic scouts** — they broaden the retrieval surface to capture
information that a single query phrasing may miss. The BGE Cross-Encoder then acts as a
**signal consolidator** — it re-scores all retrieved child chunks against the original
user intent, ensuring that expanded coverage does not introduce irrelevant noise into the
final parent-chunk context window.
