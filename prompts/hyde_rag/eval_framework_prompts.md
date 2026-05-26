# Evaluation Framework Prompts — hyde_rag

Each framework below sends its own prompts to the LLM when computing scores.
These are the internal templates each framework uses (paraphrased from source).

> **Key distinction from parent_child_semantic_rag:** A HyDE (Hypothetical Document
> Embeddings) pre-retrieval layer is inserted before the dense vector search.
>
> - **HyDE step** — `claude-haiku-4-5-20251001` generates a single realistic
>   autobiographical paragraph answering the user query. This paragraph is embedded
>   with `text-embedding-3-small` and used as the vector query. The original user
>   query is **not** used for embedding.
> - **Retrieval** — Dense vector search runs against child chunk embeddings using
>   the HyDE hypothetical vector (`top_k=25`).
> - **Re-ranking** — `BAAI/bge-reranker-large` scores the 25 retrieved child chunks
>   against the **ORIGINAL user query** (not the hypothetical doc), keeping the
>   relevance signal clean and user-aligned.
> - **Context Swap** — Top-4 child chunks trigger a parent lookup; unique 95th-percentile
>   semantic parent chunks form the final context window.
>
> Evaluation frameworks receive the swapped-in **parent chunk texts** as the retrieved
> contexts — the same de-duplicated parent chunks fed to the LLM.

---

## HyDE System Prompt (Component 1 — Pre-Retrieval)

This prompt is sent to `claude-haiku-4-5-20251001` for every question **before** any
vector retrieval occurs.  Its output is never shown to the end user.

```
You are a document generation assistant. Your task is to write a single, highly
realistic, and cohesive paragraph that answers the following query as if it were an
excerpt from an autobiographical essay by a well-known technology entrepreneur and
essayist. Write with confidence and use plausible narrative details about personal
experiences, formative moments, specific places, people, and intellectual observations
that would naturally appear in such an essay. Match the tone: reflective, direct, and
intellectually honest. Produce exactly one paragraph — no preamble, no title, no
bullet points. This is a HYPOTHETICAL document used strictly for vector similarity
matching and will never be shown to an end user as a factual answer.
```

**User turn:** the raw question from the evaluation dataset.

**Output:** a single paragraph embedded with `text-embedding-3-small` to query the
child chunk index.

---

## RAG QA Prompt (Component 4 — Final Generation)

Sent to `claude-haiku-4-5-20251001` with the de-duplicated parent chunk context and
the **original** user query:

```
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer concisely and accurately based only on the provided context.
```

- `context` = unique 95th-percentile semantic parent chunk texts joined with `\n\n---\n\n`
- `question` = raw original user query (never the HyDE hypothetical document)

---

## RAGAS

### Faithfulness
Asks the LLM to break the answer into atomic statements, then verify each against the
retrieved contexts (semantic parent chunks).

**Step 1 — decompose answer into statements:**
```
Given a question and answer, create one or more statements from each sentence in the given answer.
question: {question}
answer: {answer}
```

**Step 2 — verify each statement against context:**
```
Consider the given context and following statements, then determine whether they are
supported by the information present in the context.
Provide a brief explanation for each statement before arriving at the verdict (Yes/No).
context: {context}
statements: {statements}
```

### Answer Relevancy
Generates N reverse questions from the answer and measures cosine similarity between
them and the original question.
```
Generate {n} questions for the given answer.
answer: {answer}
```

### Context Precision
Checks whether each retrieved parent chunk was actually useful for producing the answer.
```
Given question, answer and context verify if the context was useful in arriving at the given answer.
question: {question}
context: {context}
answer: {answer}
```

---

## DeepEval

### FaithfulnessMetric
Extracts claims from the answer, then verifies each against retrieval contexts (parent chunks).

**Step 1 — extract claims:**
```
Based on the given text, please generate a comprehensive list of FACTUAL CLAIMS that
can be inferred from the provided text.
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
Verifies that retrieved nodes at the top of the ranking are more useful than lower ones.
```
Given the input and expected output, rank the nodes from the retrieval context in order
of relevance.
input: {input}
expected output: {expected_output}
retrieval context: {retrieval_context}
```

---

## LangSmith (LangChain evaluators, local)

### QA Evaluator (correctness)
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
Your task is to determine if the answer provided is faithful to the reference text by
following these guidelines:
- Faithfulness: The answer should only contain information present in the reference text.
Score: (1 for faithful, 0 for not faithful)
```

### CorrectnessEvaluator
```
Compare the following answer to the reference and rate it on correctness on a scale
from 0.0 to 1.0.
[Question]: {input}
[Reference]: {reference}
[Answer]: {output}
```

---

## TruLens (via LiteLLM → Anthropic)

### Relevance (answer relevance)
```
You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given
PROMPT as a score from 0 to 10 where 10 is the most relevant.
PROMPT: {prompt}
RESPONSE: {response}
RELEVANCE:
```

### Context Relevance
```
You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given
QUESTION as a score from 0 to 10 where 10 is the most relevant.
QUESTION: {question}
CONTEXT: {context}
RELEVANCE:
```

### Groundedness (with chain-of-thought)
```
You are a FAITHFULNESS grader. Assess whether the STATEMENT is supported by the CONTEXT.
For each sentence in the statement, reason step-by-step whether the context provides
evidence for it.
CONTEXT: {context}
STATEMENT: {statement}
FAITHFULNESS:
```

---

## BGE Cross-Encoder (not an LLM prompt — internal scoring)

The `BAAI/bge-reranker-large` model does **not** use a natural-language prompt template.
It operates as a classification head on top of a bi-encoder transformer:

- Input: concatenated `[CLS] ORIGINAL_QUERY [SEP] child_chunk_text [SEP]` token sequence
- Output: a single scalar relevance logit (higher = more relevant)
- **Query used: ORIGINAL user query** — never the HyDE hypothetical document
- No prompt engineering applies; performance comes from the model's pre-training on
  MS MARCO and BEIR-style retrieval tasks.
- The top-4 scoring child chunks trigger the **Context Swap** — their `parent_id`
  metadata is used to retrieve the full semantic parent chunk from the in-memory store.

---

## HyDE Retrieval Flow (pipeline-level, not an LLM prompt)

```python
# Component 1 — HyDE generation
hyp_doc = generate_hypothetical_document(user_query)   # → one paragraph

# Component 2 — hypothetical-vector search
hyp_vec          = embeddings.embed_query(hyp_doc)     # NOT embed_query(user_query)
broad_child_docs = store.similarity_search_by_vector(hyp_vec, k=25)

# Component 3 — cross-encoder re-rank vs ORIGINAL query
top_child_docs = rerank_documents(user_query, broad_child_docs, top_n=4)

# Component 4 — parent context swap
for child_doc in top_child_docs:
    pid = child_doc.metadata["parent_id"]
    if pid not in seen_pids:
        parent_texts.append(parent_store[pid])         # full 95th-pct semantic parent
        seen_pids.append(pid)

context = "\n\n---\n\n".join(parent_texts)             # de-duplicated, ordered
answer  = llm.invoke({"context": context, "question": user_query})
```

The HyDE hypothetical paragraph acts as a **semantic bridge** — its dense embedding
lands closer to relevant essay passages than a short question's embedding would.
The BGE Cross-Encoder then acts as a **precision filter**, using the true user query
to verify that the HyDE-retrieved child chunks are genuinely relevant before the parent
swap expands them into full narrative context blocks.
