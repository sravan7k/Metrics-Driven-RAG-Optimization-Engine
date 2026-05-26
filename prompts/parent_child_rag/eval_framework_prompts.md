# Evaluation Framework Prompts — parent_child_rag

Each framework below sends its own prompts to the LLM when computing scores.
These are the internal templates each framework uses (paraphrased from source).

> **Key distinction from reranker_rag:** The retrieval stage now performs a
> **Context Swap** after BGE reranking. The 25 retrieved *child* chunks are scored
> by the Cross-Encoder; for the top-3 winners, the full *parent* chunk is fetched
> via `parent_id` metadata. Evaluation frameworks receive these swapped-in
> **parent chunk texts** as the retrieved contexts — not the compact child chunks.
> This gives the LLM (and evaluators) a richer, more complete passage to reason over.

---

## RAGAS

### Faithfulness
Asks the LLM to break the answer into atomic statements, then verify each statement
against the retrieved contexts (parent chunks in this pipeline).

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
Extracts claims from the answer, then verifies each claim against retrieval contexts (parent chunks).

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
- The top-3 scoring child chunks trigger the **Context Swap** — their `parent_id`
  metadata is used to retrieve the full parent chunk text from the in-memory store.

---

## Parent-Child Context Swap (pipeline-level, not an LLM prompt)

```
for child_doc in top_child_docs:           # top-3 after BGE rerank
    pid = child_doc.metadata["parent_id"]
    if pid not in seen_pids:
        parent_texts.append(parent_store[pid])   # full 1500-char parent block
        seen_pids.append(pid)

context = "\n\n---\n\n".join(parent_texts)       # de-duplicated, ordered
answer  = llm.invoke({"context": context, "question": query})
```

The child chunks act as **precision anchors** — their compact size ensures the
vector search and cross-encoder score the most topically relevant passages.
The parent chunks act as **context healers** — their larger size provides the LLM
with complete, coherent narrative blocks that eliminate mid-sentence truncation
artefacts common in fixed-size chunking.
