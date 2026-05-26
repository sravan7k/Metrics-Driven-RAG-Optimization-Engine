# Conversation Prompts — hyde_rag

Prompts used in the Claude Code session to design, build, and extend the hyde_rag implementation.

---

**1.** I have a raw text document at data/paul_graham_essay.txt and a LlamaIndex-style evaluation dataset. I ran a baseline evaluation using naive fixed-size chunking (500 chars). Write a Python script to create a RAG pipeline to implement a Hypothetical Document Embeddings (HyDE) pre-retrieval layer. This layer must integrate seamlessly with our existing Hierarchical Semantic Parent-Child chunking structure and BAAI/bge-reranker-large Cross-Encoder pipeline.

Follow these strict technical and architectural specifications:

1. COMPONENT 1: HYDE GENERATION LAYER (Pre-Retrieval)
   - Implement a function `generate_hypothetical_document(user_query: str) -> str` that calls 'claude-haiku-4-5-20251001'.
   - System Prompt Instructions: The LLM must generate a single, highly realistic, and cohesive paragraph answering the user's query as if it were an excerpt from an autobiographical essay. Instruct the model to write with confidence and use plausible narrative details, explicitly stating that this is a hypothetical document used strictly for vector matching.

2. COMPONENT 2: HYDE-DRIVEN VECTOR SEARCH (Stage 1)
   - When a user query arrives, pass it to `generate_hypothetical_document` to receive the mock answer string.
   - Pass this HYPOTHETICAL paragraph (not the original user query) to the 'text-embedding-3-small' embedding model.
   - Execute a dense vector similarity search against your indexed Child Chunks using the hypothetical vector, pulling back the top 25 closest matches (`top_k_retrieve=25`).

3. COMPONENT 3: ORIGINAL-QUERY CROSS-ENCODER RE-RANKING (Stage 2)
   - Crucial Step: Take the 25 child chunks retrieved via the HyDE vector and pass them into the local 'BAAI/bge-reranker-large' Cross-Encoder model.
   - Score and re-rank these chunks against the ORIGINAL user query (do not use the hypothetical document for re-ranking, as we want to measure true alignment with what the user actually asked).
   - Sort the child chunks based on the Cross-Encoder scores and slice the absolute top performers (set `top_n_rerank=4`).

4. COMPONENT 4: PARENT CONTEXT SWAP & FINAL INFERENCE
   - Map the winning re-ranked child nodes back to their 95th-percentile Semantic Parent Chunks.
   - Deduplicate the Parent Chunks to protect the context window from redundant text blocks.
   - Synthesize the final context payload and pass it to 'claude-haiku-4-5-20251001' alongside the ORIGINAL user query for the final factual generation.

5. ENGINE STABILITY & LOGGING TRACES
   - Provide a fully unified, complete Python script. Do not use placeholders or blank mock code blocks.
   - Retain the exponential backoff / sleep delay logic during the final generation step to protect our evaluation suite from Anthropic 529 Overloaded exceptions.
   - Add clear logging or print statements tracing the query: print the generated hypothetical document, the IDs of the retrieved child chunks, and the text of the final parent blocks dispatched to Claude.

Then, write an evaluation loop that runs the dataset's questions through this basic RAG pipeline, captures the generated answers and retrieved contexts, and outputs the final RAGAS metrics - Faithfulness, Answer Relevance, and Context Precision, DeepEval, LangSmith, Arize Phoenix, TruLens.
Add the initial prompt, and results and comparison in the appropriate folders.
