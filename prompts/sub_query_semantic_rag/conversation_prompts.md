# Conversation Prompts — sub_query_semantic_rag

Prompts used in the Claude Code session to design, build, and extend the sub_query_semantic_rag implementation.

---

**1.** I have a raw text document at data/paul_graham_essay.txt and a LlamaIndex-style evaluation dataset. I ran a baseline evaluation using naive fixed-size chunking (500 chars). Write a Python script to create a RAG pipeline to implement a pure Sub-Query Expansion + Standard Semantic Chunking + BGE Re-ranker architecture (removing the Parent-Child context swap layer).

Follow these exact specifications:

1. PRE-RETRIEVAL (SUB-QUERY EXPANSION)
   - Keep the existing sub-query generation logic using 'claude-haiku-4-5-20251001' to break the user prompt down into 2 to 3 targeted sub-queries.

2. STAGE 1: SEMANTIC RETRIEVAL
   - Use 'text-embedding-3-small' to embed the sub-queries.
   - Execute the vector search directly against your standard 'SemanticChunker' nodes (the 90th percentile blocks). Do not look up or map to separate parent/child IDs.
   - Retrieve 'top_k=15' semantic chunks per sub-query and combine them into a single list, ensuring strict deduplication based on text or chunk ID.

3. STAGE 2: BGE RE-RANKING & GENERATION
   - Pass the deduplicated semantic chunks through 'BAAI/bge-reranker-large' and re-rank them against the ORIGINAL user query.
   - Slicing: Take the top 4 highest-scoring chunks (`top_n_rerank=4`) and pass them directly into the context window for 'claude-haiku-4-5-20251001' to generate the final answer.

4. EXECUTION
   - Ensure the script remains fully integrated with proper logging and the existing rate-limit / sleep delays to protect our automated evaluation loop.

Then, write an evaluation loop that runs the dataset's questions through this basic RAG pipeline, captures the generated answers and retrieved contexts, and outputs the final RAGAS metrics - Faithfulness, Answer Relevance, and Context Precision, DeepEval, LangSmith, Arize Phoenix, TruLens.
Add the initial prompt, and results and comparison in the appropriate folders.

Do not get confused. Earlier we had sub_query_rag which consists of sub query + Parent-child (Semantic rag) + Re-ranker. Now, we are implementing, sub query + Semantic rag + Re-ranker. There is no parent-child swapping in this implementation.
