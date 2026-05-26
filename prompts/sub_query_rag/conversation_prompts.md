# Conversation Prompts — sub_query_rag

Prompts used in the Claude Code session to design, build, and extend the sub_query_rag implementation.

---

**1.** I have a raw text document at data/paul_graham_essay.txt and a LlamaIndex-style evaluation dataset. I ran a baseline evaluation using naive fixed-size chunking (500 chars). Write a Python script to create a RAG pipeline to implement a Query Transformation layer using Sub-Query Expansion. This layer must sit directly on top of our existing Hierarchical Semantic Parent-Child structure and BAAI/bge-reranker-large Cross-Encoder pipeline.

Follow these precise architectural and technical specifications:

1. COMPONENT 1: SUB-QUERY EXPANSION LAYER (Pre-Retrieval)
   - Implement a function `generate_sub_queries(user_query: str) -> list[str]` that calls 'claude-haiku-4-5-20251001' or a fast LLM using OpenAI's Structured Outputs (Pydantic parsing) to ensure a guaranteed JSON list format.
   - The system prompt must instruct the LLM to analyze the user's input and break it down into 2 to 3 distinct, hyper-focused sub-queries. 
   - Ensure it handles comparative queries perfectly (e.g., If the user asks "Compare Paul Graham's experience at Interleaf vs Viaweb", it should break it down into: ["Paul Graham Interleaf work experience", "Paul Graham Viaweb startup experience"]). If a query is simple, it can return the original query alongside a single alternative semantic phrasing.

2. COMPONENT 2: SCALED VECTOR RETRIEVAL (Stage 1)
   - For a given user query, generate the sub-queries. 
   - Execute a dense vector search using 'text-embedding-3-small' against your Child Chunks for the original query AND each generated sub-query.
   - For each query execution, retrieve the top 15 child nodes. 
   - Combine all retrieved child chunks into a unified list and implement a strict de-duplication step based on unique child node IDs to handle overlap safely.

3. COMPONENT 3: BGE CROSS-ENCODER RE-RANKING (Stage 2)
   - Pass the complete de-duplicated list of child chunks through the 'BAAI/bge-reranker-large' Cross-Encoder model against the ORIGINAL user query.
   - Compute bidirectional attention scores, sort the child chunks in descending order of relevance, and slice the absolute top performers (set `top_n_rerank=3` or `4`).

4. COMPONENT 4: THE PARENT SWAP & INFERENCE
   - Take the winning re-ranked child nodes and use their `parent_id` metadata to look up and pull their corresponding 95th-percentile Semantic Parent Chunks.
   - De-duplicate the parent chunks to ensure no repetitive text blocks enter the context window.
   - Synthesize the final context payload and pass it to 'claude-haiku-4-5-20251001' for final answer generation.

5. CODE QUALITY & EXECUTION TRACING
   - Provide a fully complete, ready-to-run Python script. Do not use placeholders, empty blocks, or mock definitions.
   - Ensure the exponential backoff / sleep delay is retained during the final generation step to protect our automated evaluation pipeline from Anthropic 529 Overloaded exceptions.
   - Add clear print/logging traces showing: the original query, the generated sub-queries, the number of child chunks retrieved before vs. after de-duplication, and the final parent chunks pushed to Claude.

Capture the prompts as per previous implementations.
