# Initial Prompt — parent_child_semantic_rag

The prompt used to generate the parent_child_semantic_rag implementation (`parent_child_semantic_rag_eval.py`).

---

I have a raw text document at data/paul_graham_essay.txt and a LlamaIndex-style evaluation dataset. I ran a baseline evaluation using naive fixed-size chunking (500 chars). Write a Python script using LangChain (or LlamaIndex) and RAGAS that loads this document. Implement a Parent-Child (Small-to-Large) semantic chunking strategy, integrated with our existing BAAI/bge-reranker-large Cross-Encoder model.

1. CHUNKING & STORAGE STRATEGY (Parent-Child Decoupling)
 - Implement a data ingestion pipeline that processes documents into two layers:
     * Child Chunks: performs SemanticChunker from langchain-experimental.text_splitter. Configure the chunker to use OpenAIEmbeddings(model="text-embedding-3-small") and set the breakpoint_threshold_type to "percentile" (use a 90th percentile threshold as a starting point).
     * Parent Chunks: Larger blocks of text capturing complete narrative blocks or logical sections (use a 90th percentile threshold as a starting point).
   - In your Vector Database setup (using Chroma, FAISS, or your current vector store), embed and index ONLY the Child Chunks using 'text-embedding-3-small'.
   - Ensure each Child Chunk's metadata contains a clear reference link or ID (`parent_id`) pointing back to its corresponding Parent Chunk text stored in an in-memory dictionary or document store.

2. TWO-STAGE RETRIEVAL & SWAP LOGIC
 - Step 1 (Broad Child Sweep): When a user query enters, perform a broad semantic vector search against the Child Chunks, pulling back the top 25 most similar child nodes (`top_k_retrieve=25`).
   - Step 2 (Cross-Encoder Re-ranking): Feed these 25 hyper-focused Child Chunks into the 'BAAI/bge-reranker-large' Cross-Encoder model against the user query. Compute the bidirectional attention scores and sort them.
   - Step 3 (The Context Swap / "Healing" Step): Take the top 3 highest-scoring Child Chunks (`top_n_rerank=3`). For these 3 winning child nodes, use their `parent_id` metadata to look up and retrieve their full, unbroken Parent Chunks.

3. FINAL GENERATION WINDOW
   - De-duplicate the retrieved Parent Chunks (in case multiple winning child nodes belong to the same parent).
   - Combine the text of these unique Parent Chunks to form the clean, linear context payload.
   - Pass this complete context payload to the final LLM generation engine ('claude-haiku-4-5-20251001').

4. CODE QUALITY & SYSTEM STABILITY
   - Provide a fully integrated, functional Python script. Do not use placeholders or mock definitions.
   - Include print statements tracing a query through the process: show the text of the top matching child chunks, and then print out the larger parent chunk texts that are ultimately swapped in for generation.

Then, write an evaluation loop that runs the dataset's questions through this basic RAG pipeline, captures the generated answers and retrieved contexts, and outputs the final RAGAS metrics - Faithfulness, Answer Relevance, and Context Precision, DeepEval, LangSmith, Arize Phoenix, TruLens.
Add the initial prompt, and results and comparison in the appropriate folders.
Add the relevant details under parent child semantic rag. Do not confuse with parent child rag.
