# Initial Prompt — semantic_rag

The prompt used to generate the semantic_rag implementation (`semantic_rag_eval.py`).

---

I have a raw text document at data/paul_graham_essay.txt and a LlamaIndex-style evaluation dataset. I ran a baseline evaluation using naive fixed-size chunking (500 chars), which resulted in a Context Precision score of 0.0. Write a Python script using LangChain (or LlamaIndex) and RAGAS that loads this document, performs SemanticChunker from langchain-experimental.text_splitter. Configure the chunker to use OpenAIEmbeddings(model="text-embedding-3-small") and set the breakpoint_threshold_type to "percentile" (use a 90th percentile threshold as a starting point). Index these semantic chunks into an in-memory vector database. Then, write an evaluation loop that runs the dataset's questions through this basic RAG pipeline, captures the generated answers and retrieved contexts, and outputs the final RAGAS metrics - Faithfulness, Answer Relevance, and Context Precision, DeepEval, LangSmith, Arize Phoenix, TruLens.
