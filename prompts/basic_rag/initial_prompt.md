# Initial Prompt — basic_rag

The prompt used to generate the basic_rag implementation (`basic_rag_eval.py`).

---

I have a raw text document at data/paul_graham_essay.txt and a LlamaIndex-style evaluation dataset. Write a Python script using LangChain (or LlamaIndex) and RAGAS that loads this document, performs basic fixed-size character chunking (500 chars, 50 overlap), and indexes it into an in-memory vector database. Then, write an evaluation loop that runs the dataset's questions through this basic RAG pipeline, captures the generated answers and retrieved contexts, and outputs the final RAGAS metrics - Faithfulness, Answer Relevance, and Context Precision, DeepEval, LangSmith, Arize Phoenix, TruLens.
In the next iterations, I will also implement RAG with advanced techniques, the above output on various frameworks (RAGAS, DeepEval, LangSmith, Arize Phoenix, TruLens) should be compared. Store the output on the various frameworks accordingly.
