"""Retrieval-augmented generation over local documents and the web.

``pcb_assistant.rag.pipeline.answer`` is the entry point: a small router model picks the
source (default corpus, PDF corpus or live web search) and the matching retriever answers.
"""
