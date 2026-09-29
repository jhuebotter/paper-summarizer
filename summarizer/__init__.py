"""paper-summarizer — structured, critical summaries of research PDFs.

Extracts PDF text (docling or pypdf), asks an OpenAI-compatible LLM
(OpenRouter by default, or a local server such as LM Studio) for one JSON
object per paper, validates it with pydantic, and renders markdown.
"""

__version__ = "0.2.0.dev0"
