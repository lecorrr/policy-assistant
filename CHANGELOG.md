# Changelog

## 0.1.0 — 2026-11-10

First version.

- Corpus: law 1734-VIII parsed from the Rada's open-data text (articles, parts, the final
  provisions' points, annexes; editorial notes dropped; flattened superscripts restored), and
  15 invented policies for sales, risk, HR and all staff, with section-level roles.
- Postgres with pgvector: row-level security by role for a reader that can only select;
  ingest as the admin, in one transaction, reusing stored embeddings.
- Search: bge-m3 vectors and Ukrainian full text (pymorphy3 lemmas, every reading of an
  ambiguous word), reciprocal rank fusion, bge-reranker-v2-m3 through fastembed.
- Answers with `[n]` citations from the local model (gemma4:12b), OpenAI or Anthropic models
  through LangChain; a fixed refusal when the excerpts do not answer.
- Web page and HTTP API with signed sessions; the CLI for every step; `doctor`.
- Evaluation: 50 questions with evidence and answer patterns, recall@5 for four search modes,
  50 leak cases and 3 injections, faithfulness judge, latency and cost; the model-choice memo.
