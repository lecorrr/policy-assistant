# Changelog

## 0.1.1 — 2026-11-10

- Fixed: a failing judge was hidden. With no Anthropic credit, a trial printed "faithfulness
  not judged, errors 0" while every judge call had failed, and a judge failing on some answers
  would have averaged faithfulness over the rest without saying so. The judge now gets one test
  call before any model runs, and a judge without credit or access stops the run there. Later
  judge failures print a `judge ERROR` line, the summary says how many answers were judged
  ("46 of 48 judged, 2 judge errors"), the memo shows the same, and the command exits with 1.
- `eval-retrieval` prints a line per question with each mode's rank, says when the reranker
  loads (its first run downloads about 0.6 GB), and announces the leak test, instead of staying
  silent for minutes.
- README: clear PYTHONPATH before installing, not only before the tests; a two-question trial
  before the full run; the full run costs about $1.50, not "well under a dollar".

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
