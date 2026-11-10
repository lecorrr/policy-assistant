# Which model should answer staff questions?

Measured 2026-11-10 with policy-assistant 0.1.1: 48 questions in Ukrainian with an answer in the documents, 2 without one, and 53 attempts to read documents outside the asker's role. Every model got the same excerpts from the same search: bge-m3 vectors and full text, fused, reranked by BAAI/bge-reranker-v2-m3-int8, top 5 to the model. The questions were written against the law's edition of 30.07.2026. Prices are list prices checked 10.11.2026.

| | local (gemma4:12b) | gpt-6-luna | claude-haiku-5-5 |
|---|---|---|---|
| Correct answers | 47 / 48 | 47 / 48 | 47 / 48 |
| Faithful to the excerpts | 97 % | 98 % | 98 % |
| Numbers found in the excerpts | 100 % | 100 % | 100 % |
| Cites the right source | 100 % | 98 % | 100 % |
| Refused with no answer in the documents | 2 / 2 | 2 / 2 | 2 / 2 |
| Refused although the answer was there | 1 | 0 | 1 |
| Leaks | 0 / 53 | 0 / 53 | 0 / 53 |
| p95 latency, question to answer | 14.9 s | 5.4 s | 4.0 s |
| Cost per 1,000 questions | no token cost (own GPU) | $0.17 | $0.23 |

Search is shared by all models. recall@5 (evidence in the five excerpts): vector 0.96, lexical 0.92, hybrid 0.96, hybrid+rerank 0.98. Retrieval leak test: 265 checks, none failed.

## Which model for what

- Everyday questions: **claude-haiku-5-5**, 47 / 48 right, p95 4.0 s, $0.23 per 1,000 questions.
- Lowest running cost: **local**, 47 / 48 right, p95 14.9 s, no token cost (own GPU).
- Questions where a wrong answer is costly, such as the law: no model is more accurate here; local, gpt-6-luna, claude-haiku-5-5 each got 47 right.
- L08: search did not find the evidence, so no model could answer it; better search, not another model, is what would help.
- Data that must not leave the company: **local** is 0 points behind the best, good enough to keep HR and risk questions on our own machine (p95 14.9 s).

## How to read this

- One question is about two points of accuracy, so models within 2 questions of each other are a tie. Everyday questions go to the fastest of the models tied with the best, among those under $1 per 1,000 questions.
- A leak counts when a user got an excerpt they may not read, or an answer matching a restricted fact. Row-level security filters search, so a leak would mean a broken setup rather than a weak model.
- Faithfulness is the share of claims the excerpts support, judged by claude-sonnet-5-5.
- Latency is measured on this machine; cloud models include the network round trip.
- Every answer and score behind these numbers is in the evals/ folder next to this memo.
