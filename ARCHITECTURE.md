# Architecture

The policy assistant answers staff questions in Ukrainian from two kinds of source: the
Consumer Lending law (Закон № 1734-VIII, readable by everyone) and invented internal policies
for sales, risk and HR. A user only ever gets answers from chunks their roles may read, and
that rule is enforced by Postgres row-level security before any text reaches a model.

About 4,000 lines of Python in one package, plus a static web page. The parts that change most
often (which model answers, which embedder or reranker, where the law comes from) each live in
one adapter module.

## Data flow

```
ingest:   law TXT (data.rada.gov.ua) ─┐
          corpus/policies/*.md ───────┴─> parse ─> chunk ─> embed (Ollama bge-m3) ─> Postgres
                                                     (lemmas -> tsvector)              as admin

question: user (session cookie) ─> roles ─> Postgres as policy_reader, SET LOCAL app.roles
            ├─ vector search (pgvector HNSW, cosine)  ─┐   row-level security drops every row
            └─ full-text search (lemma tsvector)      ─┴─> whose roles miss the user's roles
                 ─> reciprocal rank fusion ─> cross-encoder rerank ─> top 5 excerpts
                 ─> answer model (local / OpenAI / Claude) ─> answer with [n] citations
```

## Modules

| Module | Owns | May use |
|---|---|---|
| `contracts.py` | shared dataclasses: `Document`, `Block`, `ChunkDraft`, `Hit`, `Answer`, `User` | nothing |
| `text.py` | normalisation, Ukrainian lemmas (pymorphy3), tsvector and tsquery literals | contracts |
| `corpus/` | parsing the law TXT and the policy Markdown, chunking with citation labels | contracts, text |
| `ranking.py` | reciprocal rank fusion | contracts |
| `answering.py` | prompt, answer parsing, citations; the `Assistant` service | contracts, ranking via `search.py` |
| `search.py` | the `Retriever`: embed, search under the user's roles, fuse, rerank | contracts, text, ranking |
| `ingest.py` | parse, chunk, embed and store the corpus | corpus, contracts |
| `config.py` | settings from `.env`, `config/models.yaml`, `config/users.yaml`, all validated | contracts |
| `auth.py` | password hashes and signed session tokens | nothing |
| `adapters/db.py` | the schema, row-level security, writes as admin, role-scoped reads | contracts |
| `adapters/embeddings.py` | Ollama `/api/embed` | contracts |
| `adapters/reranker.py` | fastembed cross-encoder, or none | contracts |
| `adapters/chat.py` | LangChain chat models for Ollama, OpenAI and Anthropic, token usage | config |
| `adapters/rada.py` | downloading the law from the Rada open-data API | nothing |
| `fakes.py` | hashing embedder, overlap reranker, extractive answerer for tests and `--fake` | text, contracts |
| `evaluation/` | the 50-question set, retrieval and answer scoring, leak tests, the memo | everything above |
| `web/` | FastAPI routes and the static page | answering, auth, config |
| `wiring.py` | builds the retriever and the assistant from settings, real or `--fake` | adapters, services |
| `doctor.py` | the checks `doctor` runs, each reported, none fatal | adapters, config |
| `cli.py` | the commands and their arguments | everything |

Rules:

- `contracts`, `text`, `corpus`, `ranking` and the prompt functions in `answering` do no I/O and
  import no database, HTTP, model or web library. `tests/test_boundaries.py` fails if they do.
- Only `adapters/db.py` writes SQL. Only `adapters/chat.py` imports LangChain.
- The services (retriever, assistant, ingest) receive their adapters as arguments, so tests hand
  them fakes; `wiring.py` builds the real ones for the commands and the web app.

## Access control

Every chunk row carries `allowed_roles text[]`. The app connects as `policy_reader`, a role that is
not a superuser, has no `BYPASSRLS` and does not own the tables. Each search runs in one
transaction that first calls `set_config('app.roles', '{staff,sales}', true)`; the policy is

```sql
USING (allowed_roles && app_current_roles())
```

so a missing or empty setting returns no rows at all. The retriever also checks every returned
row against the user's roles and raises if one slips through, which would mean the database was
set up wrong (for example, the app connected as the admin). `doctor` checks the reader role.

Roles come from `config/users.yaml` after sign-in, never from the request body. Every user also
has `staff`, the role of documents meant for everyone. A policy section can narrow its roles with
`<!-- roles: hr -->` under its heading; `sales-bonus.md` uses it for the payroll section.

What this does not cover: anyone who can run SQL as the reader can call `set_config` too. The
policy guards against the app forgetting the filter, not against a stolen reader password, so
that password stays in `.env` on the server and no request carries SQL. Isolating users from a
compromised app would take one database role per user, which this demo does not need.

## Search

- Vector: `bge-m3` (1024 dimensions) through Ollama; HNSW index, cosine distance. Row-level
  security filters after the index scan, so each search sets `hnsw.iterative_scan =
  relaxed_order` and the index keeps scanning until enough permitted rows turn up.
- Full text: Postgres has no Ukrainian dictionary, so lemmas come from pymorphy3 in Python and go
  in as a ready `tsvector` literal. Every reading of an ambiguous word is kept ("пені" is "пеня" or
  "пен"), on both sides. Queries are built the same way, so search does not depend on the
  database's locale or parser. Ranking is `ts_rank` with heading words counting double, chosen
  by measurement (see the comment above `_LEXICAL_SQL`).
- Fusion: reciprocal rank fusion (k = 60) of the two lists, 30 candidates each.
- Rerank: `BAAI/bge-reranker-v2-m3-int8` through fastembed on the CPU, over the top 20; the top 5
  go to the model. `POLICY_RERANKER=none` turns it off for comparison.

## Answers

The model gets the five excerpts, numbered, and must cite them as `[n]`. When the excerpts do not
answer the question it replies `НЕМАЄ_ВІДПОВІДІ`, which the app turns into a fixed refusal.
Citations to numbers it was not given are dropped and counted. Excerpt text is data: the system
prompt tells the model to ignore instructions inside it.

## Evaluation

`evals/questions.yaml`: 50 questions with the evidence sentence for each and regular expressions
a correct answer must match, the users each restricted question is a leak test for, and three
prompt-injection attempts.

- `eval-retrieval`: recall@5 and MRR for vector, full text, hybrid and hybrid + rerank, and the
  retrieval leak test: every question as every user, nothing outside their roles, nothing from a
  restricted source for its leak users.
- `eval-answers`: each model answers the 50 questions, the 50 leak cases and the 3 injections;
  correctness, refusals, citation checks, numbers found in the excerpts, an optional judge for
  faithfulness, p50/p95 latency, tokens and cost per 1,000 questions. The judge is checked with
  one call before the first question, and judge failures are counted like answer errors.
- `memo`: renders `docs/model-choice-memo.md` from the latest reports and keeps those reports in
  `docs/evals/`, so the evidence is committed with the memo while `data/` stays out of git.

## Recipes

- Another policy: add `corpus/policies/<id>.md` with front matter (`id`, `title`, `roles`, `owner`,
  `version`, `effective`) and numbered `##` sections, then run `ingest`.
- Another role: add it to `roles` in `config/users.yaml`; documents and users may then use it.
- Another answer model: add an entry to `config/models.yaml` with its provider, options and prices.
- Another provider: one branch in `adapters/chat.py` and its LangChain package in `pyproject.toml`.
- Another embedder: a class with `embed(texts) -> list[list[float]]`; change `POLICY_EMBEDDING_DIMS`
  and run `init-db --reset` if the size differs.
