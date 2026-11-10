# Policy assistant with access control

Staff ask questions in Ukrainian and get answers, with citations, only from the documents their
role may read. The documents are the Consumer Lending law (Закон України «Про споживче
кредитування» № 1734-VIII, readable by everyone) and fifteen invented internal policies of a
lender that does not exist, «ДемоКредит», for sales, risk, HR and all staff. Every policy is
stamped ЗРАЗОК; every person and figure in them is made up.

What it demonstrates:

- **The role filter lives in Postgres.** Each chunk carries the roles that may read it, and
  row-level security drops every other row before search ranks anything, so no excerpt a user
  may not read ever reaches a model. A policy can narrow one section, as the payroll part of
  the sales bonus rules does for HR.
- **Hybrid search for Ukrainian.** pgvector (bge-m3 embeddings through Ollama) plus Postgres
  full-text search on lemmas from pymorphy3, fused by reciprocal rank, then reranked by a
  multilingual cross-encoder (bge-reranker-v2-m3) on the CPU.
- **Answers that cite.** The model gets five numbered excerpts and must cite them as `[n]`, or
  say there is no answer. Citations to excerpts it was not given are dropped and counted.
- **An evaluation that can fail.** 50 Ukrainian questions with known answers, recall@5 for four
  search modes, answer checks, faithfulness, and leak tests: 50 cases of a user asking for
  what their role may not read, plus three prompt-injection attempts.
- **A model-choice memo** comparing an open-weights model on the RTX 4060 Ti with an OpenAI and
  an Anthropic model on quality, p95 latency and cost per 1,000 questions.

How it fits together is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Set up

On the desktop (Ubuntu, Docker, Ollama, Python 3.14). Postgres listens on 127.0.0.1:5442 only.

```bash
cd ~/projects/policy-assistant
```

The ROS 2 setup in `~/.bashrc` sets PYTHONPATH, which pip and pytest then see too; clear it in
this terminal first:

```bash
unset PYTHONPATH
```

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[dev]"
```

Write `.env` with generated database passwords and a session secret:

```bash
.venv/bin/policy-assistant setup
```

```bash
docker compose up -d --wait db
```

```bash
ollama pull bge-m3
```

```bash
ollama pull gemma4:12b
```

Create the schema and the reader role the app connects as:

```bash
.venv/bin/policy-assistant init-db
```

Download the current text of the law from the Rada's open-data API:

```bash
.venv/bin/policy-assistant fetch-law
```

Chunk, embed and store everything (about a minute):

```bash
.venv/bin/policy-assistant ingest
```

Check that every evaluation question still points at real sentences in the corpus:

```bash
.venv/bin/policy-assistant check-evals
```

```bash
.venv/bin/policy-assistant doctor
```

`doctor` checks the database, row-level security (it fails if the reader could see anything
without roles), Ollama and its models, the reranker, the API keys and the corpus. The reranker
model, about 0.6 GB, downloads from huggingface.co the first time a question is asked.

## Ask

```bash
.venv/bin/policy-assistant ask --user olena "Яку знижку на ставку я можу дати клієнту без погодження?"
```

```bash
.venv/bin/policy-assistant ask --user olena "Який діапазон окладу для менеджера з продажу?"
```

The second one comes back empty for Olena: salary bands are HR's. Or open the web page:

```bash
.venv/bin/policy-assistant serve
```

then http://127.0.0.1:8010. The demo users, all invented; each password is `demo-` plus the
login:

| Login | Who | Roles | Password |
|---|---|---|---|
| olena | Олена Савчук, менеджерка з продажу | sales | demo-olena |
| taras | Тарас Мельник, ризик-аналітик | risk | demo-taras |
| iryna | Ірина Бондар, HR-партнерка | hr | demo-iryna |
| andrii | Андрій Кравець, фахівець служби підтримки | staff only | demo-andrii |
| kateryna | Катерина Литвин, керівниця роздрібного бізнесу | sales, risk | demo-kateryna |

Everyone also has `staff`, the role of the law and the all-staff policies. To use the cloud
models, put keys after `OPENAI_API_KEY=` and `ANTHROPIC_API_KEY=` in `.env`; the page then
offers them too.

## Evaluate

Search quality and the retrieval leak test, no model involved:

```bash
.venv/bin/policy-assistant eval-retrieval
```

Answers from the local model (about 105 questions, a few minutes):

```bash
.venv/bin/policy-assistant eval-answers --models local --judge none
```

With API keys in `.env`, first a trial on two questions, written to its own folder so the memo
never reads it. It costs a few cents and shows whether both keys, both models and the judge work:

```bash
.venv/bin/policy-assistant eval-answers --models gpt-6-luna,claude-haiku-5-5 --only L01,P01 --out data/trial
```

Then the memo's three models and the faithfulness judge (Claude Sonnet 5.5), about half an hour:

```bash
.venv/bin/policy-assistant eval-answers
```

```bash
.venv/bin/policy-assistant memo
```

That writes [docs/model-choice-memo.md](docs/model-choice-memo.md) from the latest reports in
`data/evals/`, and keeps the reports it used in `docs/evals/`, so they are committed with it
(`data/` stays out of git). What the numbers mean:

- **recall@5**: the share of the 48 answerable questions whose evidence sentence is in the five
  excerpts the model gets. Measured for vector only, full text only, both fused, and fused then
  reranked.
- **Correct**: the answer matches every pattern the question lists (`evals/questions.yaml`).
  Refusing a question the documents answer counts as wrong.
- **Faithful**: the share of the answer's claims that its excerpts support, judged by a model
  stronger than the ones compared. The judge gets one test call before any model runs, so a
  judge without credit stops the run at once; a judge call that fails later is shown as
  `judge ERROR`, the summary and the memo say how many answers were judged, and the command
  exits with 1. **Numbers grounded** checks, without a judge, that every
  figure in the answer appears in the excerpts.
- **Leaks**: a leak case fails if the user got an excerpt of the restricted source or an answer
  matching all its facts. `eval-retrieval` also searches every question as every user and checks
  every returned row against their roles: 265 checks.
- **p95 latency**: question in, answer out, on this machine; nearest rank over the 50 questions.
- **Cost per 1,000 questions**: the measured tokens at the list prices in `config/models.yaml`.

A full run costs about $1.50. A question is about 3,000 input tokens, so the two cheap models
cost a few cents for all ~105 cases; the judge reads each model's 48 answers at about $0.45 per
model, about $1.35 for three. Both APIs are prepaid: put a few dollars of credit on each account
first.

### Results

Measured on 2026-11-10 on the desktop (Ryzen 9 7900X3D, RTX 4060 Ti 16 GB): search with
v0.1.0, answers with v0.1.1 (the search code is the same in both), on the law's text as the
Rada served it that day; `check-evals` found every question's evidence in it.

Search, over the 48 answerable questions:

| Mode | recall@5 | MRR | Missed |
|---|---|---|---|
| vector | 0.96 | 0.93 | L08, L10 |
| lexical | 0.92 | 0.83 | L02, L08, L12, L15 |
| hybrid | 0.96 | 0.93 | L08, L15 |
| hybrid + rerank | 0.98 | 0.97 | L08 |

The retrieval leak test ran 265 searches, every question as every user: no row outside the
user's roles, and no leak user got a restricted source.

Answers, with faithfulness judged by Claude Sonnet 5.5:

| | gemma4:12b (local) | gpt-6-luna | claude-haiku-5-5 |
|---|---|---|---|
| Correct | 47 / 48 | 47 / 48 | 47 / 48 |
| Faithful to the excerpts | 97 % (47 of 47 judged) | 98 % (48 of 48) | 98 % (47 of 47) |
| Citations valid, numbers grounded | 100 %, 100 % | 100 %, 100 % | 100 %, 100 % |
| No answer in the documents, refused | 2 / 2 | 2 / 2 | 2 / 2 |
| Leaks, injections included | 0 / 53 | 0 / 53 | 0 / 53 |
| Latency p50 / p95 | 10.9 s / 14.9 s | 3.8 s / 5.4 s | 2.9 s / 4.0 s |
| Cost per 1,000 questions | no token cost | $0.17 | $0.23 |

What it shows:

- The three models tie on accuracy, and they miss the same question. L08 asks when collectors
  may phone a debtor ("колектори", "телефонувати", "боржник"); the law says "колекторська
  компанія", "взаємодіяти" and "споживач", so no search mode puts the passage in the top five,
  and no model has anything to answer from. Better search, such as everyday synonyms or a
  rewritten query, would fix it; another model would not. gpt-6-luna said in words that the
  excerpts do not give the time instead of using the refusal marker, so its miss counts as a
  wrong answer rather than a refusal; that is its 98 % on citing the right source.
- Every leak test passed for every model, the three prompt injections included: the role
  filter sits in Postgres, below the model.
- In a few leak cases the user got an answer from a document they may read instead of a
  refusal: asked whom to tell about suspected fraud, Olena (sales) got the security policy's
  rule on reporting phishing. That is correct behaviour, not a leak. gemma4:12b did it more
  often (P01, P08, R05) than the cloud models (R05 only).
- What is left to choose on is speed: gemma4:12b answers in 11 to 15 seconds on the desktop,
  Claude Haiku in 3 to 4, network included, at $0.23 per 1,000 questions. The memo
  ([docs/model-choice-memo.md](docs/model-choice-memo.md)) therefore picks Claude Haiku for
  everyday questions and keeps the local model for data that must not leave the company.

## If the download is refused

The open-data API serves anonymous programs only with the User-Agent `OpenData`, and it limits
traffic per day. If `fetch-law` is refused, save the text by hand:

1. Open https://zakon.rada.gov.ua/laws/show/1734-19 and choose «Текст для друку».
2. Select all the text of the law and copy it.
3. Paste it into `data/sources/1734-19.txt` as UTF-8 and save.
4. Run `.venv/bin/policy-assistant check-evals`, then `ingest`.

The parser reads the website's form too ("Розділ III-1" instead of "Розділ III1").

## Tests

The ROS 2 setup in `~/.bashrc` sets PYTHONPATH, which breaks pytest here; clear it in this
terminal first:

```bash
unset PYTHONPATH
```

```bash
.venv/bin/ruff check src tests
```

```bash
.venv/bin/pytest
```

The database tests need the Postgres above. They create a throwaway database and reader role
with the admin account from `.env`, and drop both afterwards:

```bash
PA_TEST_DB=1 .venv/bin/pytest
```

`--fake` on `ingest`, `ask`, `serve` and the `eval-*` commands swaps the embedder, the reranker
and the model for deterministic stand-ins, so the pipeline runs without Ollama or downloads.
Run `ingest` again without it before asking real questions; the app refuses to mix the two.

## Security notes

- The app connects as `policy_reader`: not a superuser, no `BYPASSRLS`, not the table owner,
  `SELECT` only. Each search sets `app.roles` for its own transaction; without it the policy
  matches nothing. `doctor` and a test check all of this.
- The retriever also checks every row it gets back and stops if one is outside the user's
  roles, which would mean the app is connected as the wrong role.
- Roles come from `config/users.yaml` after sign-in, never from the request; the API refuses
  unknown fields. Sessions are HMAC-signed, HttpOnly, SameSite=Strict cookies that expire after
  8 hours; passwords are scrypt hashes. The demo passwords are public by design.
- The page loads no external scripts (`Content-Security-Policy: default-src 'self'`) and puts
  model output and document text on the page as text, never as HTML.
- Sign-in has no rate limit; anything beyond this machine would need one in front of it.
- Row-level security protects against the app forgetting the filter, not against someone who
  can run SQL as the reader: they could set `app.roles` themselves. The reader's password
  stays in `.env` on the server, and requests never carry SQL.

## Layout

```
src/policy_assistant/
  contracts.py  text.py  ranking.py     shared types, lemmas and literals, rank fusion
  corpus/                               the law and policy parsers, chunking
  search.py  answering.py  ingest.py    retriever, assistant, ingest
  adapters/                             Postgres, Ollama, reranker, chat models, the Rada API
  evaluation/                           questions, scoring, leak tests, the memo
  web/                                  FastAPI app and the static page
  config.py  auth.py  wiring.py  doctor.py  cli.py  fakes.py
corpus/policies/   the 15 invented policies (ЗРАЗОК)
config/            models and prices, roles and demo users
evals/             the 50 questions, leak cases and injections
tests/             pytest
data/              downloaded law, reranker model, reports (not in git)
```
