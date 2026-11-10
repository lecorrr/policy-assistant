"""The policy-assistant command.

    setup            write .env with generated passwords and a session secret
    doctor           check the database, row-level security, Ollama, models, keys and corpus
    init-db          create or update the schema, the reader role and its grants
    fetch-law        download the current text of law 1734-VIII from the Rada's open-data API
    ingest           parse, chunk, embed and store the corpus
    check-evals      check the evaluation set against the corpus (no database needed)
    ask              ask one question as a user
    serve            run the web page and the HTTP API
    eval-retrieval   recall@5 per search mode and the retrieval leak test
    eval-answers     answer quality, latency, cost and leaks per model
    memo             write the model-choice memo from the latest reports
    hash-password    print a password hash for config/users.yaml

--fake swaps the embedder, the reranker and the answer model for deterministic stand-ins, so the
pipeline runs without Ollama, downloads or keys. The database is still the real one.
"""

import argparse
import getpass
import logging
import secrets
import sys
from pathlib import Path

from policy_assistant import __version__
from policy_assistant.adapters import db
from policy_assistant.adapters.chat import ModelError
from policy_assistant.adapters.embeddings import EmbeddingError
from policy_assistant.adapters.rada import TEXT_URL, LawDownloadError, fetch_law_text, save_law_text
from policy_assistant.adapters.reranker import RerankerError
from policy_assistant.answering import BadQuestion, UnknownModel
from policy_assistant.config import (
    ENV_FILE,
    PROJECT_ROOT,
    ConfigError,
    load_models,
    load_settings,
    load_users,
    read_env_file,
)
from policy_assistant.corpus import (
    CorpusError,
    LawFormatError,
    PolicyFormatError,
    load_corpus,
    load_law,
)
from policy_assistant.corpus.chunking import chunk_documents
from policy_assistant.corpus.law import LAW_NREG
from policy_assistant.doctor import run_doctor
from policy_assistant.evaluation import QUESTIONS_FILE, latest_report, write_report
from policy_assistant.evaluation.answers import EvaluationAborted, evaluate_answers
from policy_assistant.evaluation.cases import EvalSetError, check_eval_set, load_eval_set
from policy_assistant.evaluation.judge import JudgeError
from policy_assistant.evaluation.memo import render_memo
from policy_assistant.evaluation.retrieval import evaluate_retrieval
from policy_assistant.search import MODES, AccessViolation
from policy_assistant.text import Lemmatizer
from policy_assistant.wiring import (
    FAKE_MODEL,
    Context,
    build_judge,
    build_services,
    check_models_runnable,
    make_embedder,
)

MEMO_FILE = PROJECT_ROOT / "docs" / "model-choice-memo.md"

# Failures the commands explain in one line instead of a traceback.
EXPECTED_ERRORS = (
    ConfigError,
    CorpusError,
    LawFormatError,
    PolicyFormatError,
    db.DatabaseError,
    EmbeddingError,
    RerankerError,
    ModelError,
    LawDownloadError,
    EvalSetError,
    EvaluationAborted,
    JudgeError,
    AccessViolation,
    BadQuestion,
    UnknownModel,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "pymorphy3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        if args.command == "setup":
            return cmd_setup(args)
        if args.command == "hash-password":
            return cmd_hash_password(args)
        settings = load_settings()
        context = Context(
            settings, load_models(settings.models_file), load_users(settings.users_file)
        )
        return args.handler(args, context)
    except EXPECTED_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


# --- commands ---


def cmd_setup(args) -> int:
    existing = read_env_file(ENV_FILE)
    wanted = {
        "POLICY_DB_ADMIN_PASSWORD": secrets.token_urlsafe(24),
        "POLICY_DB_READER_PASSWORD": secrets.token_urlsafe(24),
        "POLICY_SESSION_SECRET": secrets.token_urlsafe(48),
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
    }
    missing = {key: value for key, value in wanted.items() if key not in existing}
    if not missing:
        print(f"{ENV_FILE} already has every setting; nothing changed.")
        return 0
    lines = (
        [] if ENV_FILE.exists() else ["# Written by `policy-assistant setup`. Keep it out of git."]
    )
    lines += [f"{key}={value}" for key, value in missing.items()]
    with ENV_FILE.open("a", encoding="utf-8") as handle:
        if ENV_FILE.stat().st_size and not ENV_FILE.read_text(encoding="utf-8").endswith("\n"):
            handle.write("\n")
        handle.write("\n".join(lines) + "\n")
    ENV_FILE.chmod(0o600)
    print(f"Added to {ENV_FILE}: {', '.join(missing)}")
    print("Put API keys after OPENAI_API_KEY= and ANTHROPIC_API_KEY= to use the cloud models.")
    return 0


def cmd_hash_password(args) -> int:
    from policy_assistant.auth import hash_password

    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Again: "):
        print("error: the passwords differ", file=sys.stderr)
        return 1
    print(hash_password(password))
    return 0


def cmd_init_db(args, context: Context) -> int:
    settings = context.settings
    db.create_schema(
        settings.admin_dsn,
        settings.reader_dsn,
        dims=settings.embedding_dims,
        embedding_model=settings.embedding_model,
        reset=args.reset,
    )
    print(
        f"Schema ready in {settings.db_name} at {settings.db_host}:{settings.db_port}; "
        f"reader role {settings.db_reader_user}; {settings.embedding_dims}-dimension vectors."
    )
    if args.reset:
        print("Tables were dropped and recreated: run `policy-assistant ingest` next.")
    return 0


def cmd_fetch_law(args, context: Context) -> int:
    path = args.out or context.settings.law_file
    url = TEXT_URL.format(nreg=LAW_NREG)
    text = fetch_law_text(LAW_NREG, timeout_s=context.settings.request_timeout_s)
    note = save_law_text(text, path, url=url)
    document = load_law(path)
    articles = sum(1 for block in document.blocks if block.title.startswith("Стаття"))
    print(f"Saved {note['bytes']} bytes to {path} (sha256 {note['sha256'][:12]}…)")
    print(f"Parsed: {articles} articles, {len(document.blocks)} blocks in all.")
    return 0


def cmd_ingest(args, context: Context) -> int:
    from policy_assistant.ingest import ingest

    settings = context.settings
    law_file = args.law_file or settings.law_file
    documents = load_corpus(settings.corpus_dir, law_file, set(context.users.roles))
    db.check_embedding_setup(settings.admin_dsn, dims=settings.embedding_dims)
    lemmatizer = Lemmatizer()
    embedder = make_embedder(settings, lemmatizer, args.fake)
    report = ingest(
        documents,
        store=db.AdminStore(settings.admin_dsn),
        embedder=embedder,
        embedding_model=embedder.model,
        lemmatizer=lemmatizer,
        max_chars=settings.chunk_chars,
    )
    by_document: dict[str, int] = {}
    for chunk in report.chunks:
        by_document[chunk.doc_id] = by_document.get(chunk.doc_id, 0) + 1
    for document in report.documents:
        print(
            f"  {document.id:24} {by_document[document.id]:4} chunks  "
            f"roles {','.join(document.roles):18} {document.version}"
        )
    print(
        f"Stored {len(report.chunks)} chunks from {len(report.documents)} documents; embedded "
        f"{report.embedded} with {embedder.model}, reused {report.reused} stored embeddings."
    )
    return 0


def cmd_check_evals(args, context: Context) -> int:
    eval_set = load_eval_set(args.questions)
    law_file = args.law_file or context.settings.law_file
    documents = load_corpus(context.settings.corpus_dir, law_file, set(context.users.roles))
    chunks = chunk_documents(documents, context.settings.chunk_chars)
    problems = check_eval_set(eval_set, context.users, chunks)
    leak_cases = sum(len(q.leak_as) for q in eval_set.questions) + len(eval_set.injections)
    print(
        f"{len(eval_set.questions)} questions ({len(eval_set.answerable)} answerable), "
        f"{leak_cases} leak cases, checked against {len(chunks)} chunks."
    )
    for problem in problems:
        print(f"PROBLEM {problem}")
    print("OK" if not problems else f"{len(problems)} problems")
    return 0 if not problems else 1


def cmd_ask(args, context: Context) -> int:
    user = context.users.user(args.user)
    if user is None:
        raise ConfigError(f"no user {args.user!r} in {context.settings.users_file}")
    services = build_services(context, fake=args.fake, models=[args.model] if args.model else None)
    try:
        answer = services.assistant.ask(user, args.question, model=args.model, mode=args.mode)
    finally:
        services.close()
    seconds = answer.timings_ms.get("total_ms", 0) / 1000
    print(f"{user.name} ({', '.join(user.roles)}) · {answer.model} · {seconds:.1f} s")
    print()
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            print(f"  [{citation.number}] {citation.hit.label}")
    if answer.bad_citations:
        print(f"\nCited excerpts that do not exist: {answer.bad_citations}")
    print("\nExcerpts given to the model: " + "; ".join(hit.label for hit in answer.hits))
    print("Timings: " + ", ".join(f"{key} {value:.0f}" for key, value in answer.timings_ms.items()))
    return 0


def cmd_serve(args, context: Context) -> int:
    import uvicorn

    from policy_assistant.auth import SessionSigner
    from policy_assistant.web.app import Services, create_app

    settings = context.settings
    if settings.session_secret is None:
        raise ConfigError("POLICY_SESSION_SECRET is not set: run `policy-assistant setup`")
    services = build_services(context, fake=args.fake)
    app = create_app(
        Services(
            assistant=services.assistant,
            users=context.users,
            signer=SessionSigner(
                settings.session_secret.get_secret_value(), settings.session_hours * 3600
            ),
            store=services.store,
            cookie_secure=settings.cookie_secure,
        )
    )
    print(
        f"policy-assistant {__version__} on http://{args.host}:{args.port} "
        f"(models: {', '.join(sorted(services.assistant.generators))})"
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        services.close()
    return 0


def cmd_eval_retrieval(args, context: Context) -> int:
    eval_set = load_eval_set(args.questions)
    if not args.fake and context.settings.reranker != "none":
        print(f"Loading {context.settings.reranker}; the first run downloads about 0.6 GB.")
    services = build_services(context, fake=args.fake, need_generators=False)
    # A rank per mode for each question: '-' means its evidence was not in the top 5.
    print(f"\nSearching {len(eval_set.answerable)} questions: {', '.join(args.modes)}", flush=True)
    try:
        report = evaluate_retrieval(
            eval_set,
            services.retriever,
            context.users,
            tuple(args.modes),
            progress=lambda line: print(line, flush=True),
        )
    finally:
        services.close()
    report.update(
        kind="retrieval",
        version=__version__,
        reranker=services.retriever.reranker.name,
        embedding_model=services.embedding_model,
        fake=args.fake,
    )
    path = write_report(report, args.out or context.settings.reports_dir, "retrieval")
    print(f"\n{'mode':15} {'recall@5':>9} {'MRR':>6}  missed")
    for mode, row in report["modes"].items():
        missed = ", ".join(row["missed"]) or "-"
        print(f"{mode:15} {row['recall_at_5']:9.2f} {row['mrr']:6.2f}  {missed}")
    leaks = report["leaks"]
    print(f"Leak test: {leaks['checks']} searches, {len(leaks['failures'])} failures.")
    for failure in leaks["failures"]:
        print(f"  LEAK {failure['id']} as {failure['user']}: {failure['problem']}")
    print(f"Report: {path}")
    return 0 if leaks["passed"] else 1


def cmd_eval_answers(args, context: Context) -> int:
    eval_set = load_eval_set(args.questions)
    if args.only:
        wanted = set(args.only)
        eval_set.questions = [q for q in eval_set.questions if q.id in wanted]
        eval_set.injections = [i for i in eval_set.injections if i.id in wanted]
        if not eval_set.questions:
            raise EvalSetError(f"none of {', '.join(sorted(wanted))} is a question id")
    names = [FAKE_MODEL] if args.fake else list(args.models or context.models.memo)
    if not args.fake:
        check_models_runnable(names, context, explicit=bool(args.models))
    services = build_services(context, fake=args.fake, models=names)
    judge = None if args.fake else build_judge(args.judge, context)
    failed = False
    try:
        for name in names:
            price = context.models.models[name].price if name in context.models.models else _free()
            print(f"--- {name} ---")
            report = evaluate_answers(
                eval_set,
                services.assistant,
                context.users,
                name,
                price=price,
                judge=judge,
                progress=lambda line: None if args.quiet else print(line, flush=True),
            )
            report["retrieval"] = {
                "embedding_model": services.embedding_model,
                "reranker": services.retriever.reranker.name,
                "top_k": services.retriever.top_k,
            }
            path = write_report(report, args.out or context.settings.reports_dir, f"answers-{name}")
            summary = report["summary"]
            _print_summary(name, summary, judge=report["judge"])
            print(f"Report: {path}")
            # An incomplete judgement is a failed measurement, like a leak.
            failed = failed or not summary["leak_test_passed"] or summary["judge_errors"] > 0
    finally:
        services.close()
    return 1 if failed else 0


def cmd_memo(args, context: Context) -> int:
    reports_dir = args.reports or context.settings.reports_dir
    answers = {}
    for name in context.models.memo:
        report = latest_report(reports_dir, f"answers-{name}")
        if report is None:
            print(f"No report for {name} yet: `policy-assistant eval-answers --models {name}`")
        else:
            answers[name] = report
    if not answers:
        print(f"error: no answer reports in {reports_dir}", file=sys.stderr)
        return 1
    memo = render_memo(answers, latest_report(reports_dir, "retrieval"), context.models)
    out = args.out or MEMO_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(memo, encoding="utf-8")
    print(f"Wrote {out} ({len(answers)} of {len(context.models.memo)} models)")
    return 0


def _free():
    from policy_assistant.config import Price

    return Price(input=0, output=0)


def _print_summary(name: str, summary: dict, *, judge: str | None) -> None:
    latency = summary["latency_ms"]
    cost = summary["cost_per_1000_questions_usd"]
    print(
        f"{name}: correct {summary['correct']}/{summary['answerable']}, refused "
        f"{summary['refused_unanswerable']}/{summary['unanswerable']} unanswerable, "
        f"{summary['false_refusals']} wrong refusals; citations valid "
        f"{summary['citations_valid']}, "
        f"numbers grounded {summary['numbers_grounded']}, {_faithfulness_text(summary, judge)}"
    )
    print(
        f"  leaks {summary['leaks']}/{summary['leak_cases']} "
        f"({'PASSED' if summary['leak_test_passed'] else 'FAILED'}); p50 {latency['p50']} ms, "
        f"p95 {latency['p95']} ms; cost per 1,000 questions ${cost}; errors {summary['errors']}"
    )


def _faithfulness_text(summary: dict, judge: str | None) -> str:
    if judge is None:
        return "faithfulness not judged"
    text = (
        f"faithfulness {summary['faithfulness']} by {judge} "
        f"({summary['judged']} of {summary['answered']} judged"
    )
    errors = summary["judge_errors"]
    if errors:
        text += f", {errors} judge error{'' if errors == 1 else 's'}"
    return text + ")"


# --- arguments ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-assistant", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"policy-assistant {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("setup", help="write .env with generated passwords")
    commands.add_parser("hash-password", help="print a password hash for config/users.yaml")

    doctor = commands.add_parser("doctor", help="check everything the app needs")
    doctor.set_defaults(handler=lambda args, context: run_doctor(context))

    init = commands.add_parser("init-db", help="create or update the schema and the reader role")
    init.add_argument(
        "--reset", action="store_true", help="drop the tables first (deletes the corpus)"
    )
    init.set_defaults(handler=cmd_init_db)

    fetch = commands.add_parser("fetch-law", help="download the text of law 1734-VIII")
    fetch.add_argument(
        "--out", type=Path, help="where to save it (default data/sources/1734-19.txt)"
    )
    fetch.set_defaults(handler=cmd_fetch_law)

    ingest = commands.add_parser("ingest", help="parse, chunk, embed and store the corpus")
    ingest.add_argument(
        "--law-file", type=_existing_file, help="the law's text, if not the default"
    )
    ingest.add_argument("--fake", action="store_true", help="embed with the hashing stand-in")
    ingest.set_defaults(handler=cmd_ingest)

    check = commands.add_parser("check-evals", help="check the evaluation set against the corpus")
    check.add_argument("--questions", type=_existing_file, default=QUESTIONS_FILE)
    check.add_argument("--law-file", type=_existing_file)
    check.set_defaults(handler=cmd_check_evals)

    ask = commands.add_parser("ask", help="ask one question as a user")
    ask.add_argument("--user", required=True, help="a login from config/users.yaml")
    ask.add_argument("--model", help="a model from config/models.yaml (default: its default)")
    ask.add_argument("--mode", choices=MODES, default="hybrid+rerank")
    ask.add_argument("--fake", action="store_true")
    ask.add_argument("question")
    ask.set_defaults(handler=cmd_ask)

    serve = commands.add_parser("serve", help="run the web page and the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8010)
    serve.add_argument("--fake", action="store_true")
    serve.set_defaults(handler=cmd_serve)

    retrieval = commands.add_parser(
        "eval-retrieval", help="recall@5 per search mode and the leak test"
    )
    retrieval.add_argument("--questions", type=_existing_file, default=QUESTIONS_FILE)
    retrieval.add_argument("--modes", type=_modes, default=list(MODES), help="comma-separated")
    retrieval.add_argument("--out", type=Path, help="report folder (default data/evals)")
    retrieval.add_argument("--fake", action="store_true")
    retrieval.set_defaults(handler=cmd_eval_retrieval)

    answers = commands.add_parser("eval-answers", help="answer quality, latency, cost and leaks")
    answers.add_argument(
        "--models", type=_names, help="comma-separated (default: the memo's models)"
    )
    answers.add_argument(
        "--judge", help="model that judges faithfulness, or none (default: models.yaml)"
    )
    answers.add_argument("--only", type=_names, help="comma-separated question ids")
    answers.add_argument("--questions", type=_existing_file, default=QUESTIONS_FILE)
    answers.add_argument("--out", type=Path, help="report folder (default data/evals)")
    answers.add_argument("--quiet", action="store_true", help="print only the summary")
    answers.add_argument("--fake", action="store_true")
    answers.set_defaults(handler=cmd_eval_answers)

    memo = commands.add_parser("memo", help="write the model-choice memo from the latest reports")
    memo.add_argument("--reports", type=Path, help="report folder (default data/evals)")
    memo.add_argument("--out", type=Path, help=f"default {MEMO_FILE.relative_to(PROJECT_ROOT)}")
    memo.set_defaults(handler=cmd_memo)
    return parser


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{value} is not a file")
    return path


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("a port is 1–65535")
    return port


def _names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("give at least one name")
    return names


def _modes(value: str) -> list[str]:
    modes = _names(value)
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown modes {', '.join(unknown)}; choose from {', '.join(MODES)}"
        )
    return modes


if __name__ == "__main__":
    sys.exit(main())
