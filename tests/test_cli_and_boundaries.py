import os
import subprocess
import sys

import pytest
from conftest import SAMPLE_LAW, law_text_path

from policy_assistant import __version__, cli
from policy_assistant.config import PROJECT_ROOT


def test_every_command_has_handler():
    parser = cli.build_parser()
    commands = parser._subparsers._group_actions[0].choices
    without = {"setup", "hash-password"}  # run before settings are loaded
    for name, sub in commands.items():
        if name not in without:
            assert sub.get_default("handler") is not None, name


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert f"policy-assistant {__version__}" in capsys.readouterr().out


def test_setup_adds_only_what_is_missing(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-kept\n", encoding="utf-8")
    monkeypatch.setattr(cli, "ENV_FILE", env)
    assert cli.main(["setup"]) == 0
    values = dict(
        line.split("=", 1) for line in env.read_text(encoding="utf-8").splitlines() if "=" in line
    )
    assert values["OPENAI_API_KEY"] == "sk-kept"
    assert len(values["POLICY_SESSION_SECRET"]) >= 32 and values["POLICY_DB_READER_PASSWORD"]
    assert oct(env.stat().st_mode & 0o777) == "0o600"
    before = env.read_text(encoding="utf-8")
    assert cli.main(["setup"]) == 0
    assert (
        env.read_text(encoding="utf-8") == before and "nothing changed" in capsys.readouterr().out
    )


def test_expected_errors_are_one_line(monkeypatch, capsys):
    monkeypatch.setenv("POLICY_TOPK", "1")
    assert cli.main(["doctor"]) == 1
    assert capsys.readouterr().err.startswith("error: unknown settings POLICY_TOPK")


def test_asking_as_unknown_user(capsys):
    assert cli.main(["ask", "--user", "ghost", "Що?"]) == 1
    assert "no user 'ghost'" in capsys.readouterr().err


def test_check_evals_reports_problems_against_another_law(tmp_path, capsys):
    law = tmp_path / "law.txt"
    law.write_text(SAMPLE_LAW, encoding="utf-8")
    assert cli.main(["check-evals", "--law-file", str(law)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM L01: evidence" in out and "50 questions (48 answerable), 53 leak cases" in out


@pytest.mark.skipif(law_text_path() is None, reason="the law's text is not downloaded")
def test_check_evals_passes_on_downloaded_law(capsys):
    assert cli.main(["check-evals"]) == 0
    assert capsys.readouterr().out.strip().endswith("OK")


def test_doctor_reports_each_failure_and_carries_on(monkeypatch, capsys):
    monkeypatch.setenv("POLICY_DB_PORT", "9")
    monkeypatch.setenv("POLICY_OLLAMA_URL", "http://127.0.0.1:9")
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "FAIL database: cannot connect as the admin" in out
    assert "FAIL ollama: not reachable" in out
    assert "OK   config:" in out and "evaluation set: 50 questions" in out


def test_whole_flow_with_fake_models(database, tmp_path, monkeypatch, capsys):
    for key, value in database["env"].items():
        monkeypatch.setenv(key, value)
    law = tmp_path / "law.txt"
    law.write_text(SAMPLE_LAW, encoding="utf-8")
    reports = tmp_path / "reports"
    assert cli.main(["init-db", "--reset"]) == 0
    assert cli.main(["ingest", "--fake", "--law-file", str(law)]) == 0
    assert (
        cli.main(
            [
                "ask",
                "--fake",
                "--user",
                "olena",
                "На скільки відсоткових пунктів менеджер може знизити ставку?",
            ]
        )
        == 0
    )
    assert "Знижки та акції, розд. 1" in capsys.readouterr().out
    assert cli.main(["eval-retrieval", "--fake", "--out", str(reports)]) == 0
    assert "Leak test: 265 searches, 0 failures." in capsys.readouterr().out
    assert cli.main(["eval-answers", "--fake", "--only", "S01,H01,I01", "--out", str(reports)]) == 0
    assert "leaks 0/4 (PASSED)" in capsys.readouterr().out
    assert cli.main(["memo", "--reports", str(reports), "--out", str(tmp_path / "memo.md")]) == 1
    # Questions are embedded the way the corpus was, or not at all.
    assert cli.main(["ask", "--user", "olena", "Що?"]) == 1
    assert "embedded with fake-hashing" in capsys.readouterr().err


CORE = [
    "contracts",
    "text",
    "ranking",
    "answering",
    "corpus",
    "corpus.law",
    "corpus.policies",
    "corpus.chunking",
]
IO_LIBRARIES = [
    "psycopg",
    "psycopg_pool",
    "httpx",
    "fastapi",
    "uvicorn",
    "fastembed",
    "langchain_core",
    "langchain_ollama",
    "langchain_openai",
    "langchain_anthropic",
]


def test_core_imports_no_io_library():
    script = f"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {IO_LIBRARIES!r}:
            raise ImportError("the core may not import " + name)
sys.meta_path.insert(0, Block())
for module in {CORE!r}:
    importlib.import_module("policy_assistant." + module)
print("ok")
"""
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
