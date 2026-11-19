import json
from datetime import UTC, datetime
from pathlib import Path

from policy_assistant.config import PROJECT_ROOT

QUESTIONS_FILE = PROJECT_ROOT / "evals" / "questions.yaml"


def write_report(report: dict, directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{prefix}-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def latest_report(directory: Path, prefix: str) -> dict | None:
    # Digits must follow the prefix, so "answers-gpt-6-luna" skips "...-luna-mini".
    paths = sorted(directory.glob(f"{prefix}-[0-9]*-[0-9]*.json"))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))
