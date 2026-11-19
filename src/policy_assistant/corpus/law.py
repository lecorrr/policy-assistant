"""Parser for the official text of law 1734-VIII as the Rada's open-data API serves it."""

import re
from dataclasses import dataclass, field

from policy_assistant.contracts import STAFF_ROLE, Block, Document, Paragraph
from policy_assistant.text import normalize

LAW_ID = "law-1734"
LAW_NUMBER = "1734-VIII"
LAW_TITLE = "Про споживче кредитування"
LAW_NREG = "1734-19"  # the law's id in the Rada's database
LAW_URL = f"https://zakon.rada.gov.ua/laws/show/{LAW_NREG}"

# The website writes superscripts as hyphens: "Розділ III-1".
_SECTION = re.compile(r"^Розділ\s+([IVXLCІХ]+)(?:-?(\d+))?\.?(?:\s+(.*))?$")
_ARTICLE = re.compile(r"^Стаття\s+(\d+(?:-\d+)?)\.?(?:\s+(.*))?$")
_NUMBERED = re.compile(r"^(\d+(?:-\d+)?)\.\s+\S")
_ANNEX = re.compile(r"^Додаток\s+(\d+)\s*$")
_SIGNATURE = re.compile(r"^Президент\s+України\s*$")
_LAW_HEADER = re.compile(r"^З\s*А\s*К\s*О\s*Н\s+У\s*К\s*Р\s*А\s*Ї\s*Н\s*И$")
_ROMAN_LOOKALIKES = str.maketrans({"І": "I", "Х": "X"})  # Cyrillic letters used as numerals


class LawFormatError(ValueError):
    pass


@dataclass
class _Draft:
    section: str
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    last_part: int | None = None  # the last numbered part seen in this article
    part: str | None = None
    titled: bool = False  # an annex has taken its title from the line under "Додаток N"


def read_law_bytes(raw: bytes) -> str:
    """UTF-8, or Windows-1251 in older exports."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1251")


def parse_law(text: str, *, source: str, version: str) -> Document:
    lines = _without_notes(_clean_lines(text))
    blocks: list[Block] = []
    draft: _Draft | None = None
    mode = "head"  # head -> body -> final -> signature -> annex
    section_label = ""
    section_title = ""
    last_article = 0
    last_point: int | None = None
    preamble: list[Paragraph] = []

    def close() -> None:
        nonlocal draft
        if draft is not None and draft.paragraphs:
            blocks.append(Block(draft.section, draft.title, tuple(draft.paragraphs), (STAFF_ROLE,)))
        draft = None

    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1

        if mode in ("head", "body"):
            section = _section(line)
            if section is not None:
                close()
                section_label, section_title = section
                if not section_title and index < len(lines) and _is_upperish(lines[index]):
                    section_title = lines[index]
                    index += 1
                mode = "final" if "ПРИКІНЦЕВ" in section_title.upper() else "body"
                continue

        if mode == "head":
            if not _is_header_line(line):
                preamble.append(Paragraph(line))
        elif mode == "body":
            article = _ARTICLE.match(line)
            label = _next_label(article.group(1), last_article) if article else None
            if label is not None:
                close()
                last_article = int(label.split("-")[0])
                title = f"Стаття {label}. {article.group(2) or ''}".strip()
                draft = _Draft(section=label, title=title)
                continue
            if draft is None:
                raise LawFormatError(f"text before the first article of section {section_label}")
            _add_article_paragraph(draft, line)
        elif mode == "final":
            if _SIGNATURE.match(line):
                close()
                mode = "signature"
                continue
            numbered = _NUMBERED.match(line)
            label = _next_label(numbered.group(1), last_point) if numbered else None
            if label is not None:
                close()
                last_point = int(label.split("-")[0])
                draft = _Draft(
                    section=f"{section_label}.{label}",
                    title=f"Розділ {section_label}. {_title_case(section_title)}, пункт {label}",
                )
            if draft is None:
                raise LawFormatError("text before point 1 of the final provisions")
            draft.paragraphs.append(Paragraph(line))
        else:  # signature or annex
            annex = _ANNEX.match(line)
            if annex:
                close()
                mode = "annex"
                draft = _Draft(
                    section=f"Додаток {annex.group(1)}", title=f"Додаток {annex.group(1)}"
                )
                continue
            if mode == "annex" and draft is not None:
                if not draft.paragraphs and not draft.titled:
                    if not _is_annex_reference(line):
                        draft.title = f"{draft.title}. {line}"
                        draft.titled = True
                    continue
                draft.paragraphs.append(Paragraph(line))
    close()

    if preamble:
        blocks.insert(0, Block("преамбула", "Преамбула", tuple(preamble), (STAFF_ROLE,)))
    _check(blocks)
    return Document(
        id=LAW_ID,
        title=f"Закон України «{LAW_TITLE}» № {LAW_NUMBER}",
        kind="law",
        source=source,
        version=version,
        blocks=tuple(blocks),
        url=LAW_URL,
    )


def law_label(section: str, parts: list[str]) -> str:
    """'Закон № 1734-VIII, ст. 15, ч. 1–3'"""
    base = f"Закон № {LAW_NUMBER}"
    if section == "преамбула":
        return f"{base}, преамбула"
    if section.startswith("Додаток"):
        return f"{base}, {section.lower()}"
    if "." in section:
        division, point = section.split(".", 1)
        return f"{base}, розд. {division}, п. {point}"
    label = f"{base}, ст. {section}"
    if parts:
        label += f", ч. {parts[0]}" if parts[0] == parts[-1] else f", ч. {parts[0]}–{parts[-1]}"
    return label


def _clean_lines(text: str) -> list[str]:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return [line.strip() for line in text.split("\n") if line.strip()]


def _without_notes(lines: list[str]) -> list[str]:
    """Remove {...} notes, including ones that span several lines."""
    kept_lines = []
    depth = 0
    for line in lines:
        kept = []
        for char in line:
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    raise LawFormatError(f"a closing brace without an opening one: {line[:80]}")
                depth -= 1
            elif depth == 0:
                kept.append(char)
        text = re.sub(r"\s+", " ", "".join(kept)).strip()
        if text:
            kept_lines.append(text)
    if depth:
        raise LawFormatError("an editorial note in braces is never closed")
    return kept_lines


def _section(line: str) -> tuple[str, str] | None:
    """('III-1', 'ВРЕГУЛЮВАННЯ ...') for a section heading, else None."""
    match = _SECTION.match(line)
    if match is None:
        return None
    title = (match.group(3) or "").strip()
    if title and not _is_upperish(title):
        return None  # running text such as "Розділ II доповнити статтею ..."
    numeral = match.group(1).translate(_ROMAN_LOOKALIKES)
    insert = match.group(2)
    return (f"{numeral}-{insert}" if insert else numeral), title


def _next_label(raw: str, last: int | None) -> str | None:
    """Label that continues the numbering after `last`, else None ("62" after 6 is 6², "6-2")."""
    if "-" in raw:
        base = int(raw.partition("-")[0])
        return raw if last is not None and base == last else None
    number = int(raw)
    if last is None:
        return raw if number == 1 else None
    if number == last + 1:
        return raw
    prefix = str(last)
    rest = raw[len(prefix) :]
    if raw.startswith(prefix) and rest and not rest.startswith("0") and len(rest) <= 2:
        return f"{prefix}-{rest}"
    return None


def _add_article_paragraph(draft: _Draft, line: str) -> None:
    numbered = _NUMBERED.match(line)
    if numbered:
        label = _next_label(numbered.group(1), draft.last_part)
        if label is not None:
            draft.last_part = int(label.split("-")[0])
            draft.part = label
    draft.paragraphs.append(Paragraph(line, draft.part))


def _is_upperish(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and sum(char.isupper() for char in letters) / len(letters) > 0.8


def _is_header_line(line: str) -> bool:
    return (
        bool(_LAW_HEADER.match(line))
        or normalize(line) == normalize(LAW_TITLE)
        or line.startswith("(Відомості")
    )


def _is_annex_reference(line: str) -> bool:
    """Lines under "Додаток N" that only name the law it belongs to."""
    text = normalize(line).strip('"')
    return text in ("до закону україни", normalize(LAW_TITLE))


def _title_case(title: str) -> str:
    return title[:1].upper() + title[1:].lower() if title.isupper() else title


def _check(blocks: list[Block]) -> None:
    articles = [b.section for b in blocks if b.title.startswith("Стаття")]
    if not articles:
        raise LawFormatError("no articles found: this does not look like the text of the law")
    if articles[0] != "1":
        raise LawFormatError(f"the first article found is {articles[0]}, not 1")
