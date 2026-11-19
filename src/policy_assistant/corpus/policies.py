"""Parser for corpus/policies/*.md: YAML front matter and numbered "## N. Title" sections."""

import re
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from policy_assistant.contracts import Block, Document, Paragraph


class PolicyFormatError(ValueError):
    pass


class PolicyMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=3)
    roles: list[str] = Field(min_length=1)
    owner: str
    version: str
    effective: date


_HEADING = re.compile(r"^##\s+(\d+)\.\s+(\S.*?)\s*$")
_ROLES = re.compile(r"^<!--\s*roles:\s*(.*?)\s*-->$")


def parse_policy(path: Path, known_roles: set[str]) -> Document:
    text = path.read_text(encoding="utf-8")
    front, body_lines, first_body_line = _split_front_matter(text, path)
    try:
        meta = PolicyMeta.model_validate(front)
    except ValidationError as error:
        raise PolicyFormatError(f"{path}: front matter: {_first_problem(error)}") from None
    if meta.id != path.stem:
        raise PolicyFormatError(f"{path}: id {meta.id!r} differs from the file name {path.stem!r}")
    _check_roles(meta.roles, known_roles, f"{path}: front matter")

    blocks = []
    section: dict | None = None
    paragraph: list[str] = []

    def end_paragraph() -> None:
        if paragraph and section is not None:
            section["paragraphs"].append(Paragraph("\n".join(paragraph)))
        paragraph.clear()

    def end_section() -> None:
        end_paragraph()
        if section is None:
            return
        if not section["paragraphs"]:
            raise PolicyFormatError(
                f"{path}:{section['line']}: section {section['number']} is empty"
            )
        blocks.append(
            Block(
                section=section["number"],
                title=f"{section['number']}. {section['title']}",
                paragraphs=tuple(section["paragraphs"]),
                roles=tuple(section["roles"]),
            )
        )

    for offset, raw_line in enumerate(body_lines):
        line_no = first_body_line + offset
        line = raw_line.rstrip()
        if line.startswith("## "):
            end_section()
            heading = _HEADING.match(line)
            if heading is None:
                raise PolicyFormatError(f"{path}:{line_no}: headings must look like '## 1. Назва'")
            number = heading.group(1)
            if any(block.section == number for block in blocks):
                raise PolicyFormatError(f"{path}:{line_no}: section {number} appears twice")
            section = {
                "number": number,
                "title": heading.group(2),
                "roles": list(meta.roles),
                "paragraphs": [],
                "line": line_no,
                "fresh": True,
            }
            continue
        if line.lstrip().startswith("<!--"):
            directive = _ROLES.match(line.strip())
            if directive is None or section is None or not section["fresh"]:
                raise PolicyFormatError(
                    f"{path}:{line_no}: a comment may only set roles, on the line under a heading"
                )
            roles = [role.strip() for role in directive.group(1).split(",") if role.strip()]
            _check_roles(roles, known_roles, f"{path}:{line_no}")
            section["roles"] = roles
            section["fresh"] = False
            continue
        if section is None:
            continue  # the ЗРАЗОК banner and anything else above the first section
        section["fresh"] = False
        if line.strip():
            paragraph.append(line.strip())
        else:
            end_paragraph()
    end_section()

    if not blocks:
        raise PolicyFormatError(f"{path}: no '## 1. Назва' sections")
    return Document(
        id=meta.id,
        title=meta.title,
        kind="policy",
        source=str(path),
        version=f"{meta.version} від {meta.effective:%d.%m.%Y}",
        blocks=tuple(blocks),
    )


def _split_front_matter(text: str, path: Path) -> tuple[dict, list[str], int]:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise PolicyFormatError(f"{path}: must start with '---' front matter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise PolicyFormatError(f"{path}: the front matter is not closed with '---'") from None
    try:
        front = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as error:
        raise PolicyFormatError(f"{path}: front matter is not valid YAML: {error}") from None
    if not isinstance(front, dict):
        raise PolicyFormatError(f"{path}: front matter must be a mapping")
    return front, lines[end + 1 :], end + 2


def _check_roles(roles: list[str], known_roles: set[str], where: str) -> None:
    unknown = sorted(set(roles) - known_roles)
    if unknown:
        raise PolicyFormatError(
            f"{where}: unknown roles {', '.join(unknown)}; known: {', '.join(sorted(known_roles))}"
        )
    if not roles:
        raise PolicyFormatError(f"{where}: roles may not be empty")


def _first_problem(error: ValidationError) -> str:
    problem = error.errors()[0]
    where = ".".join(str(part) for part in problem["loc"]) or "front matter"
    return f"{where}: {problem['msg']}"
