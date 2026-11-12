"""Result model and rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from embed_parity import __version__

PASS, FAIL, WARN, SKIP, INFO = "PASS", "FAIL", "WARN", "SKIP", "INFO"
STATUSES = (PASS, FAIL, WARN, SKIP, INFO)

EXIT_OK, EXIT_FAILED, EXIT_ERROR = 0, 1, 2


@dataclass
class Check:
    """One verdict line."""

    id: str
    title: str
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    note: str = ""


@dataclass
class Section:
    """A group of checks, optionally with a table and free-text lines."""

    id: str
    title: str
    checks: list[Check] = field(default_factory=list)
    table: Table | None = None
    lines: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def add(self, check_id: str, title: str, status: str, detail: str = "", **data: Any) -> Check:
        check = Check(f"{self.id}/{check_id}", title, status, detail, data)
        self.checks.append(check)
        return check

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "checks": [c.to_json() for c in self.checks],
            "lines": self.lines,
            "data": self.data,
        }
        if self.table is not None:
            out["table"] = {"headers": self.table.headers, "rows": self.table.rows}
        return out


@dataclass
class Report:
    """Everything one ``check`` run found."""

    endpoint: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    requests: int = 0

    def checks(self) -> list[Check]:
        return [c for s in self.sections for c in s.checks]

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(STATUSES, 0)
        for check in self.checks():
            out[check.status] += 1
        return out

    def verdict(self, strict: bool = False) -> str:
        counts = self.counts()
        if counts[FAIL] or (strict and counts[WARN]):
            return FAIL
        return PASS

    def exit_code(self, strict: bool = False) -> int:
        return EXIT_FAILED if self.verdict(strict) == FAIL else EXIT_OK

    def section(self, section_id: str) -> Section | None:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None

    def to_json(self, strict: bool = False) -> dict[str, Any]:
        return {
            "tool": "embed-parity",
            "tool_version": __version__,
            "verdict": self.verdict(strict),
            "counts": self.counts(),
            "endpoint": self.endpoint,
            "reference": self.reference,
            "thresholds": self.thresholds,
            "requests": self.requests,
            "sections": [s.to_json() for s in self.sections],
        }

    def dumps(self, strict: bool = False) -> str:
        return json.dumps(self.to_json(strict), indent=2, ensure_ascii=False)


def _format_table(table: Table, indent: str = "  ") -> list[str]:
    widths = [len(h) for h in table.headers]
    for row in table.rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [indent + "  ".join(h.ljust(widths[i]) for i, h in enumerate(table.headers)).rstrip()]
    for row in table.rows:
        lines.append(indent + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
    if table.note:
        lines.append(indent + table.note)
    return lines


_COLORS = {PASS: "32", FAIL: "31", WARN: "33", SKIP: "90", INFO: "36"}


def render_text(report: Report, *, strict: bool = False, color: bool = False) -> str:
    """Human-readable report."""

    def tag(status: str) -> str:
        label = f"[{status}]"
        return f"\x1b[{_COLORS[status]}m{label}\x1b[0m" if color else label

    out: list[str] = [f"embed-parity {__version__}"]
    ep = report.endpoint
    if ep:
        bits = [str(ep.get("base_url", ""))]
        for key in ("api", "model"):
            if ep.get(key):
                bits.append(f"{key}={ep[key]}")
        out.append("endpoint   " + "  ".join(bits))
    if report.reference.get("summary"):
        out.append("reference  " + str(report.reference["summary"]))
    for section in report.sections:
        out.append("")
        out.append(f"== {section.title} ==")
        if section.table is not None:
            out.extend(_format_table(section.table))
        for line in section.lines:
            out.append("  " + line)
        title_width = max((len(c.title) for c in section.checks), default=0)
        for check in section.checks:
            detail = f"  {check.detail}" if check.detail else ""
            out.append(f"  {tag(check.status)} {check.title.ljust(title_width)}{detail}".rstrip())
    counts = report.counts()
    parts = [f"{counts[FAIL]} failed", f"{counts[WARN]} warnings", f"{counts[PASS]} passed"]
    if counts[SKIP]:
        parts.append(f"{counts[SKIP]} skipped")
    verdict = report.verdict(strict)
    out.append("")
    line = f"RESULT: {verdict}  ({', '.join(parts)}; {report.requests} requests)"
    out.append(f"\x1b[{_COLORS[verdict]};1m{line}\x1b[0m" if color else line)
    return "\n".join(out) + "\n"


def render_markdown(report: Report, *, strict: bool = False) -> str:
    """Markdown summary, suitable for a CI job summary."""
    icon = {PASS: "✅", FAIL: "❌", WARN: "⚠️", SKIP: "⏭️", INFO: "ℹ️"}  # noqa: RUF001
    verdict = report.verdict(strict)
    out = [f"## embed-parity: {icon[verdict]} {verdict}", ""]
    ep = report.endpoint
    if ep:
        out.append(
            f"Endpoint `{ep.get('base_url', '')}` (api `{ep.get('api', '')}`, "
            f"model `{ep.get('model', '')}`)"
        )
    if report.reference.get("summary"):
        out.append(f"Reference: {report.reference['summary']}")
    for section in report.sections:
        out += ["", f"### {section.title}", ""]
        for line in section.lines:
            out.append(f"> {line}")
        if section.lines:
            out.append("")
        if section.checks:
            out += ["| | check | detail |", "|---|---|---|"]
            for check in section.checks:
                detail = check.detail.replace("|", "\\|")
                out.append(f"| {icon[check.status]} {check.status} | {check.title} | {detail} |")
    return "\n".join(out) + "\n"
