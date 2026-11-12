"""Keep the output blocks in the README identical to what the tool prints.

A block is delimited by HTML comments::

    <!-- output: embed-parity demo --fault batch-mask -->
    ```text
    ...
    ```
    <!-- /output -->

To regenerate after an intentional change in output::

    EMBED_PARITY_UPDATE_README=1 pytest tests/test_readme.py
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
from pathlib import Path

import pytest

from embed_parity import __version__
from embed_parity.cli import main

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
BLOCK = re.compile(
    r"(<!-- output: (?P<cmd>embed-parity[^\n]*?) -->\n```text\n)"
    r"(?P<body>.*?)(```\n<!-- /output -->)",
    re.DOTALL,
)


def run(command: str) -> str:
    argv = shlex.split(command)[1:]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main(argv)
    return buffer.getvalue()


def blocks() -> list[tuple[str, str]]:
    text = README.read_text(encoding="utf-8")
    return [(m.group("cmd"), m.group("body")) for m in BLOCK.finditer(text)]


def test_readme_has_demo_blocks() -> None:
    commands = [cmd for cmd, _ in blocks()]
    assert len(commands) >= 3
    assert "embed-parity demo" in commands


def test_readme_output_blocks_are_current() -> None:
    if os.environ.get("EMBED_PARITY_UPDATE_README") == "1":
        text = README.read_text(encoding="utf-8")
        updated = BLOCK.sub(lambda m: m.group(1) + run(m.group("cmd")) + m.group(4), text)
        README.write_text(updated, encoding="utf-8")
    for command, body in blocks():
        assert body == run(command), (
            f"README block for `{command}` is out of date; run "
            "EMBED_PARITY_UPDATE_README=1 pytest tests/test_readme.py"
        )


def test_readme_demo_contains_a_diagnosis_line() -> None:
    body = dict(blocks())["embed-parity demo"]
    assert "endpoint matches CLS pooling without normalisation (cosine min 1.00000)" in body
    assert "the model declares mean pooling with normalisation" in body
    assert f"embed-parity {__version__}" in body


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md", "CONTRIBUTING.md", "docs"])
def test_expected_documents_exist(name: str) -> None:
    assert (ROOT / name).exists()


def test_readme_links_point_to_existing_files() -> None:
    text = README.read_text(encoding="utf-8")
    for target in re.findall(r"\]\((?!https?://|#|mailto:)([^)#]+)(?:#[^)]*)?\)", text):
        assert (ROOT / target).exists(), f"README links to missing file {target}"
