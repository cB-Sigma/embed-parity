"""The built-in probe set.

The probes are plain text written for this project. They are chosen to
separate implementation choices from one another rather than to measure
retrieval quality:

* short and long inputs make CLS, mean, last-token and max pooling disagree;
* a small retrieval task (queries and documents) shows whether a difference
  actually changes rankings;
* non-Latin scripts, emoji, code and numbers exercise the tokenizer;
* one input is far longer than any common context window, so truncation
  length and side become visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Probe:
    """One input text."""

    id: str
    text: str
    tags: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "tags": list(self.tags)}

    @staticmethod
    def from_json(obj: dict[str, Any]) -> Probe:
        return Probe(id=str(obj["id"]), text=str(obj["text"]), tags=tuple(obj.get("tags", ())))


@dataclass(frozen=True)
class ProbeSet:
    """Probes plus the ids that form the retrieval and rerank tasks."""

    probes: tuple[Probe, ...]
    queries: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()
    name: str = "builtin-v1"
    _index: dict[str, int] = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        ids = [p.id for p in self.probes]
        if len(set(ids)) != len(ids):
            raise ValueError("probe ids must be unique")
        missing = [i for i in (*self.queries, *self.documents) if i not in set(ids)]
        if missing:
            raise ValueError(f"retrieval task refers to unknown probe ids: {missing}")
        self._index.update({pid: n for n, pid in enumerate(ids)})

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.probes]

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.probes]

    def get(self, probe_id: str) -> Probe:
        return self.probes[self._index[probe_id]]

    def index(self, probe_id: str) -> int:
        return self._index[probe_id]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "probes": [p.to_json() for p in self.probes],
            "queries": list(self.queries),
            "documents": list(self.documents),
        }

    @staticmethod
    def from_json(obj: dict[str, Any]) -> ProbeSet:
        return ProbeSet(
            probes=tuple(Probe.from_json(p) for p in obj["probes"]),
            queries=tuple(obj.get("queries", ())),
            documents=tuple(obj.get("documents", ())),
            name=str(obj.get("name", "custom")),
        )


_LONG_UNIT = (
    "The harbour ledger records every vessel that enters before dawn, the cargo it "
    "declares, the berth it is given and the tide at the moment of mooring. Clerks copy "
    "each line twice, once for the customs house and once for the archive, and any entry "
    "that disagrees between the two books is investigated the same week. "
)
_LONG_TAIL = (
    "The final page lists the lighthouse keeper's notes on fog, wind direction and the "
    "colour of the evening sky above the northern breakwater."
)


def _long_text() -> str:
    # About 900 words: longer than a 512-token window for every common tokenizer,
    # with a distinctive ending so left and right truncation differ clearly.
    return _LONG_UNIT * 18 + _LONG_TAIL


_BUILTIN: tuple[Probe, ...] = (
    # --- retrieval task: queries ---------------------------------------------
    Probe("q-tides", "what causes ocean tides", ("query", "short")),
    Probe("q-sourdough", "how do I keep a sourdough starter alive", ("query", "short")),
    Probe("q-python-sort", "sort a list of dictionaries by a key in python", ("query", "short")),
    Probe("q-photosynthesis", "why are plants green", ("query", "short")),
    # --- retrieval task: documents -------------------------------------------
    Probe(
        "d-tides",
        "Tides are the regular rise and fall of sea level caused mainly by the "
        "gravitational pull of the Moon and, to a lesser extent, the Sun, combined with "
        "the rotation of the Earth.",
        ("document",),
    ),
    Probe(
        "d-waves",
        "Wind blowing across the surface of the sea transfers energy to the water and "
        "produces waves, whose height depends on wind speed, duration and fetch.",
        ("document",),
    ),
    Probe(
        "d-sourdough",
        "A sourdough starter stays healthy when it is fed fresh flour and water on a "
        "regular schedule; discard part of it first, and keep it in the refrigerator if "
        "you bake less than once a week.",
        ("document",),
    ),
    Probe(
        "d-yeast",
        "Commercial baker's yeast is a single cultivated strain that ferments quickly and "
        "predictably, which is why packaged bread recipes give exact proving times.",
        ("document",),
    ),
    Probe(
        "d-python-sort",
        "In Python, use sorted(items, key=lambda d: d['name']) or operator.itemgetter('name') "
        "to sort a list of dictionaries by one field; pass reverse=True for descending order.",
        ("document", "code"),
    ),
    Probe(
        "d-python-dict",
        "Since Python 3.7 dictionaries preserve insertion order, so iterating over a dict "
        "yields keys in the order they were added.",
        ("document",),
    ),
    Probe(
        "d-chlorophyll",
        "Chlorophyll absorbs red and blue light for photosynthesis and reflects green "
        "wavelengths, which is why most leaves look green to the human eye.",
        ("document",),
    ),
    Probe(
        "d-autumn",
        "In autumn many trees stop producing chlorophyll, and the yellow and orange "
        "carotenoids that were present all along become visible.",
        ("document",),
    ),
    # --- tokenizer and length coverage ---------------------------------------
    Probe("s-word", "lighthouse", ("short",)),
    Probe("s-two", "Hello world", ("short",)),
    Probe("s-question", "Is this the right platform for the night train to Vienna?", ("short",)),
    Probe(
        "s-numbers",
        "Invoice 2024-00317: 12 units at 49.90 EUR, VAT 19%, total 712.57 EUR.",
        ("numbers",),
    ),
    Probe("s-code", "def add(a, b):\n    return a + b  # simple sum", ("code",)),
    Probe("u-german", "Die Straßenbahn fährt über die Brücke zum Bahnhof.", ("unicode",)),
    Probe("u-chinese", "今天天气很好，我们去公园散步吧。", ("unicode", "cjk")),  # noqa: RUF001
    Probe("u-japanese", "駅までの道を教えてください。", ("unicode", "cjk")),
    Probe("u-arabic", "المكتبة تفتح أبوابها في الساعة التاسعة صباحا.", ("unicode", "rtl")),
    Probe("u-emoji", "Launch day 🚀 — coffee first ☕, then deploy!", ("unicode", "emoji")),
    Probe(
        "m-paragraph",
        "The committee met on Tuesday to review the proposal for a new cycle path along "
        "the river. Residents raised concerns about lighting and winter maintenance, "
        "while local businesses welcomed the expected increase in visitors. A revised "
        "plan with a narrower carriageway and two additional crossings will be published "
        "next month, followed by a four-week consultation. Construction could begin in "
        "spring if funding from the regional transport budget is confirmed.",
        ("medium",),
    ),
    Probe("l-ledger", _long_text(), ("long",)),
)

_QUERIES = ("q-tides", "q-sourdough", "q-python-sort", "q-photosynthesis")
_DOCUMENTS = (
    "d-tides",
    "d-waves",
    "d-sourdough",
    "d-yeast",
    "d-python-sort",
    "d-python-dict",
    "d-chlorophyll",
    "d-autumn",
)


def builtin_probes() -> ProbeSet:
    """The default probe set (24 texts, 4 queries x 8 documents)."""
    return ProbeSet(probes=_BUILTIN, queries=_QUERIES, documents=_DOCUMENTS, name="builtin-v1")


def load_probes(path: str | Path) -> ProbeSet:
    """Load probes from JSON Lines.

    Each line is ``{"id": ..., "text": ..., "tags": [...]}``. Tag a probe
    ``query`` or ``document`` to include it in the retrieval task and ``long``
    if it is meant to exceed the model's context window.
    """
    probes: list[Probe] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not valid JSON ({exc.msg})") from exc
        if "text" not in obj:
            raise ValueError(f"{path}:{lineno}: missing 'text'")
        obj.setdefault("id", f"p{lineno}")
        probes.append(Probe.from_json(obj))
    if not probes:
        raise ValueError(f"{path}: no probes found")
    return ProbeSet(
        probes=tuple(probes),
        queries=tuple(p.id for p in probes if "query" in p.tags),
        documents=tuple(p.id for p in probes if "document" in p.tags),
        name=Path(path).name,
    )
