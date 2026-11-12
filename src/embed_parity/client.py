"""HTTP clients for the embedding and rerank API shapes in common use.

Only the standard library is used. The request and response shapes were taken
from each project's documentation; see ``docs/api-shapes.md`` for the sources.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from embed_parity.mathx import decode_base64_embedding

EMBED_APIS = ("openai", "tei", "ollama", "infinity")
RERANK_APIS = ("cohere", "tei", "infinity")

_EMBED_PATHS = {
    "openai": "/v1/embeddings",
    "infinity": "/embeddings",
    "tei": "/embed",
    "ollama": "/api/embed",
}
_RERANK_PATHS = {"cohere": "/v1/rerank", "infinity": "/rerank", "tei": "/rerank"}


class EndpointError(Exception):
    """The endpoint could not be reached or answered with an error."""

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def client_error(self) -> bool:
        """True for HTTP 4xx: the server understood and refused the request."""
        return self.status is not None and 400 <= self.status < 500


class ProtocolError(EndpointError):
    """The endpoint answered 2xx but the body is not what the API promises."""


@dataclass
class EmbedResult:
    vectors: npt.NDArray[np.float64]
    prompt_tokens: int | None = None
    raw: Any = None


@dataclass
class RerankResult:
    """Scores aligned with the order of the documents that were sent."""

    scores: npt.NDArray[np.float64]
    returned_order: list[int] = field(default_factory=list)
    prompt_tokens: int | None = None
    raw: Any = None


def join_url(base_url: str, path: str) -> str:
    """Join base URL and route without doubling a ``/v1`` segment."""
    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[3:]
    return base + path


@dataclass
class Transport:
    """Shared HTTP settings."""

    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    api_key: str | None = None
    timeout: float = 60.0

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        url = join_url(self.base_url, path)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.headers)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            raise EndpointError(
                f"HTTP {exc.code} from {url}: {_short(text)}", status=exc.code, body=text
            ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise EndpointError(f"cannot reach {url}: {reason}") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"{url} did not return JSON: {_short(repr(body[:200]))}") from exc


def _short(text: str, limit: int = 300) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _matrix(rows: list[Any], what: str) -> npt.NDArray[np.float64]:
    try:
        arrays = [np.asarray(r, dtype=np.float64) for r in rows]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{what}: embeddings are not numeric arrays") from exc
    if not arrays:
        raise ProtocolError(f"{what}: no embeddings in response")
    if any(a.ndim != 1 for a in arrays) or len({a.shape[0] for a in arrays}) != 1:
        shapes = sorted({a.shape for a in arrays})
        raise ProtocolError(f"{what}: embeddings have inconsistent shapes {shapes}")
    return np.stack(arrays)


def _usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage")
    return usage if isinstance(usage, dict) else {}


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


@dataclass
class EmbeddingClient:
    """Talks to one embedding endpoint."""

    transport: Transport
    model: str | None = None
    api: str = "openai"
    path: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    dimensions: int | None = None
    requests_made: int = 0

    def __post_init__(self) -> None:
        if self.api not in EMBED_APIS:
            raise ValueError(f"unknown embeddings API {self.api!r}; use one of {EMBED_APIS}")

    @property
    def route(self) -> str:
        return self.path or _EMBED_PATHS[self.api]

    @property
    def supports_base64(self) -> bool:
        return self.api in ("openai", "infinity")

    def embed(self, texts: Sequence[str], *, encoding_format: str | None = None) -> EmbedResult:
        """Embed ``texts`` in a single request."""
        items = list(texts)
        payload: dict[str, Any]
        if self.api == "tei":
            payload = {"inputs": items}
        else:
            payload = {"input": items}
            if self.model:
                payload["model"] = self.model
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        if encoding_format and self.supports_base64:
            payload["encoding_format"] = encoding_format
        payload.update(self.extra_body)
        self.requests_made += 1
        body = self.transport.post(self.route, payload)
        result = self._parse(body, encoding_format or "float")
        if result.vectors.shape[0] != len(items):
            raise ProtocolError(
                f"sent {len(items)} inputs, received {result.vectors.shape[0]} embeddings"
            )
        return result

    def _parse(self, body: Any, encoding_format: str) -> EmbedResult:
        what = f"{self.api} response"
        if self.api == "tei":
            if not isinstance(body, list):
                raise ProtocolError(f"{what}: expected a JSON array of vectors")
            return EmbedResult(_matrix(body, what), None, body)
        if not isinstance(body, dict):
            raise ProtocolError(f"{what}: expected a JSON object")
        if self.api == "ollama":
            rows = body.get("embeddings")
            if not isinstance(rows, list):
                raise ProtocolError(f"{what}: missing 'embeddings'")
            return EmbedResult(
                _matrix(rows, what), _int_or_none(body.get("prompt_eval_count")), body
            )
        data = body.get("data")
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise ProtocolError(f"{what}: missing 'data' list")
        if all("index" in d for d in data):
            data = sorted(data, key=lambda d: int(d["index"]))
        rows = []
        for item in data:
            emb = item.get("embedding")
            if isinstance(emb, str):
                try:
                    rows.append(decode_base64_embedding(emb))
                except ValueError as exc:
                    raise ProtocolError(f"{what}: bad base64 embedding ({exc})") from exc
            elif encoding_format == "base64":
                raise ProtocolError(
                    f"{what}: asked for base64 but received {type(emb).__name__} embeddings"
                )
            else:
                rows.append(emb)
        usage = _usage(body)
        return EmbedResult(_matrix(rows, what), _int_or_none(usage.get("prompt_tokens")), body)


@dataclass
class RerankClient:
    """Talks to one rerank endpoint."""

    transport: Transport
    model: str | None = None
    api: str = "cohere"
    path: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    requests_made: int = 0

    def __post_init__(self) -> None:
        if self.api not in RERANK_APIS:
            raise ValueError(f"unknown rerank API {self.api!r}; use one of {RERANK_APIS}")

    @property
    def route(self) -> str:
        return self.path or _RERANK_PATHS[self.api]

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        """Score every document against ``query`` in a single request."""
        docs = list(documents)
        payload: dict[str, Any]
        if self.api == "tei":
            payload = {"query": query, "texts": docs}
        else:
            payload = {"query": query, "documents": docs}
            if self.model:
                payload["model"] = self.model
        payload.update(self.extra_body)
        self.requests_made += 1
        body = self.transport.post(self.route, payload)
        what = f"{self.api} rerank response"
        usage: dict[str, Any] = {}
        if isinstance(body, list):
            results, score_key = body, "score"
        elif isinstance(body, dict) and isinstance(body.get("results"), list):
            results, score_key = body["results"], "relevance_score"
            usage = _usage(body)
        else:
            raise ProtocolError(f"{what}: expected a list or an object with 'results'")
        scores = np.full(len(docs), np.nan, dtype=np.float64)
        order: list[int] = []
        for item in results:
            if not isinstance(item, dict) or "index" not in item:
                raise ProtocolError(f"{what}: result without 'index'")
            value = item.get(score_key, item.get("score", item.get("relevance_score")))
            idx = int(item["index"])
            if value is None or not 0 <= idx < len(docs):
                raise ProtocolError(f"{what}: bad result entry {item!r}")
            scores[idx] = float(value)
            order.append(idx)
        if np.any(np.isnan(scores)):
            missing = int(np.sum(np.isnan(scores)))
            raise ProtocolError(
                f"{what}: {missing} of {len(docs)} documents have no score "
                "(is top_n limiting the results?)"
            )
        tokens = _int_or_none(usage.get("prompt_tokens", usage.get("total_tokens")))
        return RerankResult(scores=scores, returned_order=order, prompt_tokens=tokens, raw=body)
