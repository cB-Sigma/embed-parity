"""A simulated embedding and rerank server with injectable faults.

It serves the toy models from :mod:`embed_parity.toy` over the same HTTP shapes
real servers use, and can be told to misbehave in the specific ways that the
checker is meant to catch. It backs ``embed-parity demo``, the test-suite and
``embed-parity serve-sim`` (handy for trying the tool without a GPU).

It is a test double. Do not expose it to a network.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import numpy.typing as npt

from embed_parity.backends import POOLINGS, InputConfig, pool
from embed_parity.toy import TOY_MODELS, ToyEncoder, ToyReranker


@dataclass
class Faults:
    """Ways in which the simulated server deviates from the reference library."""

    pooling: str | None = None  #: pool with this mode instead of the declared one
    no_normalize: bool = False  #: skip L2 normalisation
    drop_prompt: bool = False  #: ignore the model's default prompt prefix
    add_prompt: str | None = None  #: apply this named prompt to every input
    batch_mask: bool = False  #: ignore the attention mask in padded batches
    equal_length_batch: bool = False  #: break attention when a batch needs no padding
    truncate_left: bool = False  #: keep the end of over-length inputs
    max_length: int | None = None  #: truncate at this many tokens
    no_special_tokens: bool = False  #: do not add CLS/SEP
    quantize: str | None = None  #: float16, int8 or binary output
    dim: int | None = None  #: cut vectors to this size without renormalising
    base64_float64: bool = False  #: encode base64 payloads as float64
    jitter: float = 0.0  #: add random noise of this scale (non-determinism)
    usage_no_special: bool = False  #: report token usage without special tokens
    reject_long: bool = False  #: answer 413 instead of truncating
    rerank_raw: bool = False  #: return logits where the model declares sigmoid
    rerank_batch: bool = False  #: rerank scores depend on the other documents

    @staticmethod
    def parse(spec: str | None) -> Faults:
        """Parse ``"pooling=cls,no-normalize,dim=32"``."""
        faults = Faults()
        if not spec:
            return faults
        types = {f.name: f.type for f in fields(Faults)}
        for raw in spec.split(","):
            item = raw.strip()
            if not item:
                continue
            name, _, value = item.partition("=")
            key = name.strip().replace("-", "_")
            if key not in types:
                known = ", ".join(sorted(k.replace("_", "-") for k in types))
                raise ValueError(f"unknown fault {name!r}; known faults: {known}")
            kind = str(types[key])
            if kind.startswith("bool"):
                setattr(faults, key, value.strip().lower() not in ("0", "false", "no", "off"))
            elif kind.startswith("int"):
                setattr(faults, key, int(value))
            elif kind.startswith("float"):
                setattr(faults, key, float(value))
            else:
                if not value:
                    raise ValueError(f"fault {name!r} needs a value, e.g. {name}=...")
                setattr(faults, key, value.strip())
        if faults.pooling is not None and faults.pooling not in POOLINGS:
            raise ValueError(f"pooling fault must be one of {', '.join(POOLINGS)}")
        if faults.quantize is not None and faults.quantize not in ("float16", "int8", "binary"):
            raise ValueError("quantize fault must be float16, int8 or binary")
        return faults

    def active(self) -> list[str]:
        out = []
        for f in fields(Faults):
            value = getattr(self, f.name)
            if value in (None, False, 0.0):
                continue
            name = f.name.replace("_", "-")
            out.append(name if value is True else f"{name}={value}")
        return out


class HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class SimModel:
    """The serving logic, independent of HTTP so it can be unit-tested."""

    def __init__(self, model: str = "toy-embed", faults: Faults | None = None) -> None:
        if model not in TOY_MODELS:
            raise KeyError(f"unknown toy model {model!r}; known: {', '.join(TOY_MODELS)}")
        self.model_name = model
        self.encoder = ToyEncoder(TOY_MODELS[model])
        self.reranker = ToyReranker()
        self.faults = faults or Faults()
        self._rng = np.random.default_rng(0)
        self._lock = threading.Lock()

    def input_config(self, truncate: bool = True) -> InputConfig:
        info = self.encoder.describe()
        f = self.faults
        prompt_name = info.default_prompt_name
        if f.drop_prompt:
            prompt_name = None
        if f.add_prompt:
            if f.add_prompt not in info.prompts:
                raise HTTPError(500, f"simulator fault names unknown prompt {f.add_prompt!r}")
            prompt_name = f.add_prompt
        return InputConfig(
            prompt=info.prompts.get(prompt_name, "") if prompt_name else "",
            prompt_name=prompt_name,
            add_special_tokens=not f.no_special_tokens,
            max_length=min(f.max_length or info.max_seq_length, info.position_limit or 10**9)
            if truncate
            else 10**9,
            truncation_side="left" if f.truncate_left else info.truncation_side,
        )

    def embed(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        dimensions: int | None = None,
        truncate: bool = True,
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Return vectors and the token count the server reports."""
        f = self.faults
        spec = self.encoder.spec
        cfg = self.input_config()
        limit = cfg.max_length
        built = [self.encoder.build_ids(t, cfg) for t in texts]
        if f.reject_long or not truncate:
            for _, full in built:
                if full > limit:
                    raise HTTPError(
                        413, f"Input validation error: `inputs` must have less than {limit} tokens"
                    )
        ids = [b[0] for b in built]
        lengths = {len(i) for i in ids}
        if f.batch_mask and len(lengths) > 1:
            states = self.encoder.forward(ids, respect_mask=False)
        elif f.equal_length_batch and len(ids) > 1 and len(lengths) == 1:
            states = self.encoder.forward(ids, uniform_attention=True)
        else:
            states = [self.encoder.forward([i])[0] for i in ids]
        mode = f.pooling or spec.pooling
        out = np.stack([pool(s, mode) for s in states]).astype(np.float64)

        def unit(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12))

        do_norm = spec.normalize and normalize and not f.no_normalize
        if dimensions:
            if dimensions > out.shape[1] or dimensions < 1:
                raise HTTPError(400, f"dimensions must be between 1 and {out.shape[1]}")
            out = out[:, :dimensions]
        if do_norm:
            out = unit(out)
        if f.dim:
            out = out[:, : f.dim]
        if f.jitter:
            with self._lock:
                out = out + self._rng.normal(0.0, f.jitter, size=out.shape)
        if f.quantize == "float16":
            out = out.astype(np.float16).astype(np.float64)
        elif f.quantize == "int8":
            out = np.clip(np.round(out * 127.0), -128, 127)
        elif f.quantize == "binary":
            out = (out > 0).astype(np.float64)
        else:
            out = out.astype(np.float32).astype(np.float64)
        tokens = sum(len(i) for i in ids)
        if f.usage_no_special and cfg.add_special_tokens:
            tokens -= 2 * len(ids)
        return out, tokens

    def rerank(
        self, query: str, documents: list[str], *, raw_scores: bool = False
    ) -> tuple[npt.NDArray[np.float64], int]:
        logits = self.reranker.logits([(query, d) for d in documents]).astype(np.float64)
        if self.faults.rerank_batch and len(documents) > 1:
            logits = logits + 0.4 * np.cos(np.arange(len(documents)) * 1.7 + len(documents))
        raw = raw_scores or self.faults.rerank_raw
        scores = logits if raw else 1.0 / (1.0 + np.exp(-logits))
        tokens = sum(self.reranker.count_tokens(query, d) for d in documents)
        return np.asarray(scores.astype(np.float32), dtype=np.float64), tokens


def _texts(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value) and value:
        return list(value)
    raise HTTPError(422, f"`{field_name}` must be a string or a non-empty list of strings")


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": self.server.sim.model_name})
        elif self.path in ("/v1/models", "/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.server.sim.model_name, "object": "model"}],
                },
            )
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                raise HTTPError(400, "request body is not valid JSON") from exc
            if not isinstance(body, dict):
                raise HTTPError(400, "request body must be a JSON object")
            path = self.path.split("?", 1)[0].rstrip("/")
            self._send(200, self._route(path, body))
        except HTTPError as exc:
            self._send(exc.status, {"error": str(exc), "error_type": "Validation"})
        except Exception as exc:  # pragma: no cover - defensive
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _route(self, path: str, body: dict[str, Any]) -> Any:
        sim = self.server.sim
        if path in ("/v1/embeddings", "/embeddings"):
            texts = _texts(body.get("input"), "input")
            vectors, tokens = sim.embed(texts, dimensions=body.get("dimensions") or None)
            fmt = body.get("encoding_format", "float")
            if fmt not in ("float", "base64"):
                raise HTTPError(400, "encoding_format must be 'float' or 'base64'")
            data = []
            for index, vec in enumerate(vectors):
                emb: Any = vec.tolist()
                if fmt == "base64":
                    dtype = "<f8" if sim.faults.base64_float64 else "<f4"
                    emb = base64.b64encode(vec.astype(dtype).tobytes()).decode("ascii")
                data.append({"object": "embedding", "index": index, "embedding": emb})
            return {
                "object": "list",
                "data": data,
                "model": body.get("model") or sim.model_name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        if path == "/embed":
            texts = _texts(body.get("inputs"), "inputs")
            vectors, _ = sim.embed(
                texts,
                normalize=bool(body.get("normalize", True)),
                dimensions=body.get("dimensions") or None,
                truncate=bool(body.get("truncate", False)),
            )
            return vectors.tolist()
        if path == "/api/embed":
            texts = _texts(body.get("input"), "input")
            vectors, tokens = sim.embed(
                texts,
                dimensions=body.get("dimensions") or None,
                truncate=bool(body.get("truncate", True)),
            )
            return {
                "model": body.get("model") or sim.model_name,
                "embeddings": vectors.tolist(),
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": tokens,
            }
        if path in ("/rerank", "/v1/rerank", "/v2/rerank"):
            query = body.get("query")
            if not isinstance(query, str):
                raise HTTPError(422, "`query` must be a string")
            if "texts" in body:  # TEI shape
                docs = _texts(body.get("texts"), "texts")
                scores, _ = sim.rerank(query, docs, raw_scores=bool(body.get("raw_scores", False)))
                order = np.argsort(-scores, kind="stable")
                return [{"index": int(i), "score": float(scores[i])} for i in order]
            docs = _texts(body.get("documents"), "documents")
            scores, tokens = sim.rerank(query, docs)
            order = np.argsort(-scores, kind="stable")
            top_n = body.get("top_n")
            if isinstance(top_n, int) and top_n > 0:
                order = order[:top_n]
            return {
                "id": "rerank-sim",
                "model": body.get("model") or sim.reranker.name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                "results": [{"index": int(i), "relevance_score": float(scores[i])} for i in order],
            }
        raise HTTPError(404, f"no route {path}")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], sim: SimModel, verbose: bool) -> None:
        super().__init__(address, _Handler)
        self.sim = sim
        self.verbose = verbose


class SimServer:
    """Run the simulator on a local port. Usable as a context manager."""

    def __init__(
        self,
        model: str = "toy-embed",
        faults: Faults | str | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        verbose: bool = False,
    ) -> None:
        parsed = Faults.parse(faults) if isinstance(faults, str) or faults is None else faults
        self.sim = SimModel(model, parsed)
        self._httpd = _Server((host, port), self.sim, verbose)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self._httpd.server_address[0]!s}:{self.port}"

    def start(self) -> SimServer:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        try:
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> SimServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
