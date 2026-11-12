#!/usr/bin/env python3
"""A small embedding server written directly on transformers, for testing embed-parity.

It deliberately does NOT use sentence-transformers: pooling, normalisation and
truncation are re-implemented here, the way a serving stack re-implements
them. That makes it a realistic target, and its switches reproduce real
classes of bugs:

    --pooling cls        pool the wrong way
    --no-normalize       forget the Normalize module
    --ignore-mask        average over padding positions (batch-dependent output)
    --max-length 128     truncate somewhere else

Routes: POST /v1/embeddings (OpenAI shape), POST /v1/rerank (Cohere/Jina shape,
needs --rerank-model), GET /health.

    pip install torch transformers
    python examples/transformers_server.py --model sentence-transformers/all-MiniLM-L6-v2
    embed-parity check --base-url http://127.0.0.1:8088 \
        --model-id sentence-transformers/all-MiniLM-L6-v2

This is a test fixture, not a production server.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class Engine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModel.from_pretrained(args.model).eval()
        self.rerank_tokenizer = self.rerank_model = None
        if args.rerank_model:
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(args.rerank_model)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                args.rerank_model
            ).eval()

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        a = self.args
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=a.max_length, return_tensors="pt"
        )
        with self.lock, torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if a.ignore_mask:
            mask = torch.ones_like(mask)
        if a.pooling == "cls":
            pooled = hidden[:, 0]
        elif a.pooling == "max":
            pooled = hidden.masked_fill(mask == 0, -1e9).max(dim=1).values
        elif a.pooling == "last":
            last = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0]), last]
        else:
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if not a.no_normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.numpy().astype(np.float32), int(batch["attention_mask"].sum())

    def rerank(self, query: str, documents: list[str]) -> np.ndarray:
        if self.rerank_model is None or self.rerank_tokenizer is None:
            raise LookupError("started without --rerank-model")
        batch = self.rerank_tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self.lock, torch.no_grad():
            logits = self.rerank_model(**batch).logits.reshape(-1)
        if not self.args.rerank_raw:
            logits = torch.sigmoid(logits)
        return logits.numpy().astype(np.float32)


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            path = self.path.rstrip("/")
            if path == "/v1/embeddings":
                texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vectors, tokens = self.engine.embed(texts)
                as_b64 = body.get("encoding_format") == "base64"
                data = [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": base64.b64encode(v.astype("<f4").tobytes()).decode()
                        if as_b64
                        else v.tolist(),
                    }
                    for i, v in enumerate(vectors)
                ]
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": data,
                        "model": body.get("model", self.engine.args.model),
                        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                    },
                )
            elif path in ("/v1/rerank", "/rerank"):
                scores = self.engine.rerank(body["query"], body["documents"])
                order = np.argsort(-scores)
                self._send(
                    200,
                    {
                        "model": body.get("model", self.engine.args.rerank_model),
                        "results": [
                            {"index": int(i), "relevance_score": float(scores[i])} for i in order
                        ],
                    },
                )
            else:
                self._send(404, {"error": f"no route {path}"})
        except (KeyError, TypeError, ValueError, LookupError) as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--pooling", choices=("mean", "cls", "last", "max"), default="mean")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--ignore-mask", action="store_true")
    parser.add_argument("--rerank-raw", action="store_true", help="return logits, not sigmoid")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    torch.set_num_threads(max(torch.get_num_threads() // 2, 1))
    Handler.engine = Engine(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {args.model} on http://{args.host}:{args.port}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
