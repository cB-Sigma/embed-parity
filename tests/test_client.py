from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from conftest import status_of
from embed_parity.client import (
    EmbeddingClient,
    EndpointError,
    ProtocolError,
    RerankClient,
    Transport,
    join_url,
)
from embed_parity.reference import Reference
from embed_parity.report import PASS, SKIP, Report
from embed_parity.simserver import SimServer

Sim = Callable[..., SimServer]


def test_join_url() -> None:
    assert join_url("http://h:1", "/v1/embeddings") == "http://h:1/v1/embeddings"
    assert join_url("http://h:1/", "v1/embeddings") == "http://h:1/v1/embeddings"
    assert join_url("http://h:1/v1", "/v1/embeddings") == "http://h:1/v1/embeddings"
    assert join_url("http://h:1/v1/", "/v1/rerank") == "http://h:1/v1/rerank"
    assert join_url("http://h:1/prefix", "/embed") == "http://h:1/prefix/embed"


@pytest.mark.parametrize("api", ["openai", "infinity", "tei", "ollama"])
def test_all_api_shapes_return_the_same_vectors(sim: Sim, api: str) -> None:
    server = sim()
    reference = EmbeddingClient(Transport(server.url), "toy-embed").embed(["a b", "c"])
    got = EmbeddingClient(Transport(server.url), "toy-embed", api=api).embed(["a b", "c"])
    assert got.vectors.shape == (2, 64)
    assert np.allclose(got.vectors, reference.vectors)
    if api == "tei":
        assert got.prompt_tokens is None
    else:
        assert got.prompt_tokens == reference.prompt_tokens


def test_base64_decoding(sim: Sim) -> None:
    client = EmbeddingClient(Transport(sim().url), "toy-embed")
    a = client.embed(["hello"], encoding_format="float").vectors
    b = client.embed(["hello"], encoding_format="base64").vectors
    assert np.array_equal(a, b)
    assert client.requests_made == 2


def test_tei_does_not_send_base64_or_model(sim: Sim) -> None:
    client = EmbeddingClient(Transport(sim().url), "toy-embed", api="tei")
    assert not client.supports_base64
    assert client.route == "/embed"
    assert client.embed(["x"], encoding_format="base64").vectors.shape == (1, 64)


def test_extra_body_and_dimensions(sim: Sim) -> None:
    server = sim()
    client = EmbeddingClient(Transport(server.url), "toy-embed", dimensions=16)
    assert client.embed(["x"]).vectors.shape == (1, 16)
    tei = EmbeddingClient(Transport(server.url), api="tei", extra_body={"normalize": False})
    assert abs(float(np.linalg.norm(tei.embed(["x y z"]).vectors[0])) - 1.0) > 0.05


def test_tei_rejects_long_input_unless_truncate(sim: Sim) -> None:
    server = sim()
    long_text = "word " * 500
    with pytest.raises(EndpointError) as err:
        EmbeddingClient(Transport(server.url), api="tei").embed([long_text])
    assert err.value.status == 413
    assert err.value.client_error
    ok = EmbeddingClient(Transport(server.url), api="tei", extra_body={"truncate": True})
    assert ok.embed([long_text]).vectors.shape == (1, 64)


def test_http_errors(sim: Sim) -> None:
    server = sim()
    with pytest.raises(EndpointError) as err:
        EmbeddingClient(Transport(server.url), path="/nope").embed(["x"])
    assert err.value.status == 404
    with pytest.raises(EndpointError) as err:
        EmbeddingClient(Transport("http://127.0.0.1:9"), "m").embed(["x"])
    assert err.value.status is None
    assert not err.value.client_error
    assert "cannot reach" in str(err.value)


def test_protocol_errors() -> None:
    client = EmbeddingClient(Transport("http://unused"), "m")
    with pytest.raises(ProtocolError, match="missing 'data'"):
        client._parse({"oops": 1}, "float")
    with pytest.raises(ProtocolError, match="expected a JSON object"):
        client._parse([1, 2], "float")
    with pytest.raises(ProtocolError, match="inconsistent shapes"):
        client._parse({"data": [{"embedding": [1.0]}, {"embedding": [1.0, 2.0]}]}, "float")
    with pytest.raises(ProtocolError, match="asked for base64"):
        client._parse({"data": [{"embedding": [1.0]}]}, "base64")
    with pytest.raises(ProtocolError, match="bad base64"):
        client._parse({"data": [{"embedding": "QUJD"}]}, "base64")
    tei = EmbeddingClient(Transport("http://unused"), api="tei")
    with pytest.raises(ProtocolError, match="JSON array"):
        tei._parse({"data": []}, "float")
    ollama = EmbeddingClient(Transport("http://unused"), api="ollama")
    with pytest.raises(ProtocolError, match="missing 'embeddings'"):
        ollama._parse({}, "float")
    with pytest.raises(ValueError, match="unknown embeddings API"):
        EmbeddingClient(Transport("http://unused"), api="grpc")


def test_openai_response_is_reordered_by_index() -> None:
    client = EmbeddingClient(Transport("http://unused"), "m")
    body = {
        "data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}],
        "usage": {"prompt_tokens": 7},
    }
    result = client._parse(body, "float")
    assert result.vectors.ravel().tolist() == [1.0, 2.0]
    assert result.prompt_tokens == 7


def test_headers_and_api_key_are_sent() -> None:
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict[str, str | None] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            seen["auth"] = self.headers.get("Authorization")
            seen["tenant"] = self.headers.get("X-Tenant")
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        transport = Transport(
            f"http://127.0.0.1:{httpd.server_address[1]}",
            headers={"X-Tenant": "search"},
            api_key="s3cret",
        )
        EmbeddingClient(transport, "m").embed(["x"])
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert seen == {"auth": "Bearer s3cret", "tenant": "search"}


@pytest.mark.parametrize("api", ["cohere", "tei", "infinity"])
def test_rerank_api_shapes(sim: Sim, api: str) -> None:
    client = RerankClient(Transport(sim().url), "toy-rerank", api=api)
    docs = ["bread recipe", "tides of the ocean", "ocean tides explained"]
    result = client.rerank("ocean tides", docs)
    assert result.scores.shape == (3,)
    assert int(np.argmin(result.scores)) == 0
    assert sorted(result.returned_order) == [0, 1, 2]
    assert result.returned_order[-1] == 0  # sorted by score, worst last


def test_rerank_incomplete_results(sim: Sim) -> None:
    client = RerankClient(Transport(sim().url), extra_body={"top_n": 1})
    with pytest.raises(ProtocolError, match="top_n"):
        client.rerank("q", ["a", "b"])
    with pytest.raises(ValueError, match="unknown rerank API"):
        RerankClient(Transport("http://unused"), api="x")


@pytest.mark.parametrize("api", ["tei", "ollama", "infinity"])
def test_full_check_through_other_api_shapes(
    check: Callable[..., Report], toy_ref: Reference, api: str
) -> None:
    extra = {"truncate": True} if api == "tei" else {}
    report = check(None, toy_ref, api=api, extra_body=extra)
    assert report.verdict() == PASS
    expected = PASS if api == "infinity" else SKIP
    assert status_of(report, "invariance/base64") == expected
    assert status_of(report, "tokens/usage") == (SKIP if api == "tei" else PASS)
