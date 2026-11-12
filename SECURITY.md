# Security policy

## Supported versions

Security fixes are made on the latest released version.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/cb-Sigma/embed-parity/security/advisories/new)
rather than in a public issue. Include what you found, how to reproduce it and which
version is affected. You should receive an acknowledgement within a week.

## Things to know when using the tool

- **`reference` loads models.** Building a reference loads a model with
  sentence-transformers. Loading a model repository you do not trust carries the usual
  risks; custom model code is executed only if you pass `--trust-remote-code`.
- **`check` sends requests to the URL you give it**, including the API key from `--api-key`
  or `EMBED_PARITY_API_KEY` as a bearer token and any `--header` values. Keys are not written
  to reports. Error messages can include the first part of a server's error body.
- **Reference files are JSON.** They are parsed with the standard `json` module and contain
  base64-encoded arrays; nothing in them is executed. Probe texts in a reference file are
  sent to the endpoint you check, so treat a reference from an untrusted source as you would
  any input that decides what your client sends.
- **The simulated server** (`embed-parity serve-sim`, `embed-parity demo`) is a test double
  with no authentication. It binds to `127.0.0.1` by default; do not expose it.
- **`examples/transformers_server.py`** is a test fixture, not a production server.
