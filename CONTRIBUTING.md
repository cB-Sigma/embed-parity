# Contributing

Thanks for considering a contribution. The most valuable ones right now are reports from
real deployments: most rows of the README's support table say "docs only".

## Reporting a result from a real server

Run the check with `--json report.json` and open a
[server compatibility report](https://github.com/cb-Sigma/embed-parity/issues/new?template=server_report.yml)
with the server version, model and revision, the command and the report. Passing runs are
as useful as failing ones. If the diagnosis was wrong or could not express what you found,
say what the actual cause turned out to be.

## Development setup

```bash
git clone https://github.com/cb-Sigma/embed-parity
cd embed-parity
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The core package and the whole default test-suite need only numpy. Before sending a pull
request:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

### Tests against a real model

These download weights from the Hugging Face Hub and are skipped by default:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only wheel
pip install -e ".[reference]"
EMBED_PARITY_REAL_MODEL=1 pytest tests/test_real_model.py
```

Set `EMBED_PARITY_REAL_MODEL_ID` to try another model; the assertions about declared
configuration are specific to `sentence-transformers/all-MiniLM-L6-v2`.

### README output blocks

Blocks in the README between `<!-- output: ... -->` and `<!-- /output -->` are produced by
the tool, and `tests/test_readme.py` fails when they drift. After an intentional change:

```bash
EMBED_PARITY_UPDATE_README=1 pytest tests/test_readme.py
```

and review the diff.

## Design rules

- **Core stays numpy-only.** Anything that needs torch or sentence-transformers goes in
  `st_backend.py` and is imported lazily. HTTP uses the standard library.
- **A new fault needs a test that names it.** If you add a fault to the simulator or a
  variant to the reference, add a case to `tests/test_fault_diagnosis.py` asserting on the
  words of the diagnosis.
- **Say "no match" rather than guess.** A diagnosis that names the wrong cause costs more
  time than one that names none.
- **Claims need a source.** Statements about how a server behaves must come from its
  documentation (add the link and date to `docs/api-shapes.md`) or from a run (add it to
  `docs/validation.md`). The README distinguishes the two; keep it that way.
- **Reference files are a format.** Additive changes keep `version` at its current value;
  anything that changes the meaning of existing fields bumps it and keeps a reader for the
  old one.

## Adding an API shape

1. Add the request and response handling in `client.py`.
2. Add the route to the simulator in `simserver.py` so the shape is exercised in tests.
3. Document the shape and its source in `docs/api-shapes.md`.
4. Add a row to the README support table marked "docs only" unless you ran it.

## Releasing

Maintainers: update `__version__` in `src/embed_parity/__init__.py` and `CHANGELOG.md`, then
tag `vX.Y.Z`. The release workflow builds the distribution and creates a GitHub release.

Publishing to PyPI is disabled until the repository variable `PYPI_PUBLISH` is set to
`true`. Note that the name `embed-parity` on PyPI currently belongs to an unrelated
project, so publishing requires choosing a different distribution name first (and
configuring a trusted publisher for it).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Licence

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
