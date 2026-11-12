## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## How it was tested

<!-- New or changed tests; for server-facing changes, the server and model you ran against. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy` passes
- [ ] `pytest` passes
- [ ] If report output changed: `EMBED_PARITY_UPDATE_README=1 pytest tests/test_readme.py` was run and the README diff reviewed
- [ ] If a fault or variant was added: a test shows the diagnosis names it
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Claims about server behaviour cite documentation or a run, and the README says which
