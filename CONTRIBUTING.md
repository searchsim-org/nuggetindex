# Contributing to nuggetindex

Thanks for considering a contribution. The library is small enough
that any non-trivial change is best discussed in an issue first.

## Development setup

```bash
git clone https://github.com/searchsim-org/nuggetindex
cd nuggetindex
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/unit/                 # fast unit tests, run on every CI build
pytest tests/integration/          # heavier integration tests
NUGGETINDEX_LLM_TESTS=1 pytest -m llm  # tests that hit a real LLM API
```

The CI workflow at `.github/workflows/ci.yml` runs the unit-test job
on every push and PR.

## Coding standards

- `ruff format && ruff check` before committing.
- Public functions get type hints + docstrings; internal helpers do
  not need either.
- Keep individual files focused; if a module passes ~600 lines of
  source it is probably trying to do too much.

## Pull requests

- Open an issue first for anything beyond a typo or a one-line bug
  fix.
- One PR per logical change; commit history is squash-merged so the
  PR description becomes the public commit message.
- Add or update a test that demonstrates the change. PRs without
  tests are accepted only for documentation, formatting, and CI
  changes.

## Filing a bug

Use the issue template in `.github/ISSUE_TEMPLATE/bug_report.md` and
include the output of `pip show nuggetindex` plus the smallest
reproducible script.
