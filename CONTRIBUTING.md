# Contributing

Contributions that improve portability, validation, documentation, or reproducibility are welcome.

Before submitting a change:

```bash
python -m pip install -e '.[dev]'
ruff format --check src tests
ruff check src tests
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m build
```

Do not commit API keys, cookies, access tokens, copied source pages, review traces, model outputs containing sensitive data, or machine-specific paths. Dataset corrections should name the affected `benchmark_id`, explain the issue, and link to public supporting evidence.
