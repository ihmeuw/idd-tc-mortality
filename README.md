# idd-tc-mortality

Double-hurdle peaks-over-threshold model for tropical cyclone mortality estimation.

See [.claude/STATUS.md](.claude/STATUS.md) for current goals and pipeline state.

## Development

Local dev uses a project `.venv` with a uv-managed interpreter — no conda
anywhere in the Python story. uv owns the interpreter and every package
(including installing idd-tc-mortality itself editable — no `PYTHONPATH=src`).
One-time setup:

```bash
uv venv --python 3.12        # .venv on the uv-managed CPython
uv sync --all-extras         # runtime (incl. jobmon) + dev
```

The jobmon pin resolves from the IHME artifactory, so setup needs
artifactory access (anonymous read works from the cluster).

Run code via `.venv/bin/python` by absolute path (Slurm/jobmon task
commands do exactly this — no activation of any kind);
`source .venv/bin/activate` is optional for interactive shells.

Day-to-day, re-sync after pulling or editing deps:

```bash
uv sync --inexact   # add/update declared deps, keep ad-hoc installs
uv sync             # exact clean rebuild (env == lockfile)
```

Run the checks — with the venv activated, call the tools directly (or
prefix each with `.venv/bin/`):

```bash
pytest tests/
ruff check src tests
mypy src
```
