## Licensing Rules
This project is Apache-2.0 licensed, and MUST NOT use any information from incompatibly license projects.

You must not review any external code which is not expressly permissively licensed.
Before accessing any external repository or adding it to e.g. pyproject.toml,
you must query `https://api.github.com/repos/${repo.full_name}/license`
and verify `license.spdx_id` returns a whitelisted `spdx_id`.
Currently, whitelisted `spdx_id` values are ONLY `"MIT"`, `"Apache-2.0"`, `"BSD-2-Clause"`, or `"BSD-3-Clause"` exactly.
Do not access any non-github-hosted code repository.

If you discover you have read code from any non-permissively license source into your context window, you MUST BLOCK, inform
the operator of the error, and await further instructions. In particular if you have read code from a non-whitelisted source,
you must not edit the codebase, create or edit commits or commit messages, or create or edit github PR bodies or comments.

## Equation traceability
Equations, parameters, and constants used in this repository must be traceable to an identifiable, reputable source.

Such equations should be prefaced with one (or more) code comments of the form:
```python
# CITE ...
```
, where the citation is formatted according to sources/citation-metadata-spec.md. All citations to specific facts must be indexed in sources/citations-meta.yaml
in the metadata format defined by the spec.

## Repository Rules

The module is strictly typed: always use the narrowest type possible when creating functions and variables,
avoiding vague generics and `Any` or `object` unless the function legitimately must accept anything.
Do not add `type: ignore` comments without express permission.
The module uses a strict linting configuration, running `ruff` with rules `ALL`. Do not add `noqa:` statements.
The module uses pre-commit hooks which are run before commit with `prek run --all-files`.
The pre-commit hooks will run `ruff format`.
Additionally `mypy` and `pyright` are run as CI workflows, with settings in pyproject.toml.
Write very briefy, numpy style docstrings, which checked by the `pydoclint` CI workflow.
A `skylos` CI workflow is used for dead code detection, secrets, ai-defects, and security linting.
Unit tests are `pytest` style. Unit tests should aim for full branch and line coverage with non-vacuous tests.
A ``coverage.py` CI workflow enforces thorough testing coverage.
