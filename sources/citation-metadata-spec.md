# Paper Citation Metadata Specification

This document specifies a structured YAML file for paper metadata that is safe to commit to version control while the source PDFs and paper-summary Markdown files remain local. The committed YAML records stable paper identifiers, local filenames, content hashes, bibliographic metadata, fact identifiers, and the code locations that depend on those facts.

The metadata filename is `sources/citations-meta.yaml`.
The bibtex citation file filename is `sources/citations.bib`

## Goals

- Keep copyrighted PDFs and derived long-form summaries out of version control.
- Allow source comments to cite papers and specific facts in a machine-readable form.
- Allow a reviewer or tool to verify that a citation was checked against an exact local PDF and summary version by comparing SHA256 hashes.
- Provide a repository-wide cross-reference from paper facts to code locations.
- Keep the committed metadata factual and brief: titles, authors, identifiers, page/equation/table references, short claim summaries, and hashes are acceptable; copied article text, long abstracts, figures, tables, and long equation derivations are not.

## YAML File Shape

The top-level YAML document MUST be a mapping with these fields:

```yaml
schema_version: 1
papers:
  <paper_id>:
    local_files:
      pdf:
        filename: <string>
        sha256: <64 lowercase hex chars>
      summary:
        filename: <string>
        sha256: <64 lowercase hex chars>
    citation:
      preferred: <string>
      ads_bibcode: <string | null>
    topics:
      - <string>
    facts:
      <fact_id>:
        summary: <short string>
        evidence:
          pages:
            - <integer | string>
          equations:
            - <string>
          tables:
            - <string>
          figures:
            - <string>
        notes: <string | null>
    code_uses:
      - fact: <fact_id>
        locations:
          - path: <repo-relative path>
            symbol: <string | null>
            lines: <string | null>
            comment_ref: <string>
        purpose: <short string>
        checked_against:
          pdf_sha256: <64 lowercase hex chars>
          summary_sha256: <64 lowercase hex chars>
```

Fields MAY be extended in later schema versions, but tools SHOULD reject unknown `schema_version` values unless explicitly configured to tolerate them.

Each citation entry should have a corresponding entry in `sources/citations.bib`, with a code matching `preferred`. `preferred` should be the `ads_bibcode`
unless an ads bibcode is not available. Citations identified by ads bibcode MUST match the full ads bibtex citation exactly.

## Identifier Rules

`paper_id` MUST be unique within the repository and stable once introduced. It MUST match:

```text
^[a-z0-9][a-z0-9-]*$
```

Use short lowercase identifiers such as `cramer-1993`. Do not rename an identifier after code comments start citing it.

`fact_id` MUST be unique within its paper and SHOULD also match `^[a-z0-9][a-z0-9-]*$`. Fact IDs should be stable labels for specific claims, not broad paper sections. Good examples:

- `zero-frequency-sound-speed`
- `humidity-mole-fraction`
- `oxygen-relaxation-frequency`

## Local File Rules

`local_files.pdf.filename` and `local_files.summary.filename` MUST be paths relative to the `papers/` directory. They identify local files that are intentionally not committed.

`sha256` values MUST be the lowercase SHA256 digest of the exact file bytes:

```bash
shasum -a 256 papers/path/to/file.pdf
shasum -a 256 papers/path/to/file.md
```

Before validating a code citation, tooling SHOULD confirm that both local files exist and match their recorded hashes. If either hash differs, BLOCK and await maintainer instructions.

## Machine-Readable Code Comments

Code comments SHOULD cite paper facts with this format:

```python
# CITE paper=<paper_id> fact=<fact_id>
```

Multiple fact citations MAY appear on separate comment lines:

```python
# CITE paper=cramer-1993 fact=zero-frequency-sound-speed
# CITE paper=cramer-1993 fact=humidity-mole-fraction
```

The `comment_ref` field in `code_uses` MUST contain the exact `paper=<paper_id> fact=<fact_id>` payload expected in code. This lets a simple scanner compare code comments against the YAML index without parsing prose.

Line numbers in `code_uses.locations[].lines` are optional because they drift during edits. Prefer `path` plus `symbol`; include `lines` only as a convenience snapshot.

## Fact Entries

Each `facts` entry SHOULD describe one checkable fact from the paper. `summary` must be a concise paraphrase, not copied article text. `evidence` should point reviewers to the precise part of the local PDF or summary:

- `pages`: PDF page numbers or paper-stated page labels.
- `equations`: paper equation labels such as `Eq. (4)`.
- `tables`: paper table labels such as `Table I`.
- `figures`: paper figure labels if a fact comes from a figure.

Use `notes` for repository-specific interpretation, limitations, or unit conversions. Keep notes short and avoid reproducing copyrighted content.

## Code Use Index

`code_uses` is the reverse index from paper facts to code. Each entry MUST name one `fact` and one or more code `locations`.

`checked_against.pdf_sha256` and `checked_against.summary_sha256` SHOULD normally repeat the current hashes from `local_files`. They record which exact local artifacts were used when that code use was reviewed. This makes it possible to detect citations that need rechecking after a local paper or summary changes.

If a code location uses several facts from the same paper, prefer separate `code_uses` entries so each fact remains independently searchable.

## Validation Rules

A validator SHOULD check:

- `schema_version` is supported.
- Every `paper_id` and `fact_id` matches the identifier pattern.
- Every PDF and summary filename is relative, stays under `papers/`, and has the expected extension.
- Every SHA256 digest is exactly 64 lowercase hexadecimal characters.
- Every `code_uses[].fact` exists under the same paper's `facts`.
- Every `comment_ref` exactly matches `paper=<paper_id> fact=<fact_id>`.
- Every referenced code path exists.
- Local PDF and summary files, when present, hash to the recorded values.
- No field intended for metadata contains long copied passages from a paper.
