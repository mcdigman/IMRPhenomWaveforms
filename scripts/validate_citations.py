# Copyright (c) 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Validate citation metadata and BibTeX entries against the ADS export API."""

import argparse
import http.client
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

ADS_HOST = 'api.adsabs.harvard.edu'
ADS_EXPORT_PATH = '/v1/export/bibtex'
ADS_TIMEOUT_SECONDS = 60
BIB_ENTRY_START = re.compile(r'^@[A-Za-z]+\{([^,\s]+),', re.MULTILINE)
BIBCODE = re.compile(r"\d{4}[A-Za-z0-9&.:+'-]{14}[A-Za-z.]")


class CitationError(Exception):
    """A citation file or ADS response could not be processed."""


def parse_bib_entries(text: str) -> dict[str, str]:
    """
    Split BibTeX text into raw entries without normalizing them.

    Parameters
    ----------
    text : str
        BibTeX text.

    Returns
    -------
    dict[str, str]
        Entry text, without trailing whitespace, by citation key.

    Raises
    ------
    CitationError
        If text precedes the first entry or a key is duplicated.
    """
    matches = list(BIB_ENTRY_START.finditer(text))
    preamble_end = matches[0].start() if matches else len(text)
    if text[:preamble_end].strip():
        msg = 'BibTeX text contains content outside an entry'
        raise CitationError(msg)

    entries: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        key = match.group(1)
        if key in entries:
            msg = f'duplicate BibTeX key {key!r}'
            raise CitationError(msg)
        entries[key] = text[match.start() : end].rstrip()
    return entries


def parse_metadata_keys(text: str) -> tuple[set[str], set[str]]:
    """
    Read the BibTeX keys required by the citation metadata.

    Parameters
    ----------
    text : str
        Text of the citation metadata YAML file.

    Returns
    -------
    tuple[set[str], set[str]]
        ADS bibcodes, and the preferred keys of papers without a bibcode.

    Raises
    ------
    CitationError
        If the metadata is not valid YAML or does not have the expected shape.
    """
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = 'citation metadata is not valid YAML'
        raise CitationError(msg) from exc

    if loaded is None:
        return set(), set()
    papers = loaded.get('papers') if isinstance(loaded, dict) else None
    if not isinstance(papers, dict):
        msg = 'citation metadata must be a mapping with a papers mapping'
        raise CitationError(msg)

    bibcodes: set[str] = set()
    non_ads_keys: set[str] = set()
    for paper_id, paper in papers.items():
        citation = paper.get('citation') if isinstance(paper, dict) else None
        if not isinstance(citation, dict):
            msg = f'paper {paper_id!r} has no citation mapping'
            raise CitationError(msg)
        bibcode = citation.get('ads_bibcode')
        preferred = citation.get('preferred')
        if isinstance(bibcode, str):
            bibcodes.add(bibcode)
        elif bibcode is None and isinstance(preferred, str):
            non_ads_keys.add(preferred)
        else:
            msg = f'paper {paper_id!r} needs a string ads_bibcode, or a null ads_bibcode and a string preferred'
            raise CitationError(msg)
    return bibcodes, non_ads_keys


def fetch_ads_bibtex(bibcodes: Sequence[str], token: str) -> str:
    """
    Export BibTeX for a batch of bibcodes from ADS.

    Parameters
    ----------
    bibcodes : Sequence[str]
        Bibcodes to export.
    token : str
        ADS API token.

    Returns
    -------
    str
        BibTeX text returned by ADS.

    Raises
    ------
    CitationError
        If the token is missing or ADS does not return a usable response.
    """
    if not token:
        msg = 'ADS request failed: ADS_DEV_KEY is not set'
        raise CitationError(msg)

    # http.client never follows redirects, so the token only reaches ADS_HOST.
    connection = http.client.HTTPSConnection(ADS_HOST, timeout=ADS_TIMEOUT_SECONDS)
    try:
        connection.request(
            'POST',
            ADS_EXPORT_PATH,
            body=json.dumps({'bibcode': list(bibcodes)}),
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        )
        response = connection.getresponse()
        status = response.status
        body = response.read()
    except (OSError, http.client.HTTPException) as exc:
        msg = f'ADS request failed: {type(exc).__name__}'
        raise CitationError(msg) from exc
    finally:
        connection.close()

    if status != http.client.OK:
        msg = f'ADS request failed: HTTP {status}'
        raise CitationError(msg)
    try:
        payload: object = json.loads(body)
    except ValueError as exc:
        msg = 'ADS request failed: response is not valid JSON'
        raise CitationError(msg) from exc
    export = payload.get('export') if isinstance(payload, dict) else None
    if not isinstance(export, str):
        msg = 'ADS request failed: response has no export text'
        raise CitationError(msg)
    return export


def first_difference_line(first: str, second: str) -> int:
    """
    Find the first line at which two different texts diverge.

    Parameters
    ----------
    first : str
        First text.
    second : str
        Second text.

    Returns
    -------
    int
        One-based line number of the first difference.
    """
    first_lines = first.split('\n')
    second_lines = second.split('\n')
    for number, (first_line, second_line) in enumerate(zip(first_lines, second_lines, strict=False), start=1):
        if first_line != second_line:
            return number
    return min(len(first_lines), len(second_lines)) + 1


def compare_with_ads(local: dict[str, str], ads_text: str) -> list[str]:
    """
    Compare local BibTeX entries byte for byte with the ADS export.

    Parameters
    ----------
    local : dict[str, str]
        Local entry text by citation key.
    ads_text : str
        BibTeX text returned by ADS.

    Returns
    -------
    list[str]
        One message per mismatch.
    """
    remote = parse_bib_entries(ads_text)
    errors: list[str] = []
    for key, entry in sorted(local.items()):
        if key not in remote:
            errors.append(f'ADS returned no record for {key!r}')
        elif entry != remote[key]:
            line = first_difference_line(entry, remote[key])
            errors.append(f'entry {key!r} differs from ADS, first at line {line} of the entry')
    errors.extend(f'ADS returned unexpected record {key!r}' for key in sorted(remote.keys() - local.keys()))
    return errors


def validate(bib_text: str, metadata_text: str, fetch: Callable[[Sequence[str]], str]) -> list[str]:
    """
    Run the local citation checks, then the ADS comparison if they pass.

    Parameters
    ----------
    bib_text : str
        Text of the BibTeX file.
    metadata_text : str
        Text of the citation metadata YAML file.
    fetch : Callable[[Sequence[str]], str]
        Returns the BibTeX export for a batch of bibcodes.

    Returns
    -------
    list[str]
        One message per problem found.
    """
    entries = parse_bib_entries(bib_text)
    bibcodes, non_ads_keys = parse_metadata_keys(metadata_text)

    errors = [f'metadata ads_bibcode {key!r} is not in the BibTeX file' for key in sorted(bibcodes - entries.keys())]
    errors += [
        f'metadata preferred key {key!r} is not in the BibTeX file' for key in sorted(non_ads_keys - entries.keys())
    ]
    # A null ads_bibcode exempts a key from the ADS comparison, so it must not be
    # usable to exempt an entry that ADS can check.
    errors += [
        f'metadata preferred key {key!r} has a null ads_bibcode but is shaped like a bibcode'
        for key in sorted(non_ads_keys)
        if BIBCODE.fullmatch(key)
    ]
    errors += [
        f'metadata key {key!r} is referenced both with and without an ads_bibcode'
        for key in sorted(bibcodes & non_ads_keys)
    ]

    ads_entries = {key: entry for key, entry in entries.items() if key not in non_ads_keys}
    errors += [
        f'BibTeX key {key!r} is not a valid ADS bibcode' for key in sorted(ads_entries) if not BIBCODE.fullmatch(key)
    ]
    # Local failures are fixable without ADS, so do not spend API quota on them.
    if errors or not ads_entries:
        return errors

    for key in sorted(non_ads_keys):
        sys.stdout.write(f'skipping non-ADS citation {key!r}\n')
    return compare_with_ads(ads_entries, fetch(sorted(ads_entries)))


def main(argv: Sequence[str] | None = None) -> int:
    """
    Validate the citation files.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command line arguments, or None to read them from sys.argv.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bib', type=Path, default=Path('sources/citations.bib'))
    parser.add_argument('--metadata', type=Path, default=Path('sources/citations-meta.yaml'))
    args = parser.parse_args(argv)
    token = os.environ.get('ADS_DEV_KEY', '')

    try:
        # Decode the bytes directly: text mode would normalize line endings.
        bib_text = args.bib.read_bytes().decode('utf-8')
        metadata_text = args.metadata.read_bytes().decode('utf-8')
        errors = validate(bib_text, metadata_text, lambda bibcodes: fetch_ads_bibtex(bibcodes, token))
    except (CitationError, OSError, UnicodeDecodeError) as exc:
        errors = [f'{type(exc).__name__}: {exc}']

    # Every line starts with a fixed prefix and untrusted values are repr-escaped,
    # so file contents cannot inject GitHub Actions workflow commands into the log.
    for error in errors:
        sys.stderr.write(f'error: {error}\n')
    if errors:
        return 1
    sys.stdout.write('citations validated\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
