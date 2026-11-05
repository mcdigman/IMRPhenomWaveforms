# Copyright (c) 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Tests for the citation validation script."""

import http.client
import json
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.validate_citations import (
    CitationError,
    compare_with_ads,
    fetch_ads_bibtex,
    first_difference_line,
    main,
    parse_bib_entries,
    parse_metadata_keys,
    validate,
)

BIBCODE_A = '2016PhRvD..93d4007K'
BIBCODE_B = '2011PhRvD..84h4037A'
ENTRY_A = f'@ARTICLE{{{BIBCODE_A},\n       author = {{{{Khan}}, Sebastian}},\n         year = 2016\n}}'
ENTRY_B = f'@ARTICLE{{{BIBCODE_B},\n       author = {{{{Ajith}}, P.}},\n         year = 2011\n}}'
ENTRY_LOCAL = '@MISC{smith-notes,\n  year = 2020\n}'
BIB_TEXT = f'{ENTRY_A}\n{ENTRY_B}\n'
ADS_TEXT = f'{ENTRY_B}\n\n{ENTRY_A}\n\n'
REPO_ROOT = Path(__file__).resolve().parent.parent


def metadata_for(*citations: tuple[str | None, str]) -> str:
    papers = {
        f'paper-{index}': {'citation': {'preferred': preferred, 'ads_bibcode': bibcode}}
        for index, (bibcode, preferred) in enumerate(citations)
    }
    return json.dumps({'schema_version': 1, 'papers': papers})


class _FakeAds:
    """Record export requests and answer them with fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[list[str]] = []

    def __call__(self, bibcodes: Sequence[str]) -> str:
        self.requests.append(list(bibcodes))
        return self.text


class _FakeResponse:
    """Minimal stand-in for an HTTP response."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body

    def read(self) -> bytes:
        return self.body


class _FakeConnection:
    """Minimal stand-in for an HTTPS connection."""

    def __init__(self, host: str, response: _FakeResponse, error: Exception | None) -> None:
        self.host = host
        self.response = response
        self.error = error
        self.method = ''
        self.path = ''
        self.body = ''
        self.headers: dict[str, str] = {}
        self.closed = False

    def request(self, method: str, path: str, body: str, headers: dict[str, str]) -> None:
        if self.error is not None:
            raise self.error
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers

    def getresponse(self) -> _FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class _FakeServer:
    """Create fake connections and record them."""

    def __init__(self) -> None:
        self.response = _FakeResponse(200, b'{}')
        self.error: Exception | None = None
        self.instances: list[_FakeConnection] = []

    def connect(self, host: str, timeout: int) -> _FakeConnection:
        assert timeout > 0
        connection = _FakeConnection(host, self.response, self.error)
        self.instances.append(connection)
        return connection


@pytest.fixture
def fake_connection(monkeypatch: pytest.MonkeyPatch) -> _FakeServer:
    server = _FakeServer()
    monkeypatch.setattr(http.client, 'HTTPSConnection', server.connect)
    return server


def test_parse_bib_entries_splits_raw_text() -> None:
    assert parse_bib_entries(BIB_TEXT) == {BIBCODE_A: ENTRY_A, BIBCODE_B: ENTRY_B}


def test_parse_bib_entries_accepts_blank_text() -> None:
    assert parse_bib_entries('  \n') == {}


def test_parse_bib_entries_keeps_interior_carriage_returns() -> None:
    entry = ENTRY_A.replace('\n', '\r\n')
    assert parse_bib_entries(entry)[BIBCODE_A] == entry


@pytest.mark.parametrize('text', ['% comment\n' + ENTRY_A, 'stray text'])
def test_parse_bib_entries_rejects_text_outside_entries(text: str) -> None:
    with pytest.raises(CitationError, match='outside an entry'):
        parse_bib_entries(text)


def test_parse_bib_entries_rejects_duplicate_keys() -> None:
    with pytest.raises(CitationError, match='duplicate BibTeX key'):
        parse_bib_entries(f'{ENTRY_A}\n{ENTRY_A}\n')


@pytest.mark.parametrize('text', ['', '# only a comment\n'])
def test_parse_metadata_keys_treats_empty_file_as_no_citations(text: str) -> None:
    assert parse_metadata_keys(text) == (set(), set())


def test_parse_metadata_keys_separates_ads_and_non_ads_papers() -> None:
    text = metadata_for((BIBCODE_A, BIBCODE_A), (None, 'smith-notes'))
    assert parse_metadata_keys(text) == ({BIBCODE_A}, {'smith-notes'})


def test_parse_metadata_keys_rejects_invalid_yaml() -> None:
    with pytest.raises(CitationError, match='not valid YAML'):
        parse_metadata_keys('papers: [')


@pytest.mark.parametrize('text', ['- a list', 'schema_version: 1', 'papers: []'])
def test_parse_metadata_keys_rejects_wrong_top_level_shape(text: str) -> None:
    with pytest.raises(CitationError, match='papers mapping'):
        parse_metadata_keys(text)


@pytest.mark.parametrize('paper', ['not-a-mapping', {'topics': []}, {'citation': 'not-a-mapping'}])
def test_parse_metadata_keys_rejects_paper_without_citation(paper: str | dict[str, str | list[str]]) -> None:
    with pytest.raises(CitationError, match='no citation mapping'):
        parse_metadata_keys(json.dumps({'papers': {'paper-0': paper}}))


@pytest.mark.parametrize(
    'citation',
    [{'ads_bibcode': None}, {'ads_bibcode': 7, 'preferred': 'key'}, {'ads_bibcode': None, 'preferred': 7}],
)
def test_parse_metadata_keys_rejects_unusable_citation(citation: dict[str, str | int | None]) -> None:
    with pytest.raises(CitationError, match='needs a string ads_bibcode'):
        parse_metadata_keys(json.dumps({'papers': {'paper-0': {'citation': citation}}}))


def test_fetch_ads_bibtex_posts_one_batch_with_token(fake_connection: _FakeServer) -> None:
    fake_connection.response = _FakeResponse(200, json.dumps({'export': ADS_TEXT}).encode())

    assert fetch_ads_bibtex([BIBCODE_A, BIBCODE_B], 'secret-token') == ADS_TEXT

    assert len(fake_connection.instances) == 1
    connection = fake_connection.instances[0]
    assert connection.host == 'api.adsabs.harvard.edu'
    assert (connection.method, connection.path) == ('POST', '/v1/export/bibtex')
    assert json.loads(connection.body) == {'bibcode': [BIBCODE_A, BIBCODE_B]}
    assert connection.headers['Authorization'] == 'Bearer secret-token'
    assert connection.closed


def test_fetch_ads_bibtex_requires_token(fake_connection: _FakeServer) -> None:
    with pytest.raises(CitationError, match='ADS_DEV_KEY is not set'):
        fetch_ads_bibtex([BIBCODE_A], '')
    assert fake_connection.instances == []


@pytest.mark.parametrize(
    ('status', 'body', 'message'),
    [
        (429, b'{"export": "ignored"}', 'HTTP 429'),
        (302, b'', 'HTTP 302'),
        (200, b'not json', 'not valid JSON'),
        (200, b'\xff', 'not valid JSON'),
        (200, b'[]', 'no export text'),
        (200, b'{"export": 7}', 'no export text'),
    ],
)
def test_fetch_ads_bibtex_rejects_bad_responses(
    fake_connection: _FakeServer, status: int, body: bytes, message: str
) -> None:
    fake_connection.response = _FakeResponse(status, body)
    with pytest.raises(CitationError, match=f'ADS request failed: .*{message}'):
        fetch_ads_bibtex([BIBCODE_A], 'secret-token')


@pytest.mark.parametrize('error', [TimeoutError('secret-token'), http.client.BadStatusLine('secret-token')])
def test_fetch_ads_bibtex_reports_transport_errors_without_details(
    fake_connection: _FakeServer, error: Exception
) -> None:
    fake_connection.error = error
    with pytest.raises(CitationError, match='ADS request failed') as caught:
        fetch_ads_bibtex([BIBCODE_A], 'secret-token')
    assert 'secret-token' not in str(caught.value)
    assert fake_connection.instances[0].closed


@pytest.mark.parametrize(
    ('first', 'second', 'expected'),
    [('a\nb\nc', 'a\nx\nc', 2), ('a', 'b', 1), ('a\nb', 'a', 2), ('a', 'a\nb', 2)],
)
def test_first_difference_line(first: str, second: str, expected: int) -> None:
    assert first_difference_line(first, second) == expected


def test_compare_with_ads_ignores_order_and_separators() -> None:
    assert compare_with_ads({BIBCODE_A: ENTRY_A, BIBCODE_B: ENTRY_B}, ADS_TEXT) == []


@pytest.mark.parametrize(
    'local_entry',
    [
        ENTRY_A.replace('2016', '2017', 1),
        ENTRY_A.replace('year = 2016', 'year  = 2016'),
        ENTRY_A.replace('\n', '\r\n'),
    ],
)
def test_compare_with_ads_reports_any_byte_difference(local_entry: str) -> None:
    errors = compare_with_ads({BIBCODE_A: local_entry}, ENTRY_A)
    assert len(errors) == 1
    assert 'differs from ADS' in errors[0]


def test_compare_with_ads_reports_missing_and_unexpected_records() -> None:
    assert compare_with_ads({BIBCODE_A: ENTRY_A}, ENTRY_B) == [
        f'ADS returned no record for {BIBCODE_A!r}',
        f'ADS returned unexpected record {BIBCODE_B!r}',
    ]


def test_validate_passes_and_sends_one_sorted_batch() -> None:
    ads = _FakeAds(ADS_TEXT)
    assert validate(BIB_TEXT, metadata_for((BIBCODE_A, BIBCODE_A)), ads) == []
    assert ads.requests == [sorted([BIBCODE_A, BIBCODE_B])]


def test_validate_checks_every_bib_entry_with_empty_metadata() -> None:
    ads = _FakeAds(ENTRY_A)
    errors = validate(BIB_TEXT, '', ads)
    assert errors == [f'ADS returned no record for {BIBCODE_B!r}']


def test_validate_skips_non_ads_entries(capsys: pytest.CaptureFixture[str]) -> None:
    ads = _FakeAds(ENTRY_A)
    bib_text = f'{ENTRY_A}\n{ENTRY_LOCAL}\n'
    assert validate(bib_text, metadata_for((None, 'smith-notes')), ads) == []
    assert ads.requests == [[BIBCODE_A]]
    assert "skipping non-ADS citation 'smith-notes'" in capsys.readouterr().out


def test_validate_makes_no_request_without_ads_entries() -> None:
    ads = _FakeAds('')
    assert validate(ENTRY_LOCAL, metadata_for((None, 'smith-notes')), ads) == []
    assert validate('', '', ads) == []
    assert ads.requests == []


def test_validate_reports_missing_keys_without_querying_ads() -> None:
    ads = _FakeAds(ADS_TEXT)
    metadata = metadata_for((BIBCODE_B, BIBCODE_B), (None, 'missing-key'))
    assert validate(ENTRY_A, metadata, ads) == [
        f'metadata ads_bibcode {BIBCODE_B!r} is not in the BibTeX file',
        "metadata preferred key 'missing-key' is not in the BibTeX file",
    ]
    assert ads.requests == []


def test_validate_rejects_invalid_bibcode_without_querying_ads() -> None:
    ads = _FakeAds(ADS_TEXT)
    assert validate(ENTRY_LOCAL, '', ads) == ["BibTeX key 'smith-notes' is not a valid ADS bibcode"]
    assert ads.requests == []


def test_validate_rejects_null_ads_bibcode_for_bibcode_shaped_key() -> None:
    ads = _FakeAds(ENTRY_A)
    tampered = ENTRY_A.replace('year = 2016', 'year = 2017')
    assert validate(tampered, metadata_for((None, BIBCODE_A)), ads) == [
        f'metadata preferred key {BIBCODE_A!r} has a null ads_bibcode but is shaped like a bibcode',
    ]
    assert ads.requests == []


def test_validate_rejects_bibcode_also_referenced_with_null_ads_bibcode() -> None:
    ads = _FakeAds(ENTRY_A)
    tampered = ENTRY_A.replace('year = 2016', 'year = 2017')
    metadata = metadata_for((BIBCODE_A, BIBCODE_A), (None, BIBCODE_A))
    assert validate(tampered, metadata, ads) == [
        f'metadata preferred key {BIBCODE_A!r} has a null ads_bibcode but is shaped like a bibcode',
        f'metadata key {BIBCODE_A!r} is referenced both with and without an ads_bibcode',
    ]
    assert ads.requests == []


def test_validate_rejects_non_bibcode_key_referenced_both_ways() -> None:
    ads = _FakeAds('')
    metadata = metadata_for(('smith-notes', 'smith-notes'), (None, 'smith-notes'))
    assert validate(ENTRY_LOCAL, metadata, ads) == [
        "metadata key 'smith-notes' is referenced both with and without an ads_bibcode",
    ]
    assert ads.requests == []


def test_repo_bib_entries_are_valid_bibcodes() -> None:
    bib_text = (REPO_ROOT / 'sources' / 'citations.bib').read_bytes().decode('utf-8')
    entries = parse_bib_entries(bib_text)
    assert validate(bib_text, '', _FakeAds('\n'.join(entries.values()))) == []


def write_sources(tmp_path: Path, bib_text: str, metadata_text: str) -> list[str]:
    bib_path = tmp_path / 'citations.bib'
    metadata_path = tmp_path / 'citations-meta.yaml'
    bib_path.write_bytes(bib_text.encode())
    metadata_path.write_bytes(metadata_text.encode())
    return ['--bib', str(bib_path), '--metadata', str(metadata_path)]


def test_main_succeeds_when_entries_match_ads(
    tmp_path: Path,
    fake_connection: _FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('ADS_DEV_KEY', 'secret-token')
    fake_connection.response = _FakeResponse(200, json.dumps({'export': ADS_TEXT}).encode())

    assert main(write_sources(tmp_path, BIB_TEXT, '')) == 0

    assert fake_connection.instances[0].headers['Authorization'] == 'Bearer secret-token'
    assert 'citations validated' in capsys.readouterr().out


def test_main_preserves_line_endings_when_reading(
    tmp_path: Path,
    fake_connection: _FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv('ADS_DEV_KEY', 'secret-token')
    fake_connection.response = _FakeResponse(200, json.dumps({'export': ENTRY_A}).encode())

    assert main(write_sources(tmp_path, ENTRY_A.replace('\n', '\r\n'), '')) == 1

    assert 'differs from ADS' in capsys.readouterr().err


def test_main_reports_missing_token(
    tmp_path: Path,
    fake_connection: _FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv('ADS_DEV_KEY', raising=False)

    assert main(write_sources(tmp_path, BIB_TEXT, '')) == 1

    assert fake_connection.instances == []
    assert 'ADS_DEV_KEY is not set' in capsys.readouterr().err


def test_main_reports_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ['--bib', str(tmp_path / 'absent.bib'), '--metadata', str(tmp_path / 'absent.yaml')]
    assert main(arguments) == 1
    assert capsys.readouterr().err.startswith('error: FileNotFoundError')


def test_main_escapes_untrusted_values_in_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    metadata = json.dumps({'papers': {'x\n::add-mask::y': 'not-a-mapping'}})

    assert main(write_sources(tmp_path, '', metadata)) == 1

    assert all(line.startswith('error: ') for line in capsys.readouterr().err.splitlines())


def test_module_entry_point_exits_with_main_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'argv', ['validate_citations', *write_sources(tmp_path, '', '')])
    monkeypatch.delitem(sys.modules, 'scripts.validate_citations')

    with pytest.raises(SystemExit) as caught:
        runpy.run_module('scripts.validate_citations', run_name='__main__')

    assert caught.value.code == 0
