# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Numerical and compilation checks for the quadrupole QNM frequencies."""

from __future__ import annotations

import os
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Never, Protocol, overload

import h5py
import numpy as np
import pytest
from numba import njit, types
from numpy.testing import assert_allclose, assert_array_equal
from pyakima import cubic_call, cubic_call_scalar, make_akima_coeffs

from IMRPhenomWaveforms import qnm_spin
from IMRPhenomWaveforms.qnm_spin import f_damp, f_rd

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]

    class _Frequency(Protocol):
        @overload
        def __call__(self, spin: float, qnm_spin_type: str, /) -> float: ...
        @overload
        def __call__(self, spin: FloatArray, qnm_spin_type: str, /) -> FloatArray: ...
        def __call__(self, spin: float | FloatArray, qnm_spin_type: str, /) -> float | FloatArray: ...


def _table_values(spins: FloatArray, tables: qnm_spin._QNMInterpolationTables) -> list[FloatArray]:
    # The frequency splines via the public interpolation backend, and the
    # separation-constant splines directly, as the Leaver backend uses them.
    real, imag = qnm_spin._interpolated(spins, tables)
    return [
        real,
        imag,
        np.asarray(cubic_call(spins, tables.A_real, 4)),
        np.asarray(cubic_call(spins, tables.A_imag, 4)),
    ]


@pytest.fixture(scope='module')
def native_data() -> tuple[FloatArray, FloatArray]:
    resource = files('IMRPhenomWaveforms.data').joinpath('cook-qnm-v4-quadrupole.h5')
    with as_file(resource) as path, h5py.File(str(path), 'r') as data:
        positive = np.asarray(data['n00/m+02/{2,2,0}'], dtype=np.float64)
        negative = np.asarray(data['n00/m-02/{2,-2,0}'], dtype=np.float64)
    return positive, negative


@pytest.mark.parametrize('use_filename', [False, True])
def test_table_loading_outside_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, use_filename: bool) -> None:
    expected = qnm_spin._load_interp_qnm_tables()
    resource = files('IMRPhenomWaveforms.data').joinpath('cook-qnm-v4-quadrupole.h5')
    custom = tmp_path / 'custom.h5'
    custom.write_bytes(resource.read_bytes())
    decoy = tmp_path / 'IMRPhenomWaveforms' / 'data' / 'cook-qnm-v4-quadrupole.h5'
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b'Not an HDF5 file')
    monkeypatch.chdir(tmp_path)

    if use_filename:

        def unexpected_resource(_anchor: str) -> Never:
            msg = 'An explicit filename must not resolve a package resource'
            raise AssertionError(msg)

        monkeypatch.setattr(qnm_spin, 'files', unexpected_resource)

    actual = qnm_spin._load_interp_qnm_tables(str(custom) if use_filename else '')
    points = np.array([-0.7, 0.0, 0.7])
    for result, reference in zip(_table_values(points, actual), _table_values(points, expected), strict=True):
        assert_array_equal(result, reference)


def test_table_endpoints_already_present(tmp_path: Path, native_data: tuple[FloatArray, FloatArray]) -> None:
    # Rows at exactly |spin| = 1 must be used as given, not extrapolated or pinned.
    positive, negative = native_data
    positive_end = np.array([1.0, 0.99997, -8.0e-6, 0.5438, 3.0e-5])
    negative_end = np.array([1.0, 0.2915535, -0.0880258, 4.721727, -0.1982855])
    custom = tmp_path / 'endpoints.h5'
    with h5py.File(str(custom), 'w') as data:
        data['n00/m+02/{2,2,0}'] = np.vstack((positive, positive_end))
        data['n00/m-02/{2,-2,0}'] = np.vstack((negative, negative_end))
    actual = qnm_spin._load_interp_qnm_tables(str(custom))
    expected = qnm_spin._load_interp_qnm_tables()
    conversion = np.array([1 / (2 * np.pi), -1 / (2 * np.pi), 1.0, 1.0])
    interior = np.array([-0.7, 0.0, 0.7])
    endpoints = np.array([-1.0, 1.0])
    for result, reference in zip(_table_values(interior, actual), _table_values(interior, expected), strict=True):
        assert_array_equal(result, reference)
    results = _table_values(endpoints, actual)
    defaults = _table_values(endpoints, expected)
    for component in range(4):
        reference = conversion[component] * np.array([negative_end[component + 1], positive_end[component + 1]])
        assert_allclose(results[component], reference, rtol=1e-15, atol=0.0)
        assert not np.any(results[component] == defaults[component])


@pytest.mark.parametrize(('frequency', 'column', 'sign'), [(f_rd, 1, 1), (f_damp, 2, -1)])
def test_all_native_nodes(
    native_data: tuple[FloatArray, FloatArray],
    frequency: _Frequency,
    column: int,
    sign: int,
) -> None:
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    for data, orientation in zip(native_data, (1, -1), strict=True):
        actual = frequency(orientation * data[:, 0], 'InterpCook')
        assert_allclose(actual, sign * data[:, column] / (2 * np.pi), rtol=2e-14, atol=1e-18)


@pytest.mark.parametrize('method', ['InterpCook', 'LeaverCook'])
def test_schwarzschild_reference(method: str) -> None:
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=mirror-symmetry-and-tabulated-validation
    assert f_rd(0.0, method) == pytest.approx(0.747343 / (4 * np.pi), abs=4e-8)
    assert f_damp(0.0, method) == pytest.approx(0.177925 / (4 * np.pi), abs=4e-8)


@pytest.mark.parametrize('method', ['InterpCook', 'LeaverCook'])
def test_endpoints(native_data: tuple[FloatArray, FloatArray], method: str) -> None:
    # CITE paper=yang-2013-near-extremal-qnm-branches fact=matched-asymptotic-zero-damped-spectrum
    assert_allclose(f_rd(1.0, method), 1 / (2 * np.pi), atol=1.0e-20, rtol=1.0e-15)
    assert_allclose(f_damp(1.0, method), 0.0, atol=1.0e-20, rtol=1.0e-15)
    positive, negative = native_data
    grid = np.concatenate((-negative[:0:-1, 0], positive[:, 0]))
    for frequency, column, sign in [(f_rd, 1, 1), (f_damp, 2, -1)]:
        values = np.concatenate((negative[:0:-1, column], positive[:, column])) * sign / (2 * np.pi)
        expected = cubic_call_scalar(-1.0, make_akima_coeffs(grid, values), 0)
        assert frequency(-1.0, method) == expected
        # Independently of the extrapolation and pinning rules, both endpoints must continue
        # Cook's last rows, at |spin| = 1 - 1e-6 and 1 - 1e-9 respectively.
        assert frequency(-1.0, method) == pytest.approx(values[0], rel=1e-6)
        assert abs(frequency(1.0, method) - values[-1]) < 1e-5
        result = frequency(np.array([-1.0, 1.0]), method)
        assert_array_equal(result, [expected, frequency(1.0, method)])


@pytest.mark.parametrize('frequency', [f_rd, f_damp])
@pytest.mark.parametrize('method', ['InterpCook', 'LeaverCook'])
def test_invalid_spins_do_not_extrapolate(frequency: _Frequency, method: str) -> None:
    invalid = np.array([np.nextafter(-1.0, -2.0), np.nextafter(1.0, 2.0), -2, 2, np.nan, -np.inf, np.inf])
    assert np.all(np.isnan(frequency(invalid, method)))
    for spin in invalid:
        assert np.isnan(frequency(float(spin), method))


@pytest.mark.parametrize('frequency', [f_rd, f_damp])
@pytest.mark.parametrize('method', ['InterpCook', 'LeaverCook'])
def test_scalar_vector_and_strided_inputs(frequency: _Frequency, method: str) -> None:
    points = np.array([0.71, -0.31, 0.0, -0.94, 0.33, 0.71])[::-2]
    assert not points.flags.c_contiguous
    expected = [frequency(float(spin), method) for spin in points]
    assert_array_equal(frequency(points, method), expected)
    empty = frequency(np.array([], dtype=np.float64), method)
    assert empty.shape == (0,)
    assert empty.dtype == np.float64
    assert frequency(0, method) == frequency(0.0, method)


@pytest.mark.parametrize('frequency', [f_rd, f_damp])
@pytest.mark.parametrize('method', ['InterpCook', 'LeaverCook'])
def test_two_dimensional_inputs(frequency: _Frequency, method: str) -> None:
    points = np.array([[0.71, -0.31, 0.0], [-0.94, 0.33, np.nan]])
    expected = np.array([[frequency(float(spin), method) for spin in row] for row in points])
    assert np.isnan(expected[1, 2])
    assert not np.isnan(expected[:, :2]).any()
    assert not points.T.flags.c_contiguous
    for spins, reference in ((points, expected), (points.T, expected.T)):
        result = frequency(spins, method)
        assert result.shape == spins.shape
        assert result.dtype == np.float64
        assert_array_equal(result, reference)


@pytest.mark.parametrize('frequency', [f_rd, f_damp])
@pytest.mark.parametrize('method', ['InterpCook', 'LeaverCook'])
def test_complex_spin_is_rejected(frequency: _Frequency, method: str) -> None:
    with pytest.raises(TypeError, match='real scalar'):
        frequency(0.3 + 0j, method)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match='real scalar'):
        frequency(np.array([0.3 + 0j]), method)


def test_leaver_dispatch_without_frequency_selector() -> None:
    tables = qnm_spin._load_interp_qnm_tables()
    points = np.array([[0.71, -0.31], [0.0, np.nan]])
    # Last axis holds (frequency, damping) from the scalar backend.
    expected = np.array([[qnm_spin._leaver_scalar(float(spin), tables) for spin in row] for row in points])
    assert expected.shape == (*points.shape, 2)
    assert np.all(expected[..., 0] != expected[..., 1])
    assert qnm_spin._leaver(0.71, tables) == tuple(expected[0, 0])
    assert qnm_spin._leaver(0, tables) == tuple(expected[1, 0])
    for spins, reference in ((points[0], expected[0]), (points, expected), (points.T, expected.transpose(1, 0, 2))):
        result = qnm_spin._leaver(spins, tables)
        assert len(result) == 2
        for component in range(2):
            assert result[component].shape == spins.shape
            assert_array_equal(result[component], reference[..., component])
    with pytest.raises(TypeError, match='real scalar'):
        qnm_spin._leaver(0.3 + 0j, tables)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match='real scalar'):
        qnm_spin._leaver(np.array([0.3 + 0j]), tables)


def test_frequency_selector_requires_literal() -> None:
    select = qnm_spin._select_frequency
    for method in ('InterpCook', 'LeaverCook'):
        assert callable(select(types.float64, types.literal(method)))  # type: ignore[arg-type, no-untyped-call]
    with pytest.raises(NotImplementedError, match='Unrecognized qnm_spin_type'):
        select(types.float64, types.unicode_type)  # type: ignore[arg-type]


@pytest.mark.parametrize('frequency', [f_rd, f_damp])
def test_method_agreement_off_grid(frequency: _Frequency) -> None:
    points = np.linspace(-0.9973, 0.9871, 201)
    expected = frequency(points, 'LeaverCook')
    assert_allclose(frequency(points, 'InterpCook'), expected, rtol=3e-5, atol=2e-9)


@pytest.mark.skipif(os.environ.get('NUMBA_DISABLE_JIT') == '1', reason='Near-extremal truncations are very slow')
@pytest.mark.parametrize(('frequency', 'relative_tolerance'), [(f_rd, 2e-5), (f_damp, 6e-3)])
def test_method_agreement_near_extremal(frequency: _Frequency, relative_tolerance: float) -> None:
    points = np.concatenate((1 - np.geomspace(1e-9, 1e-3, 55), -1 + np.geomspace(1e-6, 1e-3, 12)))
    expected = frequency(points, 'LeaverCook')
    assert_allclose(frequency(points, 'InterpCook'), expected, rtol=relative_tolerance, atol=1e-10)


@pytest.mark.skipif(
    os.environ.get('NUMBA_DISABLE_JIT') == '1', reason='Includes near-extremal native rows requiring long fractions'
)
def test_leaver_matches_independent_native_seeds(native_data: tuple[FloatArray, FloatArray]) -> None:
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    for data, orientation in zip(native_data, (1, -1), strict=True):
        indices = np.unique(np.concatenate((np.linspace(0, len(data) - 1, 15, dtype=int), [len(data) - 1])))
        for spin, real, imaginary, _angular_real, _angular_imaginary in data[indices]:
            expected = complex(real, imaginary)
            result = (
                2
                * np.pi
                * complex(
                    f_rd(orientation * spin, 'LeaverCook'),
                    -f_damp(orientation * spin, 'LeaverCook'),
                )
            )
            assert result == pytest.approx(expected, abs=3e-9)


def test_leaver_refines_interpolated_frequency() -> None:
    assert abs(f_rd(0.9999, 'InterpCook') - f_rd(0.9999, 'LeaverCook')) > 5e-7
    assert abs(f_damp(0.9999, 'InterpCook') - f_damp(0.9999, 'LeaverCook')) > 1e-7


@pytest.mark.skipif(os.environ.get('NUMBA_DISABLE_JIT') == '1', reason='Tests the full near-extremal truncation limit')
def test_nonconvergence_is_reported() -> None:
    with pytest.raises(RuntimeError, match='did not converge'):
        f_rd(float(np.nextafter(1.0, 0.0)), 'LeaverCook')


@pytest.mark.parametrize('frequency', [f_rd, f_damp])
def test_unknown_selector(frequency: _Frequency) -> None:
    with pytest.raises(NotImplementedError, match='qnm_spin_type'):
        frequency(0.4, 'Unknown')
    with pytest.raises(NotImplementedError, match='qnm_spin_type'):
        frequency(np.array([0.1, 0.4]), 'Unknown')


@pytest.mark.skipif(
    os.environ.get('NUMBA_DISABLE_JIT') == '1', reason='Requires Numba compilation and literal signatures'
)
def test_jitted_literal_dispatch_and_reused_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    def forwarded(spin: float, method: str) -> tuple[float | FloatArray, float | FloatArray]:
        return f_rd(spin, method), f_damp(spin, method)

    def fixed(spin: FloatArray) -> tuple[FloatArray, FloatArray]:
        return f_rd(spin, 'InterpCook'), f_damp(spin, 'LeaverCook')

    compiled_forwarded = njit()(forwarded)
    compiled_fixed = njit()(fixed)
    points = np.array([-0.37, 0.0, 0.63])
    expected_fixed = compiled_fixed(points)
    expected_forwarded = {method: compiled_forwarded(0.63, method) for method in ('InterpCook', 'LeaverCook')}
    selectors = {signature[1].literal_value for signature in compiled_forwarded.signatures}  # type: ignore[attr-defined]
    assert selectors == {'InterpCook', 'LeaverCook'}
    assert all(isinstance(signature[1], types.StringLiteral) for signature in compiled_forwarded.signatures)
    assert len(compiled_forwarded.nopython_signatures) == 2

    def unexpected_setup(*_args: object, **_kwargs: object) -> Never:
        msg = 'Tables must not be rebuilt during evaluation'
        raise AssertionError(msg)

    monkeypatch.setattr(qnm_spin, '_load_interp_qnm_tables', unexpected_setup)
    monkeypatch.setattr(qnm_spin, 'make_akima_coeffs', unexpected_setup)
    monkeypatch.setattr(h5py, 'File', unexpected_setup)
    for _ in range(5):
        assert_array_equal(compiled_fixed(points), expected_fixed)
        for method, expected in expected_forwarded.items():
            assert compiled_forwarded(0.63, method) == expected


def test_recurrence_residuals(native_data: tuple[FloatArray, FloatArray]) -> None:
    angular = qnm_spin._angular
    radial = qnm_spin._radial
    newton = qnm_spin._newton
    solve = qnm_spin._leaver_root
    for data, orientation in zip(native_data, (1, -1), strict=True):
        row = data[np.argmin(abs(data[:, 0] - 0.7))]
        spin = orientation * row[0]
        w, separation = 2 * complex(row[1], row[2]), complex(row[3], row[4])
        assert abs(angular(w, separation, abs(spin) / 2, orientation * 2)) < 1e-12
        assert abs(radial(w, separation, abs(spin) / 2, orientation * 2, 1024)) < 1e-9
        result = newton(spin, w + 1e-4j, separation + 1e-4, 1024)
        assert result[2]
        assert result[0] == pytest.approx(w, abs=1e-10)
        assert result[1] == pytest.approx(separation, abs=1e-10)
        assert solve(spin, w, separation) == pytest.approx(w, abs=1e-10)
    assert angular(0.7 - 0.2j, 4 + 0j, 0.0, 2) == 0j
    row = native_data[0][-1]
    w, separation = 2 * complex(row[1], row[2]), complex(row[3], row[4])
    residual = angular(w, separation, row[0] / 2, 2, 128)
    assert abs(residual) < 1e-12
    assert angular(w, separation, row[0] / 2, 2) == pytest.approx(residual, abs=1e-14)


def _native_seed(native_data: tuple[FloatArray, FloatArray], spin: float) -> tuple[float, complex, complex]:
    data = native_data[0] if spin >= 0 else native_data[1]
    row = data[np.argmin(abs(data[:, 0] - abs(spin)))]
    return float(np.copysign(row[0], spin)), 2 * complex(row[1], row[2]), complex(row[3], row[4])


def test_newton_reports_iteration_limit(native_data: tuple[FloatArray, FloatArray]) -> None:
    spin, w, separation = _native_seed(native_data, 0.7)
    seed, separation_seed = w + 1e-3 * abs(w.imag) * (1 + 1j), separation + 1e-3
    assert qnm_spin._newton(spin, seed, separation_seed, 1024)[2]
    assert not qnm_spin._newton(spin, seed, separation_seed, 1024, 1)[2]


@pytest.mark.parametrize(
    ('depth', 'converges'),
    [
        (128, False),
        pytest.param(
            65536,
            True,
            marks=pytest.mark.skipif(
                os.environ.get('NUMBA_DISABLE_JIT') == '1', reason='Near-extremal truncations are very slow'
            ),
        ),
    ],
)
def test_newton_shallow_truncation_fails_near_extremal(
    native_data: tuple[FloatArray, FloatArray], depth: int, *, converges: bool
) -> None:
    # Seeded exactly at Cook's retrograde root, so only the truncation depth decides the outcome.
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    row = native_data[1][-1]
    spin, w, separation = -row[0], 2 * complex(row[1], row[2]), complex(row[3], row[4])
    assert spin < -0.99999
    result = qnm_spin._newton(spin, w, separation, depth)
    assert result[2] == converges
    if converges:
        assert result[0] == pytest.approx(w, abs=1e-10)
    else:
        assert abs(result[0] - w) > 0.01


@pytest.mark.parametrize('target', [0.0, 0.7, -0.7])
def test_leaver_root_single_truncation_always_raises(native_data: tuple[FloatArray, FloatArray], target: float) -> None:
    # With max_depth equal to the initial depth there is never a second truncation to agree with.
    spin, w, separation = _native_seed(native_data, target)
    converged = qnm_spin._newton(spin, w, separation, 128)
    assert converged[2]
    assert converged[0] == pytest.approx(w, abs=1e-10)
    with pytest.raises(RuntimeError, match='did not converge'):
        qnm_spin._leaver_root(spin, w, separation, 128)


@pytest.mark.parametrize(
    ('target', 'newton_converged', 'succeeds'),
    [
        (0.7, (True, True), True),
        (-0.7, (True, True), True),
        # Depths 128 and 256 both converge, but to roots about 3e-8 apart; depth 512 confirms 256.
        (0.99875, (True, True), False),
        (0.99875, (True, True, True), True),
        # Only depth 256 converges, to a truncation-biased root about 0.07 from Cook's.
        (-0.9999375, (False, True), False),
    ],
)
def test_leaver_root_requires_agreeing_truncations(
    native_data: tuple[FloatArray, FloatArray],
    target: float,
    newton_converged: tuple[bool, ...],
    *,
    succeeds: bool,
) -> None:
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    spin, w, separation = _native_seed(native_data, target)
    depths = [128 * 2**k for k in range(len(newton_converged))]
    roots = [qnm_spin._newton(spin, w, separation, depth) for depth in depths]
    assert tuple(root[2] for root in roots) == newton_converged
    if succeeds:
        assert qnm_spin._leaver_root(spin, w, separation, depths[-1]) == roots[-1][0]
        assert roots[-1][0] == pytest.approx(w, abs=3e-9)
        return
    if all(newton_converged[-2:]):
        assert abs(roots[-1][0] - roots[-2][0]) > 1e-8
    with pytest.raises(RuntimeError, match='did not converge'):
        qnm_spin._leaver_root(spin, w, separation, depths[-1])


def _scripted_newton(monkeypatch: pytest.MonkeyPatch, script: dict[int, tuple[complex, complex, bool]]) -> list[int]:
    depths: list[int] = []

    def scripted(_spin: float, _seed: complex, _separation_seed: complex, depth: int) -> tuple[complex, complex, bool]:
        depths.append(depth)
        return script[depth]

    monkeypatch.setattr(qnm_spin, '_newton', scripted)
    return depths


def test_leaver_root_forgets_roots_before_nonconvergence(monkeypatch: pytest.MonkeyPatch) -> None:
    solve = getattr(qnm_spin._leaver_root, 'py_func', qnm_spin._leaver_root)
    w, separation = 1.0 - 0.2j, 3.0 + 0.1j
    depths = _scripted_newton(
        monkeypatch,
        {
            128: (w, separation, True),
            256: (5.0 - 5.0j, 5.0 + 5.0j, False),
            512: (w, separation, True),
            1024: (w + 1e-11, separation, True),
        },
    )
    # The depth-512 root matches depth 128, but a failed truncation lies between them.
    assert solve(0.7, w, separation, 1024) == w + 1e-11
    assert depths == [128, 256, 512, 1024]


def test_leaver_root_requires_separation_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    solve = getattr(qnm_spin._leaver_root, 'py_func', qnm_spin._leaver_root)
    w, separation = 1.0 - 0.2j, 3.0 + 0.1j
    depths = _scripted_newton(
        monkeypatch,
        {
            128: (w, separation, True),
            256: (w, separation + 1e-6, True),
            512: (w + 1e-11, separation + 1e-6, True),
        },
    )
    assert solve(0.7, w, separation, 512) == w + 1e-11
    assert depths == [128, 256, 512]


@pytest.mark.parametrize(('target', 'orientation'), [(0.3, 1), (0.7, 1), (0.99, 1), (0.3, -1), (0.7, -1), (0.99, -1)])
def test_newton_spin_sign_selects_branch(
    native_data: tuple[FloatArray, FloatArray], target: float, orientation: int
) -> None:
    # Cook's m = 2 (prograde) root is not a root of the m = -2 equations at the same |a|, and vice versa.
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    spin, w, separation = _native_seed(native_data, orientation * target)
    a, m = abs(spin) / 2, orientation * 2
    assert abs(qnm_spin._radial(w, separation, a, m, 1024)) < 1e-9
    assert abs(qnm_spin._angular(w, separation, a, m)) < 1e-9
    assert abs(qnm_spin._radial(w, separation, a, -m, 1024)) > 1.0
    assert abs(qnm_spin._angular(w, separation, a, -m)) > 0.1
    correct = qnm_spin._newton(spin, w, separation, 1024)
    assert correct[2]
    assert correct[0] == pytest.approx(w, abs=1e-10)
    wrong = qnm_spin._newton(-spin, w, separation, 1024)
    assert not wrong[2]
    assert abs(wrong[0] - w) > 0.05
    with pytest.raises(RuntimeError, match='did not converge'):
        qnm_spin._leaver_root(-spin, w, separation, 1024)


@pytest.mark.parametrize('seed', [1.0 + 0.2j, 1.0 + 0.0j])
def test_newton_rejects_nonnegative_imaginary_seed(seed: complex) -> None:
    assert qnm_spin._newton(0.7, seed, 3.0 + 0.1j, 128) == (seed, 3.0 + 0.1j, False)
    with pytest.raises(RuntimeError, match='did not converge'):
        qnm_spin._leaver_root(0.7, seed, 3.0 + 0.1j, 256)


def test_newton_rejects_nonfinite_jacobian() -> None:
    # The NaN seed makes every finite difference NaN; the iterate must be returned without an update.
    seed = complex(np.nan, -0.1)
    with np.errstate(invalid='ignore'):
        w, separation, converged = qnm_spin._newton(0.5, seed, 4 + 0j, 128)
        with pytest.raises(RuntimeError, match='did not converge'):
            qnm_spin._leaver_root(0.5, seed, 4 + 0j, 256)
    assert not converged
    assert np.isnan(w.real)
    assert w.imag == -0.1
    assert separation == 4 + 0j


def _linear_fractions(
    monkeypatch: pytest.MonkeyPatch, w_root: complex, separation_root: complex
) -> tuple[Callable[[float, complex, complex, int], tuple[complex, complex, bool]], list[complex]]:
    # Replace the fractions by w - w_root and A - A_root, so one Newton step lands on the root.
    newton = getattr(qnm_spin._newton, 'py_func', qnm_spin._newton)
    evaluated: list[complex] = []

    def radial(w: complex, _separation: complex, _a: float, _m: int, _depth: int) -> complex:
        evaluated.append(w)
        return w - w_root

    def angular(_w: complex, separation: complex, _a: float, _m: int) -> complex:
        return separation - separation_root

    monkeypatch.setattr(qnm_spin, '_radial', radial)
    monkeypatch.setattr(qnm_spin, '_angular', angular)
    return newton, evaluated


@pytest.mark.parametrize(('root_real', 'expected'), [(0.02, True), (-0.01, False)])
def test_newton_rejects_nonpositive_real_part(
    monkeypatch: pytest.MonkeyPatch, root_real: float, *, expected: bool
) -> None:
    # Both roots lie inside the trust radius 0.5 |Im seed| = 0.5; only the sign of Re w differs.
    newton, _ = _linear_fractions(monkeypatch, complex(root_real, -1.0), 3.0 + 0j)
    w, separation, converged = newton(0.7, 0.01 - 1.0j, 3.1 + 0j, 128)
    assert converged == expected
    assert w == pytest.approx(complex(root_real, -1.0), abs=1e-12)
    assert separation == pytest.approx(3.0, abs=1e-12)


def test_newton_stops_at_nonfinite_iterate(monkeypatch: pytest.MonkeyPatch) -> None:
    seed, separation_seed = 1.0 - 0.2j, 3.0 + 0j
    newton, evaluated = _linear_fractions(monkeypatch, 1.0 - 0.21j, 3.0 + 0j)
    linear_radial = qnm_spin._radial

    def radial(w: complex, separation: complex, a: float, m: int, depth: int) -> complex:
        # The residual is NaN only at the seed itself, so the Jacobian stays finite.
        if not np.isfinite(abs(w)):
            msg = 'Fractions must not be evaluated at a non-finite iterate'
            raise AssertionError(msg)
        value = linear_radial(w, separation, a, m, depth)
        return complex(np.nan, np.nan) if (w, separation) == (seed, separation_seed) else value

    monkeypatch.setattr(qnm_spin, '_radial', radial)
    w, _separation, converged = newton(0.7, seed, separation_seed, 128)
    assert not converged
    assert np.isnan(abs(w))
    assert len(evaluated) == 5


@pytest.mark.parametrize(
    'target',
    [
        0.0,
        0.7,
        -0.7,
        0.99,
        -0.99,
        pytest.param(
            0.999999,
            marks=pytest.mark.skipif(
                os.environ.get('NUMBA_DISABLE_JIT') == '1', reason='Near-extremal truncations are very slow'
            ),
        ),
    ],
)
def test_leaver_recovers_perturbed_native_roots(native_data: tuple[FloatArray, FloatArray], target: float) -> None:
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=units-and-radiative-boundary-conditions
    spin, w, separation = _native_seed(native_data, target)
    seed = w + 0.2 * abs(w.imag) * (1 + 1j)
    separation_seed = separation + 0.01 * (1 - 1j)
    tolerance = 3e-9
    # The initial frequency is deliberately outside the reference tolerance.
    assert abs(seed - w) > 100 * tolerance
    result = qnm_spin._leaver_root(spin, seed, separation_seed)
    assert result == pytest.approx(w, abs=tolerance)


@pytest.mark.parametrize('target', [0.0, 0.7, -0.7, 0.999999])
def test_newton_quadratic_convergence(native_data: tuple[FloatArray, FloatArray], target: float) -> None:
    newton = qnm_spin._newton
    spin, w, separation = _native_seed(native_data, target)
    root, root_separation, converged = newton(spin, w, separation, 1024)
    assert converged
    seed, separation_seed = w + 1e-3 * abs(w.imag) * (1 + 1j), separation + 1e-3
    errors = []
    for iterations in range(1, 4):
        iterate, iterate_separation, done = newton(spin, seed, separation_seed, 1024, iterations)
        assert done == (iterations == 3)
        errors.append(max(abs(iterate - root), abs(iterate_separation - root_separation)))
    assert errors[1] <= 10 * errors[0] ** 2
    assert errors[2] <= 1e-14
    restarted = newton(spin, root, root_separation, 1024, 1)
    assert restarted[2]
    assert restarted[0] == pytest.approx(root, rel=1e-13)
    assert restarted[1] == pytest.approx(root_separation, rel=1e-13)


@pytest.mark.parametrize(('offset', 'expected'), [(0.3, True), (0.6, False)])
@pytest.mark.parametrize('target', [0.0, 0.7, -0.7, 0.999999])
def test_newton_trust_radius_scales_with_damping(
    native_data: tuple[FloatArray, FloatArray], target: float, offset: float, *, expected: bool
) -> None:
    newton = qnm_spin._newton
    spin, w, separation = _native_seed(native_data, target)
    root = newton(spin, w, separation, 1024)[0]
    result = newton(spin, w + offset * abs(w.imag), separation, 1024)
    assert result[2] == expected
    assert (result[0] == pytest.approx(root, rel=1e-13)) == expected


def test_newton_rejects_degenerate_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    newton = getattr(qnm_spin._newton, 'py_func', qnm_spin._newton)
    angular = qnm_spin._angular

    def proportional(w: complex, separation: complex, a: float, m: int, _depth: int) -> complex:
        return 7.3 * angular(w, separation, a, m)

    monkeypatch.setattr(qnm_spin, '_radial', proportional)
    assert newton(0.7, 1.0 - 0.17j, 3.0 + 0.1j, 128) == (1.0 - 0.17j, 3.0 + 0.1j, False)
