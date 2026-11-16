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
from pyakima import cubic_call_scalar, make_akima_coeffs

from IMRPhenomWaveforms import qnm_spin
from IMRPhenomWaveforms.qnm_spin import f_damp, f_rd

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]

    class _Frequency(Protocol):
        @overload
        def __call__(self, spin: float, qnm_spin_type: str, /) -> float: ...
        @overload
        def __call__(self, spin: FloatArray, qnm_spin_type: str, /) -> FloatArray: ...
        def __call__(self, spin: float | FloatArray, qnm_spin_type: str, /) -> float | FloatArray: ...


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
    for component in range(4):
        assert_array_equal(
            qnm_spin._interpolated(points, actual, component),
            qnm_spin._interpolated(points, expected, component),
        )


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
    for component in range(4):
        assert_array_equal(
            qnm_spin._interpolated(interior, actual, component),
            qnm_spin._interpolated(interior, expected, component),
        )
        result = qnm_spin._interpolated(endpoints, actual, component)
        reference = conversion[component] * np.array([negative_end[component + 1], positive_end[component + 1]])
        assert_allclose(result, reference, rtol=1e-15, atol=0.0)
        assert not np.any(result == qnm_spin._interpolated(endpoints, expected, component))


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


@pytest.mark.parametrize('component', [0, 1])
def test_leaver_dispatch_without_frequency_selector(component: int) -> None:
    tables = qnm_spin._load_interp_qnm_tables()
    points = np.array([[0.71, -0.31], [0.0, np.nan]])
    expected = np.array([[qnm_spin._leaver_scalar(float(spin), tables, component) for spin in row] for row in points])
    assert qnm_spin._leaver(0.71, tables, component) == expected[0, 0]
    assert qnm_spin._leaver(0, tables, component) == expected[1, 0]
    assert_array_equal(qnm_spin._leaver(points[0], tables, component), expected[0])
    assert_array_equal(qnm_spin._leaver(points, tables, component), expected)
    with pytest.raises(TypeError, match='real scalar'):
        qnm_spin._leaver(0.3 + 0j, tables, component)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match='real scalar'):
        qnm_spin._leaver(np.array([0.3 + 0j]), tables, component)


def test_frequency_selector_requires_literals() -> None:
    select = qnm_spin._select_frequency
    assert callable(select(types.float64, types.literal('InterpCook'), types.literal(0)))  # type: ignore[arg-type, no-untyped-call]
    with pytest.raises(NotImplementedError, match='Unrecognized qnm_spin_type'):
        select(types.float64, types.unicode_type, types.literal(0))  # type: ignore[arg-type, no-untyped-call]
    with pytest.raises(NotImplementedError, match='Unrecognized qnm_spin_type'):
        select(types.float64, types.literal('InterpCook'), types.int64)  # type: ignore[arg-type, no-untyped-call]


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
    limited = newton(row[0], w + 1e-7j, separation + 1e-4, 1024, 1)
    assert not limited[2]
    row = native_data[1][-1]
    w, separation = 2 * complex(row[1], row[2]), complex(row[3], row[4])
    wrong_branch = newton(-row[0], w, separation, 128)
    assert not wrong_branch[2]
    assert abs(wrong_branch[0] - w) > 0.01
    with pytest.raises(RuntimeError, match='did not converge'):
        solve(-row[0], w, separation, 128)
    with np.errstate(invalid='ignore'):
        invalid = newton(0.5, complex(np.nan, 0), 4 + 0j, 128)
        assert not invalid[2]
        with pytest.raises(RuntimeError, match='did not converge'):
            solve(0.5, complex(np.nan, 0), 4 + 0j, 128)


def _native_seed(native_data: tuple[FloatArray, FloatArray], spin: float) -> tuple[float, complex, complex]:
    data = native_data[0] if spin >= 0 else native_data[1]
    row = data[np.argmin(abs(data[:, 0] - abs(spin)))]
    return float(np.copysign(row[0], spin)), 2 * complex(row[1], row[2]), complex(row[3], row[4])


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
    assert abs(result - seed) > 0.1 * abs(seed - w)


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
