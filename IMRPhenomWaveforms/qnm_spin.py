# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Fundamental quadrupole Kerr frequencies in inverse remnant-mass units."""

from __future__ import annotations

from contextlib import nullcontext
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, overload

import h5py
import numba.extending
import numpy as np
from numba import njit, typeof, types
from numba.misc.special import literally
from pyakima import SplineCoeffs, cubic_call, make_akima_coeffs

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


class _QNMInterpolationTables(NamedTuple):
    f_real: SplineCoeffs
    f_imag: SplineCoeffs
    A_real: SplineCoeffs
    A_imag: SplineCoeffs


def _load_interp_qnm_tables(filename: str = '') -> _QNMInterpolationTables:
    source = (
        nullcontext(Path(filename))
        if filename
        else as_file(files('IMRPhenomWaveforms.data').joinpath('cook-qnm-v4-quadrupole.h5'))
    )
    # Load Cook's data and prepare immutable, endpoint-pinned splines once.
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    with source as path, h5py.File(str(path), 'r') as data:
        positive = np.asarray(data['n00/m+02/{2,2,0}'], dtype=np.float64)
        negative = np.asarray(data['n00/m-02/{2,-2,0}'], dtype=np.float64)
    grid = np.concatenate((-negative[:0:-1, 0], positive[:, 0]))
    values = np.concatenate((negative[:0:-1, 1:], positive[:, 1:]))
    # Convert Cook's M omega (M = 1, e^{-i omega t}, Im omega < 0) to M f_rd and M f_damp.
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=teukolsky-and-boundary-conventions
    values[:, 0] /= 2 * np.pi
    values[:, 1] /= -2 * np.pi
    splines = []
    for column in range(4):
        original = make_akima_coeffs(grid, np.ascontiguousarray(values[:, column]))
        # Extrapolate to the endpoints if they are not included
        x_grid = np.ascontiguousarray(grid[np.abs(grid) <= 1.0])
        y_grid = np.ascontiguousarray(values[np.abs(grid) <= 1.0, column])

        if x_grid[0] > -1.0:
            low = cubic_call(-1.0, original, 0)
            x_grid = np.concatenate(([-1.0], x_grid))
            y_grid = np.concatenate(([low], y_grid))
        if x_grid[-1] < 1.0:
            high = cubic_call(1.0, original, 0)
            # CITE paper=yang-2013-near-extremal-qnm-branches fact=matched-asymptotic-zero-damped-spectrum
            if column == 0:
                high = 1 / (2 * np.pi)
            elif column == 1:
                high = 0.0
            x_grid = np.concatenate((x_grid, [1.0]))
            y_grid = np.concatenate((y_grid, [high]))
        spline = make_akima_coeffs(x_grid, y_grid)
        splines.append(spline)
    return _QNMInterpolationTables(splines[0], splines[1], splines[2], splines[3])


@njit()
def _angular(w: complex, separation: complex, a: float, m: int, depth: int = 64) -> complex:
    # Evaluate Leaver's uninverted angular fraction for spin weight -2. At a = 0
    # beta_0 vanishes for l = k1 + k2 = 2, so the uninverted form selects l = 2.
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=kerr-angular-series-and-coefficients
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=continued-fraction-root-and-inversions
    spheroidicity = a * w
    k1 = abs(m + 2) / 2
    k2 = abs(m - 2) / 2
    tail = 0j
    for n in range(depth, -1, -1):
        alpha = -2 * (n + 1) * (n + 2 * k1 + 1)
        beta = (
            n * (n - 1)
            + 2 * n * (k1 + k2 + 1 - 2 * spheroidicity)
            - 2 * spheroidicity * (2 * k1 - 1)
            + (k1 + k2) * (k1 + k2 + 1)
            - spheroidicity**2
            - 2
            - separation
        )
        gamma_next = 2 * spheroidicity * (n + k1 + k2 - 1)
        if n == depth:
            tail = beta
        else:
            tail = beta - alpha * gamma_next / tail
    return tail


@njit()
def _radial(w: complex, separation: complex, a: float, m: int, depth: int) -> complex:
    # Evaluate Leaver's uninverted radial fraction for spin weight -2 with a
    # finite truncation; the fundamental is validated against Cook's data.
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=kerr-radial-series-and-coupled-root
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=continued-fraction-root-and-inversions
    b = np.sqrt((1 - 2 * a) * (1 + 2 * a))
    horizon = (w / 2 - a * m) / b
    c0 = 3 - 1j * w - 2j * horizon
    c1 = -4 + 2j * w * (2 + b) + 4j * horizon
    c2 = 1 - 3j * w - 2j * horizon
    c3 = w**2 * (4 + 2 * b - a**2) - 2 * a * m * w + 1 + (2 + b) * 1j * w
    c3 += -separation + (4 * w + 2j) * horizon
    c4 = -1 - 2 * w**2 + 1j * w - (4 * w + 2j) * horizon
    tail = 0j
    for n in range(depth, -1, -1):
        alpha = n**2 + (c0 + 1) * n + c0
        beta = -2 * n**2 + (c1 + 2) * n + c3
        gamma_next = (n + 1) ** 2 + (c2 - 3) * (n + 1) + c4 - c2 + 2
        if n == depth:
            tail = beta
        else:
            tail = beta - alpha * gamma_next / tail
    return tail


@njit()
def _newton(
    spin: float,
    seed: complex,
    separation_seed: complex,
    depth: int,
    max_iterations: int = 20,
) -> tuple[complex, complex, bool]:
    # Solve the radial and angular fractions simultaneously for (omega, A),
    # without leaving the seeded branch.
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=kerr-radial-series-and-coupled-root
    # Leaver's units are 2M = 1 with 0 <= a < 1/2.
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=units-and-radiative-boundary-conditions
    a = abs(spin) / 2
    # Negative spin maps to the positive-frequency m = -2 branch at |a|.
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    m = 2 if spin >= 0 else -2
    if seed.imag >= 0:
        return seed, separation_seed, False
    dtype = np.asarray(seed).dtype
    eps = np.finfo(dtype).eps
    relative_step = np.cbrt(eps)
    step_w = relative_step * abs(seed.imag)
    w, separation = seed, separation_seed
    for _ in range(max_iterations):
        step_a = relative_step * abs(separation)
        radial = _radial(w, separation, a, m, depth)
        angular = _angular(w, separation, a, m)
        dw_radial = (_radial(w + step_w, separation, a, m, depth) - _radial(w - step_w, separation, a, m, depth)) / (
            2 * step_w
        )
        da_radial = (_radial(w, separation + step_a, a, m, depth) - _radial(w, separation - step_a, a, m, depth)) / (
            2 * step_a
        )
        dw_angular = (_angular(w + step_w, separation, a, m) - _angular(w - step_w, separation, a, m)) / (2 * step_w)
        da_angular = (_angular(w, separation + step_a, a, m) - _angular(w, separation - step_a, a, m)) / (2 * step_a)
        determinant = dw_radial * da_angular - da_radial * dw_angular
        cancellation = abs(dw_radial * da_angular) + abs(da_radial * dw_angular)
        if not np.isfinite(cancellation) or abs(determinant) <= relative_step**2 * cancellation:
            return w, separation, False
        correction_w = (da_angular * radial - da_radial * angular) / determinant
        correction_a = (dw_radial * angular - dw_angular * radial) / determinant
        w -= correction_w
        separation -= correction_a
        if not np.isfinite(abs(w)) or abs(w - seed) > 0.5 * abs(seed.imag) or w.imag >= 0 or w.real <= 0:
            return w, separation, False
        if max(abs(correction_w), abs(correction_a)) <= np.sqrt(eps) * max(abs(w), abs(separation)):
            return w, separation, True
    return w, separation, False


@njit()
def _leaver_root(
    spin: float,
    seed: complex,
    separation_seed: complex,
    max_depth: int = 1048576,
) -> complex:
    # Require agreement between successive continued-fraction truncations; this
    # depth-doubling test is a numerical convergence heuristic, not a source result.
    previous = np.inf + 1j * 0
    previous_separation = np.inf + 1j * 0
    depth = 128
    while depth <= max_depth:
        w, separation, converged = _newton(spin, seed, separation_seed, depth)
        if converged:
            if max(abs(w - previous), abs(separation - previous_separation)) < 5e-10:
                return w
            previous, previous_separation = w, separation
        else:
            previous = np.inf + 1j * 0
        depth *= 2
    msg = 'Leaver continued fractions did not converge at this spin'
    raise RuntimeError(msg)


@overload
def _interpolated(spin: float, tables: _QNMInterpolationTables, component: int) -> float: ...
@overload
def _interpolated(spin: FloatArray, tables: _QNMInterpolationTables, component: int) -> FloatArray: ...
@njit()
def _interpolated(spin: float | FloatArray, tables: _QNMInterpolationTables, component: int) -> float | FloatArray:
    spline = tables[component]
    return cubic_call(spin, spline, 4)


@njit()
def _leaver_scalar(spin: float, tables: _QNMInterpolationTables, component: int) -> float:
    if not np.isfinite(spin) or spin <= -1 or spin >= 1:
        return _interpolated(spin, tables, component)
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=units-and-radiative-boundary-conditions
    seed = 4 * np.pi * (_interpolated(spin, tables, 0) - 1j * _interpolated(spin, tables, 1))
    # Cook's A (his Eq. 4) equals Leaver's A_lm (Eq. 16), as both use c = a omega.
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=kerr-angular-series-and-coefficients
    separation = cubic_call(spin, tables[2], 4) + 1j * cubic_call(spin, tables[3], 4)
    w = _leaver_root(spin, seed, separation)
    if component == 0:
        return w.real / (4 * np.pi)
    return -w.imag / (4 * np.pi)


@njit()
def _leaver_vector(spins: FloatArray, tables: _QNMInterpolationTables, component: int) -> FloatArray:
    if spins.ndim == 1:
        return np.array([_leaver_scalar(float(spin), tables, component) for spin in spins])
    flat_spins = spins.ravel()
    return np.array([_leaver_scalar(float(spin), tables, component) for spin in flat_spins]).reshape(spins.shape)


@overload
def _leaver(spin: float, tables: _QNMInterpolationTables, component: int) -> float: ...
@overload
def _leaver(spin: FloatArray, tables: _QNMInterpolationTables, component: int) -> FloatArray: ...
def _leaver(spin: float | FloatArray, tables: _QNMInterpolationTables, component: int) -> float | FloatArray:
    implementation = _select_leaver(typeof(spin), typeof(tables), types.literal(component))  # type: ignore[no-untyped-call]
    return implementation(spin, tables, component)  # type: ignore[no-any-return]


@numba.extending.overload(_leaver, prefer_literal=True, strict=False)
def _select_leaver[SpinType: float | FloatArray, TableType: _QNMInterpolationTables, ComponentType: int](
    spin: type[SpinType], _tables: type[TableType], _component: type[ComponentType]
) -> Callable[[SpinType, TableType, ComponentType], SpinType]:
    if isinstance(spin, (types.Float, types.Integer)):

        def _leaver_loc_scalar(spin: float, tables: TableType, component: ComponentType) -> float:
            return _leaver_scalar(spin, tables, component)

        return _leaver_loc_scalar  # type: ignore[bad-return, unused-ignore]
    if isinstance(spin, types.Array) and isinstance(spin.dtype, (types.Float, types.Integer)):

        def _leaver_loc_vector(spin: FloatArray, tables: TableType, component: ComponentType) -> FloatArray:
            return _leaver_vector(spin, tables, component)

        return _leaver_loc_vector  # type: ignore[bad-return, unused-ignore]
    msg = 'spin must be a real scalar or one-dimensional real array'
    raise TypeError(msg)


@overload
def _frequency(spin: float, qnm_spin_type: str, component: int) -> float: ...
@overload
def _frequency(spin: FloatArray, qnm_spin_type: str, component: int) -> FloatArray: ...
def _frequency(spin: float | FloatArray, qnm_spin_type: str, component: int) -> float | FloatArray:
    implementation = _select_frequency(typeof(spin), types.literal(qnm_spin_type), types.literal(component))  # type: ignore[no-untyped-call]
    return implementation(spin, qnm_spin_type, component)  # type: ignore[no-any-return]


@numba.extending.overload(_frequency, prefer_literal=True, strict=False)
def _select_frequency[SpinType: float | FloatArray, QNMType: str, ComponentType: int](
    spin: type[SpinType], qnm_spin_type: type[QNMType], component: type[ComponentType]
) -> Callable[[SpinType, QNMType, ComponentType], SpinType]:
    if not isinstance(qnm_spin_type, types.StringLiteral) or not isinstance(component, types.IntegerLiteral):
        msg = 'Unrecognized qnm_spin_type'
        raise NotImplementedError(msg)
    name = qnm_spin_type.literal_value

    tables = _load_interp_qnm_tables()
    index = component.literal_value
    if name == 'InterpCook':
        backend = _interpolated
    elif name == 'LeaverCook':
        backend = _leaver
    else:
        msg = 'Unrecognized qnm_spin_type'
        raise NotImplementedError(msg)

    if isinstance(spin, (types.Float, types.Integer)):

        def _frequency_loc_scalar(spin: float, _qnm_spin_type: QNMType, _component: ComponentType) -> float:
            return backend(spin, tables, index)

        return _frequency_loc_scalar  # type: ignore[bad-return, unused-ignore]
    if isinstance(spin, types.Array) and isinstance(spin.dtype, (types.Float, types.Integer)):

        def _frequency_loc_vector(spin: FloatArray, _qnm_spin_type: QNMType, _component: ComponentType) -> FloatArray:
            return backend(spin, tables, index)

        return _frequency_loc_vector  # type: ignore[bad-return, unused-ignore]

    msg = 'spin must be a real scalar or one-dimensional real array'
    raise TypeError(msg)


@overload
def f_rd(spin: float, qnm_spin_type: str) -> float: ...
@overload
def f_rd(spin: FloatArray, qnm_spin_type: str) -> FloatArray: ...
@njit()
def f_rd(spin: float | FloatArray, qnm_spin_type: str) -> float | FloatArray:
    """Return the fundamental quadrupole ringdown frequency, M f_rd.

    Parameters
    ----------
    spin : float | FloatArray
        Signed remnant spin, scalar or one-dimensional array.
    qnm_spin_type : str
        Literal 'InterpCook' or 'LeaverCook'.

    Returns
    -------
    float | FloatArray
        Frequency; NaN for nonfinite spins or spins outside [-1, 1].

    Notes
    -----
    Unknown selectors raise NotImplementedError. Leaver nonconvergence raises
    RuntimeError; this can occur very close to extremality. Exact endpoints use
    pinned values, with the negative endpoint extrapolated once.
    """
    return _frequency(spin, literally(qnm_spin_type), 0)


@overload
def f_damp(spin: float, qnm_spin_type: str) -> float: ...
@overload
def f_damp(spin: FloatArray, qnm_spin_type: str) -> FloatArray: ...
@njit()
def f_damp(spin: float | FloatArray, qnm_spin_type: str) -> float | FloatArray:
    """Return the fundamental quadrupole damping frequency, M/(2 pi tau).

    Parameters
    ----------
    spin : float | FloatArray
        Signed remnant spin, scalar or one-dimensional array.
    qnm_spin_type : str
        Literal 'InterpCook' or 'LeaverCook'.

    Returns
    -------
    float | FloatArray
        Positive damping frequency; NaN outside the finite spin domain [-1, 1].

    Notes
    -----
    Unknown selectors raise NotImplementedError. Leaver nonconvergence raises
    RuntimeError; this can occur very close to extremality. Exact endpoints use
    pinned values, with the negative endpoint extrapolated once.
    """
    return _frequency(spin, literally(qnm_spin_type), 1)
