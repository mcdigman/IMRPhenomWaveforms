# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Agreement checks between MINPACK HYBRD and the Newton Leaver solver."""

from __future__ import annotations

import os
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, NamedTuple

import h5py
import numpy as np
import pytest
from numba import cfunc, njit
from numba.np.numpy_support import carray

from IMRPhenomWaveforms import qnm_spin

if os.environ.get('NUMBA_DISABLE_JIT') == '1':
    pytest.skip('HYBRD callbacks require Numba compilation', allow_module_level=True)

minpack_numba = pytest.importorskip('minpack_numba')

if TYPE_CHECKING:
    from ctypes import _Pointer, c_double, c_int32, c_void_p

    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


class _HybrdResult(NamedTuple):
    """HYBRD solution and termination diagnostics."""

    frequency: complex
    separation: complex
    info: int
    residual: float
    evaluations: int


@njit()
def _fill_residual(x: FloatArray, fvec: FloatArray, parameters: FloatArray) -> None:
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=units-and-radiative-boundary-conditions
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    spin = parameters[0]
    a = abs(spin) / 2
    m = 2 if spin >= 0 else -2
    w = complex(x[0], x[1])
    separation = complex(x[2], x[3])
    radial = qnm_spin._radial(w, separation, a, m, int(parameters[1]))
    angular = qnm_spin._angular(w, separation, a, m)
    fvec[0] = radial.real
    fvec[1] = radial.imag
    fvec[2] = angular.real
    fvec[3] = angular.imag


@cfunc(minpack_numba.hybrd_sig)
def _residual(
    _n: int, x: _Pointer[c_double], fvec: _Pointer[c_double], _iflag: _Pointer[c_int32], udata: c_void_p
) -> None:
    # Solve the equations without Newton's branch restrictions.
    parameters = carray(udata, (5,), np.float64)
    _fill_residual(carray(x, (4,), np.float64), carray(fvec, (4,), np.float64), parameters)


@cfunc(minpack_numba.hybrd_sig)
def _guarded_residual(
    _n: int, x: _Pointer[c_double], fvec: _Pointer[c_double], iflag: _Pointer[c_int32], udata: c_void_p
) -> None:
    # Parameters: spin, depth, real seed, imaginary seed, trust radius.
    parameters = carray(udata, (5,), np.float64)
    w = complex(x[0], x[1])
    if abs(w - complex(parameters[2], parameters[3])) > parameters[4] or w.imag >= 0 or w.real <= 0:
        iflag[0] = -1
        return
    _fill_residual(carray(x, (4,), np.float64), carray(fvec, (4,), np.float64), parameters)


_residual_address = _residual.address
_guarded_residual_address = _guarded_residual.address
assert _residual_address is not None
assert _guarded_residual_address is not None
_RESIDUAL_ADDRESS = _residual_address
_GUARDED_RESIDUAL_ADDRESS = _guarded_residual_address


@njit()
def _hybrd(
    spin: float,
    seed: complex,
    separation_seed: complex,
    depth: int,
    callback_address: int = _RESIDUAL_ADDRESS,
) -> _HybrdResult:
    x0 = np.array([seed.real, seed.imag, separation_seed.real, separation_seed.imag])
    parameters = np.array([spin, float(depth), seed.real, seed.imag, 0.5 * abs(seed.imag)])
    # The default xtol (about sqrt(eps)) leaves ~1e-10 relative error from distant seeds.
    x, fvec, _fjac, _r, _qtf, nfev, info = minpack_numba.hybrd(callback_address, x0, xtol=1e-10, udata=parameters)
    return _HybrdResult(complex(x[0], x[1]), complex(x[2], x[3]), int(info), float(np.max(np.abs(fvec))), int(nfev))


@pytest.fixture(scope='module')
def native_data() -> tuple[FloatArray, FloatArray]:
    resource = files('IMRPhenomWaveforms.data').joinpath('cook-qnm-v4-quadrupole.h5')
    with as_file(resource) as path, h5py.File(str(path), 'r') as data:
        positive = np.asarray(data['n00/m+02/{2,2,0}'], dtype=np.float64)
        negative = np.asarray(data['n00/m-02/{2,-2,0}'], dtype=np.float64)
    return positive, negative


def _native_seed(native_data: tuple[FloatArray, FloatArray], spin: float) -> tuple[float, complex, complex]:
    # CITE paper=cook-2014-kerr-qnm-spectral-method fact=quadrupole-eigenvalue-data
    # CITE paper=leaver-1985-kerr-qnm-eigenproblem fact=units-and-radiative-boundary-conditions
    data = native_data[0] if spin >= 0 else native_data[1]
    row = data[np.argmin(abs(data[:, 0] - abs(spin)))]
    return float(np.copysign(row[0], spin)), 2 * complex(row[1], row[2]), complex(row[3], row[4])


@pytest.mark.parametrize('offset', [-0.3, 1e-3, 0.3])
@pytest.mark.parametrize('depth', [128, 1024])
@pytest.mark.parametrize('target', [0.0, 0.3, 0.7, -0.7, 0.99, -0.99, 0.999999])
def test_unguarded_hybrd_agrees_with_newton(
    native_data: tuple[FloatArray, FloatArray], target: float, depth: int, offset: float
) -> None:
    spin, w, separation = _native_seed(native_data, target)
    seed = w + offset * abs(w.imag)
    separation_seed = separation + 1e-2 * offset
    newton = qnm_spin._newton(spin, seed, separation_seed, depth)
    hybrd = _hybrd(spin, seed, separation_seed, depth)
    # This comparison tests the same finite truncation, not its physical accuracy.
    assert newton[2]
    assert hybrd.info == 1
    assert hybrd.evaluations > 1
    assert hybrd.residual < 1e-9
    assert hybrd.frequency == pytest.approx(newton[0], rel=1e-12)
    assert hybrd.separation == pytest.approx(newton[1], rel=1e-12)
    assert abs(hybrd.frequency - seed) > 0.1 * abs(seed - w)
    a = abs(spin) / 2
    m = 2 if spin >= 0 else -2
    assert abs(qnm_spin._radial(newton[0], newton[1], a, m, depth)) < 1e-9
    assert abs(qnm_spin._angular(newton[0], newton[1], a, m)) < 1e-9


@pytest.mark.parametrize('depth', [128, 1024])
@pytest.mark.parametrize('target', [0.0, 0.3, 0.7, -0.7, 0.99, -0.99, 0.999999])
def test_trust_radius_rejects_a_distant_true_root(
    native_data: tuple[FloatArray, FloatArray], target: float, depth: int
) -> None:
    # The trust radius is conservative: unguarded HYBRD reaches the same truncated root
    # that Newton finds from Cook's seed, but it lies outside 0.5 |Im seed|.
    spin, w, separation = _native_seed(native_data, target)
    reference = qnm_spin._newton(spin, w, separation, depth)
    seed = w + 0.6 * abs(w.imag)
    separation_seed = separation + 0.006
    unguarded = _hybrd(spin, seed, separation_seed, depth)
    guarded = _hybrd(spin, seed, separation_seed, depth, _GUARDED_RESIDUAL_ADDRESS)
    newton = qnm_spin._newton(spin, seed, separation_seed, depth)

    assert reference[2]
    assert unguarded.info == 1
    assert unguarded.evaluations > 1
    assert unguarded.residual < 1e-9
    # From this distant seed HYBRD stops at its xtol = 1e-10 relative step tolerance.
    assert unguarded.frequency == pytest.approx(reference[0], rel=1e-10)
    assert unguarded.separation == pytest.approx(reference[1], rel=1e-10)
    assert abs(unguarded.frequency - seed) > 0.5 * abs(seed.imag)
    assert guarded.info == -1
    assert not newton[2]
