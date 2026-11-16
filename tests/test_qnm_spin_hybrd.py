# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Agreement checks between MINPACK HYBRD and the Newton Leaver solver."""

from __future__ import annotations

import os
from importlib.resources import as_file, files
from typing import TYPE_CHECKING

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


@cfunc(minpack_numba.hybrd_sig)
def _residual(
    _n: int, x: _Pointer[c_double], fvec: _Pointer[c_double], iflag: _Pointer[c_int32], udata: c_void_p
) -> None:
    # Parameters: spin, depth, real seed, imaginary seed, trust radius.
    parameters = carray(udata, (5,), np.float64)
    spin = parameters[0]
    a = abs(spin) / 2
    m = 2 if spin >= 0 else -2
    w = complex(x[0], x[1])
    separation = complex(x[2], x[3])
    if abs(w - complex(parameters[2], parameters[3])) > parameters[4] or w.imag >= 0 or w.real <= 0:
        iflag[0] = -1
        return
    radial = qnm_spin._radial(w, separation, a, m, int(parameters[1]))
    angular = qnm_spin._angular(w, separation, a, m)
    fvec[0] = radial.real
    fvec[1] = radial.imag
    fvec[2] = angular.real
    fvec[3] = angular.imag


_RESIDUAL_ADDRESS = _residual.address


@njit()
def _hybrd(spin: float, seed: complex, separation_seed: complex, depth: int) -> tuple[complex, complex, bool]:
    x0 = np.array([seed.real, seed.imag, separation_seed.real, separation_seed.imag])
    parameters = np.array([spin, float(depth), seed.real, seed.imag, 0.5 * abs(seed.imag)])
    # The default xtol (about sqrt(eps)) leaves ~1e-10 relative error from distant seeds.
    x, _fvec, _fjac, _r, _qtf, _nfev, info = minpack_numba.hybrd(_RESIDUAL_ADDRESS, x0, xtol=1e-10, udata=parameters)
    return complex(x[0], x[1]), complex(x[2], x[3]), info == 1


@pytest.fixture(scope='module')
def native_data() -> tuple[FloatArray, FloatArray]:
    resource = files('IMRPhenomWaveforms.data').joinpath('cook-qnm-v4-quadrupole.h5')
    with as_file(resource) as path, h5py.File(str(path), 'r') as data:
        positive = np.asarray(data['n00/m+02/{2,2,0}'], dtype=np.float64)
        negative = np.asarray(data['n00/m-02/{2,-2,0}'], dtype=np.float64)
    return positive, negative


def _native_seed(native_data: tuple[FloatArray, FloatArray], spin: float) -> tuple[float, complex, complex]:
    data = native_data[0] if spin >= 0 else native_data[1]
    row = data[np.argmin(abs(data[:, 0] - abs(spin)))]
    return float(np.copysign(row[0], spin)), 2 * complex(row[1], row[2]), complex(row[3], row[4])


@pytest.mark.parametrize(('offset', 'expected'), [(0.0, True), (1e-3, True), (0.3, True), (0.6, False)])
@pytest.mark.parametrize('depth', [128, 1024])
@pytest.mark.parametrize('target', [0.0, 0.3, 0.7, -0.7, 0.99, -0.99, 0.999999])
def test_hybrd_agrees_with_newton(
    native_data: tuple[FloatArray, FloatArray], target: float, depth: int, offset: float, *, expected: bool
) -> None:
    spin, w, separation = _native_seed(native_data, target)
    seed = w + offset * abs(w.imag)
    separation_seed = separation + 1e-2 * offset
    newton = qnm_spin._newton(spin, seed, separation_seed, depth)
    hybrd = _hybrd(spin, seed, separation_seed, depth)
    assert newton[2] == expected
    assert hybrd[2] == expected
    if expected:
        assert hybrd[0] == pytest.approx(newton[0], rel=1e-12)
        assert hybrd[1] == pytest.approx(newton[1], rel=1e-12)
