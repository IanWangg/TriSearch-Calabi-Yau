"""Extremal generators of a pointed cone using reusable, single-thread MOSEK LPs.

For each generator r_i, minimize r_i @ y subject to R_other @ y >= 0 and
-1 <= y <= 1. This LP is always feasible and bounded. A negative optimum
provides a separating functional; at zero optimum the LP dual supplies
nonnegative reconstruction weights. Both certificates are checked explicitly.
"""

from __future__ import annotations

import atexit
from functools import lru_cache
import os
from pathlib import Path
import time

import numpy as np


class NonPointedConeError(ValueError):
    """The cone requires CYTools' lineality-space decomposition."""


class ExtremalRaySolveError(RuntimeError):
    """No validated reconstruction or separating certificate was obtained."""


@lru_cache(maxsize=1)
def _environment(process_id):
    # The PID prevents reusing a parent's environment after a fork. Managed
    # geometry workers use spawn and each own their environment and license.
    import mosek
    from core.cytools_config import MOSEK_LICENSE_PATH

    if MOSEK_LICENSE_PATH and Path(MOSEK_LICENSE_PATH).is_file():
        os.environ.setdefault("MOSEKLM_LICENSE_FILE", MOSEK_LICENSE_PATH)
    environment = mosek.Env()
    atexit.register(environment.__exit__, None, None, None)
    return environment


def _prepare_rays(rays):
    rays = np.asarray(rays)
    if rays.ndim != 2 or rays.shape[1] == 0:
        raise ValueError("rays must be a matrix with positive ambient dimension.")
    if not np.issubdtype(rays.dtype, np.number) or not np.isfinite(rays).all():
        raise ValueError("rays must contain finite real numbers.")
    if np.iscomplexobj(rays):
        raise ValueError("rays must contain real numbers.")
    rays = rays[np.any(rays != 0, axis=1)].copy()
    if np.issubdtype(rays.dtype, np.integer) and len(rays):
        # Positive rescaling preserves directions; never flip a ray's sign.
        divisors = np.gcd.reduce(rays, axis=1)
        if np.any(divisors <= 0):
            raise ValueError("Ray coordinates exceed supported integer normalization.")
        rays //= divisors[:, None]
    rays = np.unique(rays, axis=0)
    scaled = rays.astype(np.float64)
    if len(scaled):
        scaled /= np.max(np.abs(scaled), axis=1)[:, None]
    return rays, scaled


def _configure_task(task, optimizer):
    import mosek

    task.putintparam(mosek.iparam.num_threads, 1)
    task.putintparam(mosek.iparam.optimizer, getattr(mosek.optimizertype, optimizer))
    # Presolve is important for the degenerate Mori-cap constraint systems.
    task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)
    task.putdouparam(mosek.dparam.basis_tol_x, 1e-9)
    task.putdouparam(mosek.dparam.basis_tol_s, 1e-9)
    task.putdouparam(mosek.dparam.optimizer_max_time, 30.0)


def _solve_checked(task, validate, diagnostics):
    """Retry an inconclusive solve once with a fresh basis and other simplex."""
    import mosek

    original = task.getintparam(mosek.iparam.optimizer)
    status = termination = None
    try:
        for attempt in range(2):
            if attempt:
                alternate = (mosek.optimizertype.primal_simplex
                             if original == mosek.optimizertype.dual_simplex
                             else mosek.optimizertype.dual_simplex)
                task.putintparam(mosek.iparam.optimizer, alternate)
                task.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.none)
                task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)
                diagnostics["retries"] += 1
            termination = task.optimize()
            diagnostics["lp_solves"] += 1
            diagnostics["optimizer_sec"] += task.getdouinf(mosek.dinfitem.optimizer_time)
            status = task.getsolsta(mosek.soltype.bas)
            if termination == mosek.rescode.ok:
                result = validate(status)
                if result is not None:
                    return result
    finally:
        task.putintparam(mosek.iparam.optimizer, original)
        task.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.free)
        task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)
    raise ExtremalRaySolveError(
        f"MOSEK LP remained inconclusive after two attempts: {termination}, {status}."
    )


def _pointed(environment, rays, optimizer, tolerance, diagnostics):
    """A strictly positive functional exists exactly for pointed V-cones."""
    import mosek

    count, dimension = rays.shape
    with environment.Task(0, 0) as task:
        _configure_task(task, optimizer)
        task.appendcons(count)
        task.appendvars(dimension)
        task.putvarboundslice(0, dimension, [mosek.boundkey.fr] * dimension,
                             [0.0] * dimension, [0.0] * dimension)
        task.putconboundslice(0, count, [mosek.boundkey.lo] * count,
                             [1.0] * count, [0.0] * count)
        for column in range(dimension):
            indices = np.flatnonzero(rays[:, column])
            task.putacol(column, indices.tolist(), rays[indices, column].tolist())

        def validate(status):
            if status == mosek.solsta.optimal:
                functional = np.asarray(task.getxx(mosek.soltype.bas))
                if np.isfinite(functional).all() and np.min(rays @ functional) >= 1.0 - tolerance:
                    return True
            elif status == mosek.solsta.prim_infeas_cer:
                weights = np.asarray(task.gety(mosek.soltype.bas))
                scale = np.max(np.abs(weights))
                if np.isfinite(scale) and scale > 0:
                    weights = weights / scale
                    if (weights.min() >= -tolerance and weights.sum() > tolerance
                            and np.max(np.abs(rays.T @ weights)) <= tolerance):
                        return False
            return None

        return _solve_checked(task, validate, diagnostics)


def mosek_extremal_rays(rays, *, optimizer="primal_simplex", tolerance=1e-7, diagnostics=None):
    """Return deterministically ordered primitive generators of a pointed cone.

    Integer inputs are reduced by positive GCD and exact duplicates/zero rows
    are removed. Floating inputs are not rounded or merged approximately.
    Non-pointed cones raise ``NonPointedConeError`` so callers can preserve
    CYTools' ``minimal=True`` lineality-space convention. ``diagnostics`` may
    be a caller-owned dict; it retains scalars only, never a task or geometry.
    """
    import mosek

    if optimizer not in ("primal_simplex", "dual_simplex"):
        raise ValueError("optimizer must be primal_simplex or dual_simplex.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    started = time.perf_counter()
    original_count = len(rays)
    generators, scaled = _prepare_rays(rays)
    count, dimension = scaled.shape
    stats = {} if diagnostics is None else diagnostics
    stats.update(input_rays=original_count, unique_rays=count, dimension=dimension,
                 optimizer=optimizer, lp_solves=0, retries=0, optimizer_sec=0.0)
    if count <= 1:
        stats.update(extremal_rays=count, total_sec=time.perf_counter() - started)
        return generators
    environment = _environment(os.getpid())
    if not _pointed(environment, scaled, optimizer, tolerance, stats):
        raise NonPointedConeError("Cone is not pointed; use the lineality-space decomposition.")

    active = np.ones(count, dtype=bool)
    keep = np.ones(count, dtype=bool)
    with environment.Task(0, 0) as task:
        _configure_task(task, optimizer)
        task.appendcons(count)
        task.appendvars(dimension)
        task.putvarboundslice(0, dimension, [mosek.boundkey.ra] * dimension,
                             [-1.0] * dimension, [1.0] * dimension)
        task.putconboundslice(0, count, [mosek.boundkey.lo] * count,
                             [0.0] * count, [0.0] * count)
        for column in range(dimension):
            indices = np.flatnonzero(scaled[:, column])
            task.putacol(column, indices.tolist(), scaled[indices, column].tolist())

        for index, target in enumerate(scaled):
            active[index] = False
            task.putconbound(index, mosek.boundkey.fr, 0.0, 0.0)
            task.putclist(list(range(dimension)), target.tolist())

            def validate(status):
                if status != mosek.solsta.optimal:
                    return None
                separating = np.asarray(task.getxx(mosek.soltype.bas))
                if (not np.isfinite(separating).all()
                        or np.max(np.abs(separating)) > 1.0 + tolerance
                        or np.any(scaled[active] @ separating < -tolerance)):
                    return None
                if target @ separating < -tolerance:
                    return False
                # At optimum zero, box-bound multipliers vanish and the row
                # dual is a conic reconstruction of the candidate generator.
                weights = np.asarray(task.gety(mosek.soltype.bas))
                if (np.isfinite(weights).all() and weights.min() >= -tolerance
                        and np.max(np.abs(weights[~active])) <= tolerance
                        and np.max(np.abs(scaled.T @ weights - target)) <= tolerance):
                    return True
                return None

            try:
                redundant = _solve_checked(task, validate, stats)
            except ExtremalRaySolveError as error:
                raise ExtremalRaySolveError(f"Ray {index}/{count}: {error}") from error
            if redundant:
                keep[index] = False
            else:
                active[index] = True
                task.putconbound(index, mosek.boundkey.lo, 0.0, 0.0)
    result = generators[keep]
    stats.update(extremal_rays=len(result), total_sec=time.perf_counter() - started)
    return result
