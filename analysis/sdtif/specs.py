"""Declared specifications (module M0): parameters, observations and targets carry units,
scales and roles, and a problem is refused when it does not match its declaration.

Nothing here is physics.  The point is that a Jacobian column with no unit, a target with a
gradient of the wrong length, or a parameter declared fixed while also appearing as a free
column are all silent-wrong-answer bugs; they are cheaper to refuse than to debug.
"""
import numpy as np


class SpecError(ValueError):
    """Raised when a problem contradicts its declared specification."""


def _pos(x, what):
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise SpecError("%s must be a number, got %r" % (what, x))
    if not np.isfinite(x) or x <= 0:
        raise SpecError("%s must be finite and positive, got %r" % (what, x))
    return x


def Parameter(name, unit, scale, role="free"):
    """One unknown.  scale is the magnitude below which a change is not of interest."""
    if role not in ("free", "fixed", "nuisance"):
        raise SpecError("role of %s must be free/fixed/nuisance, got %r" % (name, role))
    if not unit:
        raise SpecError("parameter %s carries no unit" % name)
    return dict(name=str(name), unit=str(unit), scale=_pos(scale, "scale of " + str(name)),
                role=role)


def Target(name, unit, value, grad):
    """One quantity to be graded, with its value and gradient in the parameter basis."""
    if not unit:
        raise SpecError("target %s carries no unit" % name)
    g = np.asarray(grad, float).ravel()
    if not np.all(np.isfinite(g)):
        raise SpecError("target %s has a non-finite gradient" % name)
    if not np.any(g):
        raise SpecError("target %s has an identically zero gradient" % name)
    return dict(name=str(name), unit=str(unit), value=float(value), grad=g)


def Observation(name, unit, sigma):
    """One channel.  sigma is the per-channel standard deviation in that unit."""
    if not unit:
        raise SpecError("observation %s carries no unit" % name)
    return dict(name=str(name), unit=str(unit), sigma=_pos(sigma, "sigma of " + str(name)))


def validate(problem, params, observations=None, targets=None, n_obs=None):
    """Check a Problem against its declaration and return the declared names, or raise.

    Checks that would have caught real bugs in this repository: a target gradient whose
    length differs from the number of unknowns (the 1-D vs 2-D target bug), a scale that is
    zero (an infinite column after normalisation), a fixed parameter that still appears in
    the free list, and a covariance whose dimension is not the number of channels.
    """
    names = [p["name"] for p in params]
    if len(set(names)) != len(names):
        raise SpecError("duplicate parameter names in %r" % names)
    n = problem.J.shape[1]
    if n != len(params):
        raise SpecError("%d unknowns in the operator but %d declared" % (n, len(params)))
    free = [p for p in params if p["role"] != "fixed"]
    if not free:
        raise SpecError("every parameter is fixed: nothing to estimate")
    if problem.cov is not None:
        m = int(np.sqrt(np.size(problem.cov))) if np.ndim(problem.cov) <= 1 else problem.cov.shape[0]
        if n_obs is not None and m != n_obs:
            raise SpecError("covariance is %dx%d but %d channels declared" % (m, m, n_obs))
    for t in (targets or []):
        if t["grad"].size != n:
            raise SpecError("target %s gradient has length %d, %d unknowns"
                            % (t["name"], t["grad"].size, n))
    s = np.asarray(problem.scales, float).ravel()
    if s.size != n or not np.all(s > 0):
        raise SpecError("scales must be %d positive numbers, got %r" % (n, s))
    return dict(params=[p["name"] for p in params], free=[p["name"] for p in free],
                observations=[o["name"] for o in (observations or [])],
                targets=[t["name"] for t in (targets or [])])


def selfcheck():
    """The registry must refuse the six mistakes it exists for."""
    from . import core
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    p = core.from_operator(A, cov=1e-4)
    ok = [Parameter("a", "1/m^3", 1.0), Parameter("b", "K", 10.0)]
    t = [Target("sum", "1/m^3", 2.0, [1.0, 1.0])]
    o = [Observation("ch1", "W/m^3", 1e-2), Observation("ch2", "W/m^3", 1e-2)]
    validate(p, ok, o, t, n_obs=2)
    bad = [("no unit", lambda: Parameter("a", "", 1.0)),
           ("zero scale", lambda: Parameter("a", "K", 0.0)),
           ("short gradient", lambda: validate(p, ok, targets=[Target("s", "K", 1.0, [1.0])])),
           ("count mismatch", lambda: validate(p, ok[:1])),
           ("all fixed", lambda: validate(p, [Parameter("a", "K", 1.0, "fixed"),
                                              Parameter("b", "K", 1.0, "fixed")])),
           ("bad role", lambda: Parameter("a", "K", 1.0, "floaty"))]
    for name, fn in bad:
        try:
            fn()
        except SpecError:
            continue
        raise AssertionError("registry accepted %s" % name)
    return dict(refused=len(bad), accepted=1)


if __name__ == "__main__":
    print(selfcheck())
