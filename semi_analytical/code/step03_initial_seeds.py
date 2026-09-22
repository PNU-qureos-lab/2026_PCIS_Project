"""English documentation is provided in README.md."""
import math
import numpy as np
from scipy.optimize import minimize_scalar
from settings import PAIRS
from search_state import CandidateState, EvaluationLimit

def initialize_seed(seed):
    """English documentation is provided in README.md."""
    if seed.initial_status != "uncomputed":
        raise ValueError("Each candidate may only be initialized once")
    start_evaluations = seed.evaluations

    def objective(z):
        """English documentation is provided in README.md."""
        z = float(z)
        if z in seed.initial_values:
            return seed.initial_values[z]
        before = seed.evaluations
        value = seed._evaluate_at(z, seed.initial_f)
        seed.initial_values[z] = value
        seed.initial_trials.append(dict(z=z, c4=value, evaluations=seed.evaluations,
                                        counted=seed.evaluations > before))
        return value

    try:
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        objective(0.0)
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            # The model rationale and parameter policy are documented in README.md.
            # The model rationale and parameter policy are documented in README.md.
            result = minimize_scalar(objective, method="brent", bracket=(0.0, 1.0),
                options={"xtol": seed.settings.initial_xtol,
                         "maxiter": seed.settings.initial_maxiter})
        seed.initial_solver_success = bool(result.success and math.isfinite(float(result.fun)))
        seed.initial_solver_message = str(result.message)
        seed.initial_nit = int(result.nit) if hasattr(result, "nit") else None
        seed.initial_solver_nfev = int(result.nfev) if hasattr(result, "nfev") else None
        if result.success and math.isfinite(float(result.fun)):
            seed.initial_status = "success"
        elif (seed.initial_nit is not None and seed.initial_nit >= seed.settings.initial_maxiter
              or "maximum" in str(result.message).lower()):
            seed.initial_status = "maxiter"
        else:
            seed.initial_status = "no_bracket"
    except EvaluationLimit:
        seed.initial_status = "evaluation_limit"
        seed.initial_solver_message = "Per-seed C4 evaluation limit reached during initialization"
    except (ValueError, RuntimeError, OverflowError) as exc:
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        seed.initial_status = "no_bracket"
        seed.initial_solver_message = str(exc)
    seed.initialization_evaluations = seed.evaluations - start_evaluations
    # The model rationale and parameter policy are documented in README.md.
    # The model rationale and parameter policy are documented in README.md.
    finite = [(value, z) for z, value in seed.initial_values.items()
              if z >= 0 and math.isfinite(z) and math.isfinite(value)]
    if not finite:
        raise ValueError(f"No finite initial candidate ({seed.initial_status}): {seed.initial_solver_message}")
    seed.initial_cost, seed.z_origin = min(finite, key=lambda item: (item[0], item[1]))
    seed.initial_boundary_selected = seed.z_origin == 0.0
    # The model rationale and parameter policy are documented in README.md.
    seed.initial_key = seed.best_key = (0, 0)
    seed.best_cost = seed.initial_cost
    seed.cache[seed.initial_key] = seed.initial_cost
    return seed.initial_cost

def build_initial_seeds(observed, context, library, settings):
    """English documentation is provided in README.md."""
    settings.validate()
    seeds = [CandidateState(observed, context, library, pair, fraction, settings)
             for pair in PAIRS for fraction in settings.seed_fractions]
    for seed in seeds:
        initialize_seed(seed)
    return seeds
