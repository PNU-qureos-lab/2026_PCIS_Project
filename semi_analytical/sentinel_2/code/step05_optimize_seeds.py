"""English documentation is provided in README.md."""
import math
import numpy as np
from model import forward, components, MATERIALS
from search_state import EvaluationLimit

# The model rationale and parameter policy are documented in README.md.
# The model rationale and parameter policy are documented in README.md.
NEIGHBOR_OFFSETS = tuple((dz, df) for dz in (-1, 0, 1) for df in (-1, 0, 1)
                         if (dz, df) != (0, 0))

def optimize_seed(seed):
    """English documentation is provided in README.md."""
    key, cost = seed.initial_key, seed.initial_cost
    if key is None:
        raise ValueError("Initialize before optimization")
    seed.log(key, cost, 0, "initial")
    cycle, reason = 0, "no_c4_improvement"
    try:
        if seed.initial_status == "evaluation_limit":
            raise EvaluationLimit
        while True:
            cycle += 1
            options = []
            # The model rationale and parameter policy are documented in README.md.
            # The model rationale and parameter policy are documented in README.md.
            for dz, df in NEIGHBOR_OFFSETS:
                neighbor = (key[0] + dz, key[1] + df)
                options.append((seed.evaluate(neighbor), neighbor))
            # The model rationale and parameter policy are documented in README.md.
            next_cost, next_key = min(options, key=lambda item: (item[0], item[1]))
            if not next_cost < cost:
                break
            dz, df = next_key[0] - key[0], next_key[1] - key[1]
            event = "move_diagonal" if dz and df else "move_depth" if dz else "move_fraction"
            key, cost = next_key, next_cost
            # The model rationale and parameter policy are documented in README.md.
            seed.log(key, cost, cycle, event)
    # The model rationale and parameter policy are documented in README.md.
    except EvaluationLimit:
        reason = "evaluation_limit"
        # The model rationale and parameter policy are documented in README.md.
        key, cost = seed.best_key, seed.best_cost
    seed.log(key, cost, cycle, reason)
    z, f = seed.coordinates(key)
    iz, _ = seed.coordinates(seed.initial_key)
    pred = forward(z, seed.rho(f), seed.context)
    # The model rationale and parameter policy are documented in README.md.
    cf = components(seed.pair, f)
    return dict(seed_id=seed.seed_id, pair="_".join(seed.pair),
                initial_F=seed.initial_f, initial_z=iz, initial_c4=seed.initial_cost,
                z_hat=z, F=f, **{f"f_{m}":cf[m] for m in MATERIALS},
                c4=float(cost), rms_relative_error_pct=100*math.sqrt(cost),
                status=reason, cycles=cycle, evaluations=seed.evaluations,
                **seed.initial_metadata(),
                f_negative=f < 0, f_ge1=f >= 1, f_outside=(f < 0 or f > 1),
                fraction_label="negative" if f < 0 else "above_one" if f > 1 else "pure_first" if f == 1 else "pure_second" if f == 0 else "within_range",
                bottom_reflectance_negative=bool(np.any(seed.rho(f) < 0)),
                bottom_reflectance_above_one=bool(np.any(seed.rho(f) > 1)),
                **{f"Rrs_model_{w}":float(v) for w,v in zip((443,490,560,665),pred)})

def optimize_seeds(selected_seeds):
    """English documentation is provided in README.md."""
    records, traces = [], []
    for seed in selected_seeds:
        records.append(optimize_seed(seed))
        traces.extend(seed.history)
    return records, traces
