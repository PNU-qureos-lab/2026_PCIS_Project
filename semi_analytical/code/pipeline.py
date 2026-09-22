"""English documentation is provided in README.md."""
from dataclasses import asdict
from settings import Settings
from step03_initial_seeds import build_initial_seeds
from step04_select_seeds import select_seeds
from step05_optimize_seeds import optimize_seeds
from step06_final_solution import classify_candidates

def solve(observed, context, library, settings=None):
    """English documentation is provided in README.md."""
    settings = settings or Settings()
    settings.validate()
    seeds = build_initial_seeds(observed, context, library, settings)  # 3
    selected, initials = select_seeds(seeds, settings)                 # 4
    records, traces = optimize_seeds(selected)                         # 5
    summary, records = classify_candidates(records, settings)          # 6
    summary['total_evaluations'] = sum(seed.evaluations for seed in seeds)
    summary['settings'] = asdict(settings)
    return summary, records, traces, initials

def solve_payload(payload):
    """English documentation is provided in README.md."""
    observed, context, library, settings = payload
    try:
        return solve(observed, context, library, settings)
    except (ValueError, AssertionError, RuntimeError) as exc:
        return repr(exc)
