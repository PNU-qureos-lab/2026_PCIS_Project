"""English documentation is provided in README.md."""
import math
import numpy as np
from settings import MATERIALS

def classify_candidates(records, settings):
    """English documentation is provided in README.md."""
    ranked = sorted(records, key=lambda r:(r['c4'],r['seed_id']))
    if not ranked or not math.isfinite(ranked[0]['c4']):
        raise ValueError("No finite optimized candidates")
    representatives = []
    for rank, row in enumerate(ranked, 1):
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        signature = np.array([row['z_hat'], *(row['f_'+m] for m in MATERIALS)])
        match = next((i for i, rep in enumerate(representatives)
                      if np.allclose(signature, rep[0], rtol=0, atol=settings.same_solution_atol)), None)
        row['rank'] = rank
        row['same_solution_duplicate'] = match is not None
        if match is None:
            match = len(representatives)
            representatives.append((signature, row))
        row['solution_group'] = match+1
        row['same_solution_as'] = representatives[match][1]['seed_id']
    winner = ranked[0]
    # The model rationale and parameter policy are documented in README.md.
    cutoff = winner['c4']*(1+settings.near_relative_c4)
    for row in ranked:
        row['near_best'] = bool(row['c4'] <= cutoff)
        row['relative_c4_gap'] = ((row['c4']-winner['c4'])/winner['c4']
                                  if winner['c4'] > 0 else (0.0 if row['c4'] == 0 else math.inf))
        row['is_representative'] = row['seed_id'] == winner['seed_id']
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        row['is_alternative'] = row['near_best'] and row['solution_group'] != winner['solution_group'] and not row['same_solution_duplicate']
    alternatives = [row for row in ranked if row['is_alternative']]
    # The model rationale and parameter policy are documented in README.md.
    summary = dict(winner)
    summary.update(has_alternative=bool(alternatives), alternative_count=len(alternatives),
                   unique_solution_count=len(representatives),
                   same_solution_duplicate_count=len(ranked)-len(representatives),
                   total_evaluations=sum(r['evaluations'] for r in ranked),
                   any_evaluation_limit=any(r['status']=='evaluation_limit' for r in ranked),
                   alternative_z_min=min([winner['z_hat']]+[r['z_hat'] for r in alternatives]),
                   alternative_z_max=max([winner['z_hat']]+[r['z_hat'] for r in alternatives]))
    return summary, ranked
