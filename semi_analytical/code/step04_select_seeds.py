"""English documentation is provided in README.md."""

def select_seeds(seeds, settings):
    """English documentation is provided in README.md."""
    # The model rationale and parameter policy are documented in README.md.
    # The model rationale and parameter policy are documented in README.md.
    ranked = sorted(seeds, key=lambda seed:(seed.initial_cost, seed.seed_id))
    if len(ranked) < settings.top_k:
        raise ValueError("Not enough initialized seeds for top_k")
    rows = []
    for index, seed in enumerate(ranked):
        z, _ = seed.coordinates(seed.initial_key)
        rows.append(dict(seed_id=seed.seed_id, initial_rank=index+1,
            initial_F=seed.initial_f, initial_z=z, initial_c4=seed.initial_cost,
            selected=index < settings.top_k, **seed.initial_metadata(),
            initial_trials=seed.initial_trials))
    # The model rationale and parameter policy are documented in README.md.
    return ranked[:settings.top_k], rows
