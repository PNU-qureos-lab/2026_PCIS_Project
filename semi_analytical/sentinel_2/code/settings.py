"""English documentation is provided in README.md."""
from dataclasses import dataclass
from pathlib import Path
import math

BANDS_NM = (443, 490, 560, 665)
MATERIALS = ("sand", "seagrass")
PAIRS = (("sand", "seagrass"),)
DEFAULT_DATA = Path("input_values")
DEFAULT_BOTTOM = Path("input_values/reference_data")
# The model rationale and parameter policy are documented in README.md.
FIGURE5_DEPTH_RANGE_M = 4.5
FIGURE5_COEFFICIENT_RANGE = 1.0

@dataclass(frozen=True)
class Settings:
    """English documentation is provided in README.md."""
    # The model rationale and parameter policy are documented in README.md.
    depth_step: float = 0.1
    # The model rationale and parameter policy are documented in README.md.
    fraction_step: float = 0.01
    # The model rationale and parameter policy are documented in README.md.
    seed_fractions: tuple = (0.75, 0.5, 0.25)
    # The model rationale and parameter policy are documented in README.md.
    top_k: int = 3
    # The model rationale and parameter policy are documented in README.md.
    max_evaluations: int = 10000
    # The model rationale and parameter policy are documented in README.md.
    initial_xtol: float = 1e-4
    # The model rationale and parameter policy are documented in README.md.
    initial_maxiter: int = 500
    # The model rationale and parameter policy are documented in README.md.
    near_relative_c4: float = 0.01
    # The model rationale and parameter policy are documented in README.md.
    same_solution_atol: float = 1e-9

    def validate(self):
        """English documentation is provided in README.md."""
        if self.depth_step <= 0 or self.fraction_step <= 0:
            raise ValueError("Steps must be positive")
        if self.max_evaluations < 20:
            raise ValueError("At least 20 evaluations are required")
        if not math.isfinite(self.initial_xtol) or self.initial_xtol <= 0:
            raise ValueError("Initial Brent xtol must be finite and positive")
        if self.initial_maxiter < 1:
            raise ValueError("Initial Brent maxiter must be positive")
        if not 1 <= self.top_k <= len(PAIRS) * len(self.seed_fractions):
            raise ValueError("Invalid top_k")
        if self.near_relative_c4 < 0 or self.same_solution_atol < 0:
            raise ValueError("Tolerances cannot be negative")

SETTING_NOTES = {
    "depth_step": "Fixed depth-search increment: 0.1 m.",
    "fraction_step": "Fixed bottom-fraction search increment: 0.01.",
    "top_k": "Optimize the three lowest-C4 initial seeds.",
    "near_relative_c4": "Report alternative solutions within 1% of minimum C4.",
    "max_evaluations": "Limit of 10,000 C4 evaluations per seed, including initialization.",
    "initial_xtol": "Brent relative tolerance 1e-4 for initial depth estimation.",
    "initial_maxiter": "Maximum 500 Brent iterations.",
    "same_solution_atol": "Numerical equality tolerance of 1e-9.",
    "search_neighborhood": "Compare four axial and four diagonal neighbours and move to the lowest C4.",
    "stopping": "Stop when none of the eight neighbours improves C4.",
    "fraction_bounds": "Fractions are not clipped during optimization.",
    "c4_denominator_floor": "No denominator floor; exactly zero observations are invalid.",
    "figure5_reference_ranges": "Reference ranges are 4.5 m for depth and 1 for fractions.",
}
