"""English documentation is provided in README.md."""
import math
import numpy as np
from model import c4

class EvaluationLimit(Exception):
    """English documentation is provided in README.md."""
    pass


class CandidateState:
    """English documentation is provided in README.md."""
    def __init__(self, observed, context, library, pair, initial_f, settings):
        """English documentation is provided in README.md."""
        self.observed = np.asarray(observed, float)
        c4(self.observed, self.observed)
        self.context, self.pair, self.initial_f, self.settings = context, pair, initial_f, settings
        self.rho_a = np.asarray(library[pair[0]], float)
        self.rho_b = np.asarray(library[pair[1]], float)
        self.seed_id = f"{pair[0]}_{pair[1]}_F{initial_f:.2f}"
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        self.cache = {}
        self.physical_cache = {}
        self.initial_values = {}
        self.initial_trials = []
        self.depth_terms = {}
        self.z_origin = None
        self.evaluations = 0
        self.best_key, self.best_cost = None, math.inf
        self.initial_key, self.initial_cost = None, math.inf
        self.initialization_evaluations = 0
        self.initial_status = "uncomputed"
        self.initial_solver_success = False
        self.initial_solver_message = ""
        self.initial_nit = None
        self.initial_solver_nfev = None
        self.initial_boundary_selected = False
        self.history = []

    def coordinates(self, key):
        """English documentation is provided in README.md."""
        nz, nf = key
        if self.z_origin is None:
            raise ValueError("Initialize before requesting lattice coordinates")
        return self.z_origin+nz*self.settings.depth_step, round(self.initial_f+nf*self.settings.fraction_step, 12)

    def rho(self, fraction):
        """English documentation is provided in README.md."""
        return fraction*self.rho_a+(1-fraction)*self.rho_b

    def remember(self, key, value):
        """English documentation is provided in README.md."""
        self.cache[key] = float(value)
        if value < self.best_cost:
            self.best_key, self.best_cost = key, float(value)

    def _evaluate_at(self, z, f):
        """English documentation is provided in README.md."""
        z, f = float(z), float(f)
        if not math.isfinite(z) or z < 0 or not math.isfinite(f):
            return math.inf
        physical_key = (z, f)
        if physical_key in self.physical_cache:
            return self.physical_cache[physical_key]
        # The model rationale and parameter policy are documented in README.md.
        if self.evaluations >= self.settings.max_evaluations:
            raise EvaluationLimit
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        terms = self.depth_terms.get(z)
        if terms is None:
            terms = []
            for j in range(4):
                eb = math.exp(-float(self.context['xb'][j])*z)/math.pi
                water = -float(self.context['rrs_dp'][j])*math.expm1(-float(self.context['xc'][j])*z)
                terms.append((water,eb,float(self.rho_a[j]),float(self.rho_b[j]),float(self.observed[j])))
            self.depth_terms[z] = terms
        total = 0.0
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        for water,eb,rho_a,rho_b,obs in terms:
            below = water+(f*rho_a+(1-f)*rho_b)*eb
            denominator = 1-1.7*below
            if denominator <= 0 or not math.isfinite(below):
                total = math.inf
                break
            predicted = 0.52*below/denominator
            residual = (predicted-obs)/obs
            total += residual*residual
        value = total/4.0
        if not math.isfinite(value): value=math.inf
        self.evaluations += 1
        self.physical_cache[physical_key] = value
        return value

    def evaluate(self, key):
        """English documentation is provided in README.md."""
        if key in self.cache:
            return self.cache[key]
        z, f = self.coordinates(key)
        value = self._evaluate_at(z, f)
        self.remember(key, value)
        return value

    def initial_metadata(self):
        """English documentation is provided in README.md."""
        return dict(initial_method="scipy_brent_nonnegative_boundary",
                    initial_status=self.initial_status,
                    initial_solver_success=self.initial_solver_success,
                    initial_solver_message=self.initial_solver_message,
                    initial_xtol=self.settings.initial_xtol,
                    initial_nit=self.initial_nit,
                    initial_solver_nfev=self.initial_solver_nfev,
                    initial_boundary_selected=self.initial_boundary_selected,
                    initialization_evaluations=self.initialization_evaluations)

    def log(self, key, cost, cycle, event):
        """English documentation is provided in README.md."""
        z, f = self.coordinates(key)
        self.history.append(dict(seed_id=self.seed_id, cycle=cycle, event=event,
                                 z=z, F=f, c4=cost, evaluations=self.evaluations))
