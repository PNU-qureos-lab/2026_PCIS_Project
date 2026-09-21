"""한 seed의 좌표, C4 평가 횟수, 중간 기록을 보관하는 공통 코드입니다.

계산 방향과 종료 규칙은 step03_initial_seeds.py 및 step05_optimize_seeds.py에 있습니다.
여기서는 같은 점을 중복 계산하지 않도록 기억하고 고정 스텝 좌표를 실제 z·F로 바꿉니다.
실제 수심·실제 혼합비는 받지 않습니다.
계산 횟수는 반복 횟수와 다릅니다. 한 번의 ± 비교는 새 점을 최대 두 번 평가합니다.
"""
import math
import numpy as np
from model import c4

class EvaluationLimit(Exception):
    """Seed별 C4 평가 한도에 닿았음을 알리는 전용 예외입니다.

    계산 한도를 '오차 개선이 없어 끝났다'고 잘못 기록하지 않기 위해 따로 구분합니다.
    """
    pass


class CandidateState:
    """초기 F 하나와 바닥쌍 하나에 대한 탐색 상태를 보관합니다.

    물리 좌표 캐시는 실제 (z,F), 반복 좌표 캐시는 초기값 대비 정수 이동 횟수를 사용합니다.
    동일한 물리 상태는 초기 수심 계산과 반복 단계에서 함께 재사용할 수 있습니다.
    """
    def __init__(self, observed, context, library, pair, initial_f, settings):
        """관측 네 밴드와 두 순수 바닥 스펙트럼, 설정을 받아 빈 상태를 만듭니다.

        initial_f는 아직 최적화되지 않은 시작 계수입니다.
        z_origin은 3단계가 정하기 전에는 None이며 임의 수심을 넣지 않습니다.
        초기 solver 상태·계산 횟수·이동 기록을 각각 별도 변수로 보관합니다.
        """
        self.observed = np.asarray(observed, float)
        c4(self.observed, self.observed)
        self.context, self.pair, self.initial_f, self.settings = context, pair, initial_f, settings
        self.rho_a = np.asarray(library[pair[0]], float)
        self.rho_b = np.asarray(library[pair[1]], float)
        self.seed_id = f"{pair[0]}_{pair[1]}_F{initial_f:.2f}"
        # cache: 정수 이동 좌표별 C4. physical_cache: 실제 (z,F)별 C4.
        # initial_values/trials: 연속 초기 수심 계산의 점수와 모든 시도 기록.
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
        """정수 좌표 (nz,nf)를 실제 수심 z와 혼합 계수 F로 변환합니다.

        기본 nz=-1은 초기 수심보다 0.1 m 얕다는 뜻이며 그 자체가 음수 수심은 아닙니다.
        예: 초기 2.37 m에서 nz=-1이면 2.27 m입니다.
        F의 소수 12자리 반올림은 0.01 반복 덧셈의 부동소수점 흔들림 정리입니다.
        F를 0~1에 맞추거나 이동 간격을 줄이는 처리가 아닙니다.
        """
        nz, nf = key
        if self.z_origin is None:
            raise ValueError("Initialize before requesting lattice coordinates")
        return self.z_origin+nz*self.settings.depth_step, round(self.initial_f+nf*self.settings.fraction_step, 12)

    def rho(self, fraction):
        """현재 F로 두 순수 바닥 스펙트럼을 선형 결합합니다. F 범위를 제한하지 않습니다.
        """
        return fraction*self.rho_a+(1-fraction)*self.rho_b

    def remember(self, key, value):
        """평가한 격자 좌표의 C4를 저장하고 지금까지의 최저값이면 갱신합니다.
        """
        self.cache[key] = float(value)
        if value < self.best_cost:
            self.best_key, self.best_cost = key, float(value)

    def _evaluate_at(self, z, f):
        """실제 (z,F) 한 점의 D2 예측과 C4를 계산합니다.

        음수 수심이나 비유한 좌표는 무한대 C4로 거절하고 모델 계산 횟수에 넣지 않습니다.
        이미 계산한 같은 (z,F)는 캐시에서 읽으므로 횟수가 늘지 않습니다.
        새 물리 점을 평가할 때만 횟수 1을 더하며 초기 수심과 반복 계산이 같은 한도를 씁니다.

        수심만으로 정해지는 지수항을 저장해 속도를 높입니다.
        아래 스칼라 네 밴드 식은 model.forward와 model.c4를 풀어 쓴 동일 계산입니다.
        분모 하한, F 제한, 새 오차식을 추가하지 않습니다.
        수면 변환의 분모가 유효하지 않으면 그 점의 오차를 무한대로 기록합니다.
        """
        z, f = float(z), float(f)
        if not math.isfinite(z) or z < 0 or not math.isfinite(f):
            return math.inf
        physical_key = (z, f)
        if physical_key in self.physical_cache:
            return self.physical_cache[physical_key]
        # 이 검사는 새 점을 계산하기 전에 합니다. 이미 계산한 점의 재조회는 한도를 쓰지 않습니다.
        if self.evaluations >= self.settings.max_evaluations:
            raise EvaluationLimit
        # 수면 경계 변환 전의 바닥 혼합항은 F에 대해 선형입니다.
        # 수심만으로 정해지는 항을 저장합니다. 이동 스텝과 오차식은 그대로입니다.
        terms = self.depth_terms.get(z)
        if terms is None:
            terms = []
            for j in range(4):
                eb = math.exp(-float(self.context['xb'][j])*z)/math.pi
                water = -float(self.context['rrs_dp'][j])*math.expm1(-float(self.context['xc'][j])*z)
                terms.append((water,eb,float(self.rho_a[j]),float(self.rho_b[j]),float(self.observed[j])))
            self.depth_terms[z] = terms
        total = 0.0
        # 네 밴드 각각의 상대 제곱오차를 합한 다음 4로 나눕니다.
        # 음수 F라도 유효한 모델 값이면 같은 식으로 평가합니다.
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
        """정수 이동 좌표를 평가하고 반복 단계에서의 최저 C4를 기억합니다.

        같은 좌표가 있으면 캐시를 그대로 반환합니다. 탐색 방향은 이 함수가 정하지 않습니다.
        """
        if key in self.cache:
            return self.cache[key]
        z, f = self.coordinates(key)
        value = self._evaluate_at(z, f)
        self.remember(key, value)
        return value

    def initial_metadata(self):
        """초기 수심 수치해법의 상태와 정밀도·계산 횟수를 기록용 사전으로 반환합니다.

        initial_solver_nfev는 solver가 보고한 호출 수입니다.
        initialization_evaluations는 캐시·정의역 거절을 반영한 실제 C4 계산 수여서 다를 수 있습니다.
        """
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
        """채택한 좌표와 종료 사건을 trajectory.csv용 기록에 추가합니다.

        event는 initial, move_fraction, move_depth 또는 종료 이유입니다.
        이 기록은 모든 거절된 이웃 점을 나열한 표가 아니라 채택한 이동과 종료의 경로입니다.
        """
        z, f = self.coordinates(key)
        self.history.append(dict(seed_id=self.seed_id, cycle=cycle, event=event,
                                 z=z, F=f, c4=cost, evaluations=self.evaluations))
