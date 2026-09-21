"""원래 변수의 최급강하법: 해석 기울기 + Brent 수치 선 최소화.

IOP에서는 x=(P,G,X)에 비음수 제약을 적용한다. 내부에서는 d=-∇C(x)이며,
G/X=0 경계에서는 음수로 나가려는 성분만 0으로 만든 허용 최급강하 방향을 쓴다.
공유 H/F 함수의 무제약 방향은 그대로 둔다. 실제 15차 H/F는 별도 12차 격자법이다.
변수별 배율, Armijo 계수, 0.5배/2배 학습률 규칙은 없다.
α는 φ(α)=C(x-α∇C(x)), α≥0의 일변수 최소화로 매번 새로 계산한다.
Brent의 포물선 보간/황금분할은 표준 선 최소화의 내부 연산이다.
비볼록 비용의 수치 선 최소화와 다중 시작점은 전역 최소를 증명하지 않는다.
P>=0 중 P=0은 ln(P)가 정의되지 않아 제외한다. 유효 영역은 P>0, G>=0, X>=0이다.
P에 임의의 양수 하한을 두지 않고, 분모!=0과 유한 계산 조건은 유지한다.
"""
from types import SimpleNamespace
import numpy as np
from numba import njit
from hope_iop_model import A0, A1, S, Y
from model import WAVES, G0, G1

EPS = np.finfo(np.float64).eps
GRADIENT_TOL = np.sqrt(EPS)
LINE_RTOL = np.sqrt(EPS)
GOLDEN_RATIO = (1.0 + np.sqrt(5.0))/2.0
GOLDEN_COMPLEMENT = (3.0 - np.sqrt(5.0))/2.0
# 계산 예산이며 수렴 조건이 아니다. 도달 시 성공으로 처리하지 않는다.
MAX_ITERATIONS = 200000
# 실제 원단위 PGX는 조건수가 커서 일부 출발점에 약 489만 회 평가가 필요했다.
# 아래 값은 계산을 강제로 중단할 때의 자원 한도일 뿐 보폭/오차 허용조건이 아니다.
# H/F 파이프라인은 Settings의 자체 예산을 별도 전달한다.
MAX_EVALUATIONS = 20000000
TRACE_STRIDE = 100
TRACE_COLUMNS = ('iteration','C_IOP','projected_gradient_inf','alpha','line_evaluations',
                 'P','G','X','delta_P','delta_G','delta_X','nfev','step_norm')
HF_TRACE_COLUMNS = ('iteration','C4','gradient_inf','alpha','line_evaluations',
                    'z','F','delta_z','delta_F','nfev','step_norm')
SOLVER_CONFIG = dict(
    solver='nonnegative_pgx_feasible_steepest_descent', gradient='analytic_chain_rule',
    coordinates='original (P,G,X) or (z,F); no variable scaling',
    direction='PGX: negative gradient projected onto feasible tangent cone; HF unchanged',
    update='x_new=x-alpha*feasible_gradient; line interval ends at the first bound',
    line_search='Brent analytic directional-derivative root; Brent scalar minimization fallback',
    line_search_initial_interval='gradient norm squared / residual directional curvature',
    line_search_relative_tolerance=LINE_RTOL,
    directional_derivative_root_tolerance='2*epsilon*abs(alpha)+1ULP(alpha)',
    cost_display_roundoff='epsilon*(1+abs(previous_cost)); never changes recorded cost',
    gradient_tolerance=GRADIENT_TOL,
    max_iterations=MAX_ITERATIONS, max_evaluations=MAX_EVALUATIONS,
    trace_stride=TRACE_STRIDE, trace_first_iterations=20,
    parameter_bounds={'P':[0.0,None], 'G':[0.0,None], 'X':[0.0,None]},
    mathematical_domain='P>0 for real log; G>=0; X>=0; nonzero denominators; finite arithmetic',
    convergence='PGX: feasible projected-gradient/KKT residual; HF: original gradient; no global certificate',
    line_search_guarantee='numerical local line minimum only; no global guarantee',
    scale_argument='diagnostics only; does not affect iterations', other_optimizer_fallback=False)


@njit(cache=True, error_model='numpy')
def _value_jac(pgx, observed, aw, bbw):
    """PGX의 C_IOP, 물리 기울기, 상대잔차 Jacobian, 상대잔차를 계산한다.

    da/dP=a0+a1*(ln(P)+1), da/dG=exp(-S*(λ-440)), dbb/dX=(550/λ)^Y.
    u=bb/(a+bb), Rrs=.52*(g0+g1*u)*u/(1-1.7*(g0+g1*u)*u)를 연쇄 미분한다.
    4밴드 C=mean(r²)이므로 gradient=(1/2)*J.T@r이며 관측값은 분모에 그대로 둔다.
    """
    residual = np.zeros(4)
    jac = np.zeros((4,3))
    gradient = np.zeros(3)
    p,g,x = pgx
    if p <= 0.0 or g < 0.0 or x < 0.0 or not np.all(np.isfinite(pgx)):
        return np.inf,gradient,jac,residual
    log_p = np.log(p)
    for j in range(4):
        shape_g = np.exp(-S*(WAVES[j]-440.0))
        shape_x = (550.0/WAVES[j])**Y
        a = aw[j]+(A0[j]+A1[j]*log_p)*p+g*shape_g
        bb = bbw[j]+x*shape_x
        total = a+bb
        if total == 0.0 or not np.isfinite(total):
            return np.inf,gradient,jac,residual
        u = bb/total
        below = (G0+G1*u)*u
        denominator = 1.0-1.7*below
        if denominator == 0.0:
            return np.inf,gradient,jac,residual
        residual[j] = (0.52*below/denominator-observed[j])/observed[j]
        common = 0.52*(G0+2.0*G1*u)/(denominator*denominator*observed[j]*total*total)
        jac[j,0] = common*(-bb)*(A0[j]+A1[j]*(log_p+1.0))
        jac[j,1] = common*(-bb)*shape_g
        jac[j,2] = common*a*shape_x
    cost = np.sum(residual*residual)/4.0
    for k in range(3):
        for j in range(4):
            gradient[k] += 0.5*jac[j,k]*residual[j]
    if not (np.isfinite(cost) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(jac))):
        return np.inf,gradient,jac,residual
    return cost,gradient,jac,residual


@njit(cache=True, error_model='numpy')
def _hf_value_jac(hf, observed, xc, xb, dp, rho_first, rho_second):
    """z,F의 C4와 해석 미분. F<0/F>1 및 z<0도 수식이 정의되면 그대로 계산한다.

    rho=rho_second+F*(rho_first-rho_second),
    b=dp*(1-exp(-xc*z))+(rho/pi)*exp(-xb*z),
    db/dz=dp*xc*exp(-xc*z)-(rho/pi)*xb*exp(-xb*z),
    db/dF=(rho_first-rho_second)*exp(-xb*z)/pi, dRrs/db=.52/(1-1.7*b)^2.
    """
    residual = np.zeros(4)
    jac = np.zeros((4,2))
    gradient = np.zeros(2)
    if not np.all(np.isfinite(hf)):
        return np.inf,gradient,jac,residual
    z,fraction = hf
    for j in range(4):
        ec,eb = np.exp(-xc[j]*z),np.exp(-xb[j]*z)
        rho = rho_second[j]+fraction*(rho_first[j]-rho_second[j])
        below = dp[j]*(-np.expm1(-xc[j]*z))+(rho/np.pi)*eb
        denominator = 1.0-1.7*below
        if denominator == 0.0 or not np.isfinite(below):
            return np.inf,gradient,jac,residual
        residual[j] = (0.52*below/denominator-observed[j])/observed[j]
        common = 0.52/(denominator*denominator*observed[j])
        jac[j,0] = common*(dp[j]*xc[j]*ec-(rho/np.pi)*xb[j]*eb)
        jac[j,1] = common*((rho_first[j]-rho_second[j])/np.pi)*eb
    cost = np.sum(residual*residual)/4.0
    for k in range(2):
        for j in range(4):
            gradient[k] += 0.5*jac[j,k]*residual[j]
    if not (np.isfinite(cost) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(jac))):
        return np.inf,gradient,jac,residual
    return cost,gradient,jac,residual


@njit(cache=True, error_model='numpy')
def _evaluate(x,data,kind):
    if kind == 0:
        result = _value_jac(x,data[0],data[1],data[2])
    else:
        result = _hf_value_jac(x,data[0],data[1],data[2],data[3],data[4],data[5])
    # data 마지막 행은 이 단일 시작점의 진단 카운터 전용이다. 무효 평가도
    # nfev에 포함하며, 목적함수나 다음 스텝에 카운터를 사용하지 않는다.
    if not np.isfinite(result[0]):
        data[6,0] += 1.0
    return result


@njit(cache=True, error_model='numpy')
def _feasible_gradient(x,gradient,kind):
    """비음수 경계의 허용 최급강하 방향과 KKT 수렴 잔차를 동시에 만든다.

    G/X=0에서 g_i>0이면 -g_i는 음수 영역으로 향하므로 그 성분만 0으로 둔다.
    g_i<0이면 양수 쪽으로 나갈 수 있으므로 경계에서도 계속 이동한다.
    내부점은 원래 기울기와 정확히 같다. 임의의 '경계 근처' 허용폭은 사용하지 않는다.
    P는 로그의 열린 정의역이라 0에 놓이지 않는다. H/F에는 이 제약을 적용하지 않는다.
    """
    feasible = gradient.copy()
    if kind == 0:
        for k in range(1,3):
            if x[k] == 0.0 and gradient[k] > 0.0:
                feasible[k] = 0.0
    return feasible


@njit(cache=True, error_model='numpy')
def _line_limit(x,gradient,kind):
    """x-alpha*g가 허용 영역에 머무르는 최대 alpha를 부등식에서 직접 구한다.

    G/X의 닫힌 경계는 alpha=x_i/g_i까지 허용한다. P=0은 로그가 정의되지 않아
    경계 직전의 표현 가능한 실수까지 허용한다. 이전 alpha나 임의 배수는 쓰지 않는다.
    """
    end = np.inf
    if kind == 0:
        if gradient[0] > 0.0:
            end = np.nextafter(x[0]/gradient[0],0.0)
            while end > 0.0 and x[0]-end*gradient[0] <= 0.0:
                end = np.nextafter(end,0.0)
        for k in range(1,3):
            if gradient[k] > 0.0:
                end = min(end,x[k]/gradient[k])
    return end


@njit(cache=True, error_model='numpy')
def _line_point(alpha,x,gradient,kind):
    """허용 직선 위의 점. 경계 도달 시 곱셈 반올림 오차만 정확한 0으로 정리한다.

    alpha가 x_i/g_i와 정확히 같을 때만 해당 좌표를 0으로 둔다. 범위 밖 추정값을
    사후에 자르는 작업이 아니며, 목적함수 평가와 실제 갱신에 동일한 점을 사용한다.
    """
    point = x-alpha*gradient
    if kind == 0:
        for k in range(1,3):
            if gradient[k] > 0.0 and alpha == x[k]/gradient[k]:
                point[k] = 0.0
    return point


@njit(cache=True, error_model='numpy')
def _line_cost(alpha,x,gradient,data,kind):
    cost,_,_,_ = _evaluate(_line_point(alpha,x,gradient,kind),data,kind)
    return cost


@njit(cache=True, error_model='numpy')
def _line_derivative_root(x,gradient,data,kind,right,budget):
    """φ'가 음수에서 양수로 바뀌는 구간에서 표준 Brent 근을 구한다.

    비용의 마지막 비트가 같아도 해석 방향미분은 더 작은 이동을 구별할 수 있다.
    따라서 φ'=0을 푸는 것은 다른 최적화 방향을 추가하는 것이 아니라 같은
    일변수 선 최소화의 정밀도를 높이는 작업이다. 수렴 구간 폭은 α의 기계 정밀도다.
    """
    a,b = 0.0,right
    fa = -np.sum(gradient*gradient)
    cb,gb,_,_ = _evaluate(_line_point(b,x,gradient,kind),data,kind)
    fb = -np.sum(gb*gradient)
    nfev = 1
    if not np.isfinite(cb) or not np.isfinite(fb):
        return 0.0,np.inf,nfev,False
    # 비용값이 반올림 때문에 먼저 증가해 보이더라도 φ'<0이면 선 최소점을
    # 지나지 않았다. 동일한 표준 황금비 구간 구성으로 도함수 부호 변화를 찾는다.
    domain_end = _line_limit(x,gradient,kind)
    while fb < 0.0 and nfev < budget:
        # 허용 구간의 끝에서도 감소 방향이면 그 끝이 제약된 선 최소점이다.
        # φ'=0이 없다는 이유로 실패 처리하면 G/X=0 해에 도달할 수 없다.
        if b == domain_end:
            return b,cb,nfev,True
        next_b = min(b*GOLDEN_RATIO,domain_end)
        if not np.isfinite(next_b) or next_b <= b:
            return 0.0,np.inf,nfev,False
        b = next_b
        cb,gb,_,_ = _evaluate(_line_point(b,x,gradient,kind),data,kind)
        nfev += 1
        fb = -np.sum(gb*gradient)
        if not np.isfinite(cb) or not np.isfinite(fb):
            return 0.0,np.inf,nfev,False
    c,fc = a,fa
    d,e = b-a,b-a
    while nfev < budget:
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c,fc = a,fa
            d,e = b-a,b-a
        if abs(fc) < abs(fb):
            old_b,old_fb = b,fb
            a,fa = old_b,old_fb
            b,fb = c,fc
            c,fc = old_b,old_fb
        tolerance = 2.0*EPS*abs(b)+abs(np.nextafter(b,np.inf)-b)
        midpoint = (c-b)/2.0
        if abs(midpoint) <= tolerance or fb == 0.0:
            final_cost = _line_cost(b,x,gradient,data,kind)
            return b,final_cost,nfev+1,True
        if abs(e) >= tolerance and abs(fa) > abs(fb):
            s = fb/fa
            if a == c:
                p,q = 2.0*midpoint*s,1.0-s
            else:
                q,r = fa/fc,fb/fc
                p = s*(2.0*midpoint*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0*p < min(3.0*midpoint*q-abs(tolerance*q),abs(e*q)):
                e,d = d,p/q
            else:
                d,e = midpoint,midpoint
        else:
            d,e = midpoint,midpoint
        a,fa = b,fb
        if abs(d) > tolerance:
            b += d
        else:
            b += tolerance if midpoint > 0.0 else -tolerance
        cb,gb,_,_ = _evaluate(_line_point(b,x,gradient,kind),data,kind)
        nfev += 1
        fb = -np.sum(gb*gradient)
        if not np.isfinite(cb) or not np.isfinite(fb):
            return 0.0,np.inf,nfev,False
    return 0.0,np.inf,nfev,False


@njit(cache=True, error_model='numpy')
def _line_minimize(x,cost,gradient,jac,data,kind,budget):
    """음의 기울기 반직선에서 구간을 구성한 뒤 Brent로 일변수 최소점을 찾는다.

    잔차 1차 Taylor 모형의 선 최소점 α0=||g||²/(.5*||Jg||²)를 첫 간격으로 쓴다.
    이는 최종 스텝이 아니며 Gauss-Newton 방향도 아니다. 방향은 항상 -g다.
    비용이 다시 증가할 때까지 황금비로 구간을 확장한 후 포물선 보간과 황금분할을
    사용한다. 감소량 임의 계수나 직전 반복의 학습률은 사용하지 않는다.
    반환 ok=False는 계산 예산 소진 또는 표현 정밀도에서의 선탐색 실패다.
    """
    g2 = np.sum(gradient*gradient)
    curvature = 0.0
    for j in range(4):
        jg = 0.0
        for k in range(len(x)):
            jg += jac[j,k]*gradient[k]
        curvature += 0.5*jg*jg
    if not np.isfinite(g2) or curvature <= 0.0 or not np.isfinite(curvature):
        return 0.0,cost,0,False
    seed = g2/curvature
    if seed <= 0.0 or not np.isfinite(seed) or budget < 1:
        return 0.0,cost,0,False
    # P>0과 G/X>=0에서 유도한 허용 선 구간만 탐색한다.
    domain_end = _line_limit(x,gradient,kind)
    seed = min(seed,domain_end)
    if seed <= 0.0 or not np.any(x-seed*gradient != x):
        return 0.0,cost,0,False
    fseed = _line_cost(seed,x,gradient,data,kind)
    nfev = 1
    left,right = 0.0,seed
    best,fbest = 0.0,cost
    if fseed < cost:
        best,fbest = seed,fseed
        while right < domain_end:
            if nfev >= budget:
                return best,fbest,nfev,False
            new_right = min(right*GOLDEN_RATIO,domain_end)
            if not np.isfinite(new_right) or new_right <= right:
                return best,fbest,nfev,False
            fnew = _line_cost(new_right,x,gradient,data,kind)
            nfev += 1
            previous = right
            right = new_right
            if fnew >= fbest:
                break
            left = previous/GOLDEN_RATIO if previous != seed else 0.0
            best,fbest = right,fnew
    # 해석 방향미분을 이용한 Brent 근으로 같은 선 최소점을 먼저 정밀 계산한다.
    # 상대잔차는 예측/관측에서 1을 빼므로 작은 비용에도 O(eps)의 절대 반올림이
    # 남는다. eps*(1+C)는 이 무차원 계산의 표시 정밀도 검사이며 수렴 기준은 아니다.
    if nfev < budget-1:
        root,root_cost,root_calls,root_ok = _line_derivative_root(
            x,gradient,data,kind,right,budget-nfev)
        nfev += root_calls
        if root_ok and root > 0.0 and root_cost <= cost+EPS*(1.0+abs(cost)):
            return root,root_cost,nfev,True
    # scalar_x,w,v는 선 위 스칼라 위치이며 최저 비용의 세 보간점을 보관한다.
    scalar_x = best
    # 0만 유한한 경우에도 0을 incumbent으로 둔다. inf<=inf로 비교하며 유한한
    # 왼쪽 구간을 버리는 오류를 막고, 표준 황금분할이 우측 무효 구간을 수축시킨다.
    fx = fbest
    w,v = scalar_x,scalar_x
    fw,fv = fx,fx
    previous_step,proposed_step = 0.0,0.0
    while nfev < budget:
        midpoint = (left+right)/2.0
        # sqrt(eps) 상대 정밀도와 해당 α의 1 ULP를 쓴다. 단위 있는 임의 xatol은 없다.
        ulp = abs(np.nextafter(scalar_x,np.inf)-scalar_x)
        tolerance = LINE_RTOL*abs(scalar_x)+ulp
        tolerance2 = 2.0*tolerance
        if (abs(scalar_x-midpoint) <= tolerance2-(right-left)/2.0
                or np.nextafter(left,right) >= right):
            return best,fbest,nfev,best > 0.0 and fbest < cost
        use_parabola = False
        if abs(previous_step) > tolerance and np.isfinite(fx+fw+fv):
            r = (scalar_x-w)*(fx-fv)
            q = (scalar_x-v)*(fx-fw)
            p = (scalar_x-v)*q-(scalar_x-w)*r
            q = 2.0*(q-r)
            if q > 0.0:
                p = -p
            q = abs(q)
            saved_step = previous_step
            previous_step = proposed_step
            if (q > 0.0 and abs(p) < abs(q*saved_step)/2.0
                    and p > q*(left-scalar_x) and p < q*(right-scalar_x)):
                proposed_step = p/q
                u = scalar_x+proposed_step
                if u-left < tolerance2 or right-u < tolerance2:
                    proposed_step = tolerance if scalar_x < midpoint else -tolerance
                use_parabola = True
        if not use_parabola:
            previous_step = right-scalar_x if scalar_x < midpoint else left-scalar_x
            proposed_step = GOLDEN_COMPLEMENT*previous_step
        if abs(proposed_step) >= tolerance:
            u = scalar_x+proposed_step
        else:
            u = scalar_x+(tolerance if proposed_step > 0.0 else -tolerance)
        if u == scalar_x:
            u = np.nextafter(scalar_x,right if scalar_x < midpoint else left)
        fu = _line_cost(u,x,gradient,data,kind)
        nfev += 1
        if fu < fbest:
            best,fbest = u,fu
        if fu <= fx:
            if u >= scalar_x:
                left = scalar_x
            else:
                right = scalar_x
            v,fv = w,fw
            w,fw = scalar_x,fx
            scalar_x,fx = u,fu
        else:
            if u < scalar_x:
                left = u
            else:
                right = u
            if fu <= fw or w == scalar_x:
                v,fv = w,fw
                w,fw = u,fu
            elif fu <= fv or v == scalar_x or v == w:
                v,fv = u,fu
    return best,fbest,nfev,False


@njit(cache=True, error_model='numpy')
def _solve(start,data,kind,max_iterations,max_evaluations,gradient_tol,trace_stride):
    x = start.copy()
    cost,gradient,jac,residual = _evaluate(x,data,kind)
    nfev,nit,status = 1,0,0
    dimension = len(x)
    trace = np.zeros((max_iterations//trace_stride+24,7+2*dimension))
    state = np.zeros(7+2*dimension)
    state[1] = cost
    state[2] = np.max(np.abs(_feasible_gradient(x,gradient,kind))) if np.isfinite(cost) else np.inf
    state[5:5+dimension] = x
    state[-2] = nfev
    trace[0] = state
    ntrace = 1
    if not np.isfinite(cost):
        return x,cost,jac,residual,-4,nit,nfev,np.inf,trace[:ntrace]
    for iteration in range(1,max_iterations+1):
        feasible = _feasible_gradient(x,gradient,kind)
        optimality = np.max(np.abs(feasible))
        if optimality <= gradient_tol:
            status = 1
            break
        if nfev >= max_evaluations:
            status = -2
            break
        alpha,trial_cost,line_evals,ok = _line_minimize(
            x,cost,feasible,jac,data,kind,max_evaluations-nfev-1)
        nfev += line_evals
        if not ok:
            status = -2 if nfev >= max_evaluations-1 else -3
            break
        trial = _line_point(alpha,x,feasible,kind)
        if not np.any(trial != x):
            status = -3
            break
        new_cost,new_gradient,new_jac,new_residual = _evaluate(trial,data,kind)
        nfev += 1
        if not np.isfinite(new_cost) or new_cost > cost+EPS*(1.0+abs(cost)):
            status = -3
            break
        delta = trial-x
        x,cost = trial,new_cost
        gradient,jac,residual = new_gradient,new_jac,new_residual
        nit = iteration
        optimality = np.max(np.abs(_feasible_gradient(x,gradient,kind)))
        state[0:5] = np.array([float(iteration),cost,optimality,alpha,float(line_evals)])
        state[5:5+dimension] = x
        state[5+dimension:5+2*dimension] = delta
        state[-2] = nfev
        state[-1] = np.sqrt(np.sum(delta*delta))
        if iteration <= 20 or iteration % trace_stride == 0 or optimality <= gradient_tol:
            trace[ntrace] = state
            ntrace += 1
    optimality = np.max(np.abs(_feasible_gradient(x,gradient,kind)))
    if optimality <= gradient_tol:
        status = 1
    if trace[ntrace-1,0] != nit:
        trace[ntrace] = state
        ntrace += 1
    return x,cost,jac,residual,status,nit,nfev,optimality,trace[:ntrace]


def _result(raw,invalid_evaluations):
    x,cost,jac,residual,status,nit,nfev,optimality,trace = raw
    messages = {1:'physical_gradient_stationary',0:'maximum_iterations',
                -2:'maximum_evaluations',-3:'line_search_or_numerical_stagnation',
                -4:'undefined_or_nonfinite_initial_model'}
    return SimpleNamespace(x=x,cost=cost,jac=jac,fun=residual,status=status,
        success=status==1,message=messages[status],nit=nit,nfev=nfev,
        njev=nfev,optimality=optimality,trace=trace,
        invalid_evaluations=int(invalid_evaluations))


def _controls(max_iterations,max_evaluations,gradient_tol,trace_stride):
    values = (MAX_ITERATIONS if max_iterations is None else int(max_iterations),
              MAX_EVALUATIONS if max_evaluations is None else int(max_evaluations),
              float(gradient_tol),int(trace_stride))
    if min(values[0],values[1],values[3]) < 1 or not np.isfinite(values[2]) or values[2] <= 0:
        raise ValueError('Computation budgets, trace stride and tolerance must be positive')
    return values


def _inputs(start,bands,dimension):
    x = np.asarray(start,dtype=np.float64)
    arrays = [np.asarray(v,dtype=np.float64) for v in bands]
    if x.shape != (dimension,) or not np.isfinite(x).all():
        raise ValueError('Starting point has wrong dimension or nonfinite values')
    if any(v.shape != (4,) or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Exactly four finite bands are required')
    if np.any(arrays[0] == 0):
        raise ValueError('Relative residual requires nonzero observations')
    data = np.zeros((7,4),dtype=np.float64)
    for row,array in enumerate(arrays):
        data[row] = array
    return x,data


def solve_pgx(start,observed,aw,bbw,scale,*,max_iterations=MAX_ITERATIONS,
              max_evaluations=MAX_EVALUATIONS,gradient_tol=GRADIENT_TOL,
              trace_stride=TRACE_STRIDE):
    """한 PGX 시작점을 독립 실행한다. scale은 호환/진단용이며 궤적에 관여하지 않는다.

    유효 시작점은 P>0, G>=0, X>=0이다. 위반 입력은 임의로 바꾸지 않고 거절한다.
    내부에서는 원래 -gradient, 경계에서는 허용 접선 방향으로 이동한다.
    최종 optimality와 이력의 projected_gradient_inf는 경계 KKT 잔차이며,
    경계에서 0일 필요가 없는 원래 기울기의 최댓값은 raw_gradient_inf에 따로 둔다.
    a_phi나 총 a/bb의 별도 부호 하한과 P_MIN은 추가하지 않는다.
    """
    x,data = _inputs(start,(observed,aw,bbw),3)
    if x[0] <= 0.0 or np.any(x < 0.0):
        raise ValueError('PGX must be nonnegative; P must be strictly positive for ln(P)')
    diagnostic_scale = np.asarray(scale,dtype=np.float64)
    if (diagnostic_scale.shape != (3,) or not np.isfinite(diagnostic_scale).all()
            or np.any(diagnostic_scale <= 0)):
        raise ValueError('Diagnostic scale must contain three positive finite values')
    raw = _solve(x,data,0,*_controls(
        max_iterations,max_evaluations,gradient_tol,trace_stride))
    result = _result(raw,data[6,0])
    result.raw_gradient_inf = float(np.max(np.abs(0.5*result.jac.T@result.fun)))
    result.projected_gradient_inf = result.optimality
    if result.success:
        result.message = 'nonnegative_projected_gradient_stationary'
    return result


def solve_hf(start,observed,xc,xb,dp,rho_first,rho_second,*,
             max_iterations=MAX_ITERATIONS,max_evaluations=MAX_EVALUATIONS,
             gradient_tol=GRADIENT_TOL,trace_stride=TRACE_STRIDE):
    """한 (z,F) 시작점. 매 이동은 원래 z/F 단위의 -∇C4와 평행하다."""
    x,data = _inputs(start,(observed,xc,xb,dp,rho_first,rho_second),2)
    raw = _solve(x,data,1,*_controls(
        max_iterations,max_evaluations,gradient_tol,trace_stride))
    return _result(raw,data[6,0])
