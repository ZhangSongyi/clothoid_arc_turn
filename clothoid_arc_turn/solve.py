import scipy.special as sc
import math
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from .fresnel_trigonometry import fresnelDe, fresnelDAndDe, inverseFresnelDeCNewton, inverseFresnelDeTNewton, fresnelDeWithDerivatives, fresnelMcDsWithLmbdaAndDeltaDeltaDerivatives, fresnelMcDsWithDeltaDeltaDerivatives2
from typing import Optional
import typing
import numpy.typing as npt

# left = dS / mC; right = tan(delta_delta - delta_phi)
def solveDeltaDeltaNewton(delta: float, lmbda: float, delta_phi: float, halley: bool = False) -> float:
    if lmbda == 0:
        return 0
    current_delta_delta = 0
    delta_delta_change = 1
    iter = 0
    while abs(delta_delta_change) > 1e-3:
        if current_delta_delta == 0:
            res = delta_phi
            fds, fdc, fdse, fdce = fresnelDAndDe(delta, lmbda)
            dres = 1 / (fdce / fds / lmbda)
            new_delta_delta = current_delta_delta - res / dres
        else:
            if not halley:
                mC, dS, _, _, mC_delta, dS_delta = fresnelMcDsWithLmbdaAndDeltaDeltaDerivatives(delta, current_delta_delta, lmbda)
                # left = math.atan(dS / mC)
                # right = current_delta_delta - delta_phi
                # dleft = (mC * dS_delta - dS * mC_delta) / (dS**2 + mC ** 2)
                # dright = 1
                left = math.atan(dS / mC)
                right = current_delta_delta - delta_phi
                dleft = (mC * dS_delta - dS * mC_delta) / (dS**2 + mC ** 2)
                dright = 1
                res = left-right
                # fix error in angle calc
                if res > math.pi:
                    res -= math.pi
                if res < -math.pi:
                    res += math.pi
                dres = dleft - dright
                new_delta_delta = current_delta_delta - res / dres
            else:
                mC, dS, mC_delta, dS_delta, mC_delta2, dS_delta2 = fresnelMcDsWithDeltaDeltaDerivatives2(delta, current_delta_delta, lmbda)
                # 0th
                left = math.atan(dS / mC)
                right = current_delta_delta - delta_phi
                res = left-right
                # 1st
                sqr = dS ** 2 + mC ** 2
                parta = mC * dS_delta
                partb = dS * mC_delta
                dleft = (parta - partb) / (sqr)
                dright = 1
                dres = dleft - dright
                # 2nd
                sqr_delta = 2 * (dS * dS_delta + mC * mC_delta)
                parta_delta = mC * dS_delta2 + mC_delta * dS_delta
                partb_delta = dS * mC_delta2 + dS_delta * mC_delta
                ddleft = (sqr * (parta_delta - partb_delta) - sqr_delta * (parta - partb)) / (sqr**2)
                ddright = 0
                ddres = ddleft - ddright
                new_delta_delta = current_delta_delta - (2 * res * dres) / (2 * dres * dres - res * ddres)
        if abs(new_delta_delta) > abs(delta):
            if new_delta_delta * delta > 0:
                new_delta_delta = delta
            else:
                new_delta_delta = -delta
        delta_delta_change = new_delta_delta - current_delta_delta
        current_delta_delta = new_delta_delta
        iter+=1
    # print(iter, end=', ', flush=True)
    return current_delta_delta

@dataclass_json
@dataclass
class TrianlgeParams:
    """Parameters of the envelope triangle"""
    delta                 : float
    """Half turning angle between entering pose and leaving pose"""
    entering_ll           : float
    """Length of the extra entering straight line, happens when symmetry==True and entering triangle edge is longer"""
    leaving_ll            : float
    """Length of the extra leaving straight line, happens when symmetry==True and leaving triangle edge is longer"""
    entering_point        : npt.NDArray[np.float64]
    """Last point of the extra entering straight line"""
    leaving_point         : npt.NDArray[np.float64]
    """First point of the extra leaving straight line"""
    chord_middle_point    : npt.NDArray[np.float64]
    """Middle point of ex_entering_point and ex_leaving_point"""
    triangle_vertex_point : npt.NDArray[np.float64]
    """Vertex point of the envelope triangle"""
    delta_phi             : float
    """Half of the difference of phi"""
    triangle_t            : float
    """Half length between entering pose and leaving pose"""
    triangle_n            : float
    """Distance between triangle_vertex and chord_middle_point"""
    min_lmbda             : float
    """Estimated lower bound of the parameter lambda"""
    triangle_n_direction  : npt.NDArray[np.float64]
    """Direction of the vector from chord_middle_point to triangle_vertex (delta < 0) or its reversed version (delta > 0)"""
    triangle_t_direction  : npt.NDArray[np.float64]
    """Direction of the vector from entering point to the leaving point"""
    turn_max_lmbda: float
    """Max lmbda limitiation to keep fresnelTanE less than 5"""
def solveTrianlgeParams(raw_entering_pose: npt.NDArray[np.float64],
                        raw_leaving_pose: npt.NDArray[np.float64],
                        symmetry: bool = True, input_max_lmbda: float = 1) -> Optional[TrianlgeParams]:
    """Generate the basic parameters of the problem.

    mainly about the envelope triangle of the clothoid arc turn

    Args:
        raw_entering_pose: Entering position and orientation
        raw_leaving_pose: Leaving position and orientation
        symmetry: Generate symmetry turn or not

    Returns:
        Parameters of the envelope triangle
    """
    x_diff: float = raw_leaving_pose[0] - raw_entering_pose[0]
    y_diff: float = raw_leaving_pose[1] - raw_entering_pose[1]
    heading_diff: float = raw_leaving_pose[2] - raw_entering_pose[2]
    turn_max_lmbda = 1.0

    # clamp heading_diff to [-pi, pi]
    while heading_diff > math.pi:
        heading_diff -= 2 * math.pi
    while heading_diff <= -math.pi:
        heading_diff += 2 * math.pi

    raw_entering_triangle_edge = (x_diff * math.sin(raw_leaving_pose[2]) - y_diff * math.cos(raw_leaving_pose[2])) / math.sin(heading_diff)
    raw_leaving_triangle_edge = -(x_diff * math.sin(raw_entering_pose[2]) - y_diff * math.cos(raw_entering_pose[2])) / math.sin(heading_diff)
    triangle_vertex = np.array([
        raw_entering_pose[0] + raw_entering_triangle_edge * math.cos(raw_entering_pose[2]),
        raw_entering_pose[1] + raw_entering_triangle_edge * math.sin(raw_entering_pose[2]),
    ], dtype=np.float64)

    if raw_entering_triangle_edge <= 0 or raw_leaving_triangle_edge <= 0:
        # rotation > pi
        if heading_diff < 0:
            heading_diff += 2 * math.pi
        else:
            heading_diff -= 2 * math.pi


    delta = heading_diff / 2
    # find turn max lmbda
    if abs(delta) > math.pi / 2:
        # check fdte at 1.0
        fdse1, fdce1 = fresnelDe(delta, 1)
        fdte1 = fdse1 / fdce1
        if (abs(fdte1) > 5 and fdte1 * delta > 0) or (fdte1 * delta < 0):
            target_fdte = 5 if delta > 0 else -5
            turn_max_lmbda = inverseFresnelDeTNewton(delta, target_fdte)

    if symmetry:
        triangle_edge_min = min(raw_entering_triangle_edge, raw_leaving_triangle_edge)
        entering_ll = raw_entering_triangle_edge - triangle_edge_min
        leaving_ll = raw_leaving_triangle_edge - triangle_edge_min
        delta_phi = 0
        min_lmbda = 0
    else:
        entering_ll = 0
        leaving_ll = 0

        # max theta for only one clothoid
        safe_heading_diff = heading_diff
        single_fds, single_fdc = fresnelDe(safe_heading_diff, min(input_max_lmbda, turn_max_lmbda))
        max_phi = math.atan(single_fds/single_fdc)
        if max_phi * safe_heading_diff < 0:
            if safe_heading_diff > 0:
                max_phi += math.pi
            else:
                max_phi -= math.pi

        chord_orientation = math.atan2(y_diff, x_diff)
        entering_phi = chord_orientation - raw_entering_pose[2]
        while entering_phi > math.pi:
            entering_phi -= 2 * math.pi
        while entering_phi <= -math.pi:
            entering_phi += 2 * math.pi

        leaving_phi = raw_leaving_pose[2] - chord_orientation
        while leaving_phi > math.pi:
            leaving_phi -= 2 * math.pi
        while leaving_phi <= -math.pi:
            leaving_phi += 2 * math.pi

        if abs(entering_phi) > abs(max_phi):
            # cut leaving part
            entering_phi = max_phi
            leaving_phi = heading_diff - max_phi

            new_leaving_edge = abs(raw_entering_triangle_edge * math.sin(entering_phi) / math.sin(leaving_phi))
            leaving_ll = raw_leaving_triangle_edge - new_leaving_edge

        if abs(leaving_phi) > abs(max_phi):
            # cut entering part
            leaving_phi = max_phi
            entering_phi = heading_diff - max_phi

            new_entering_edge = abs(raw_leaving_triangle_edge * math.sin(leaving_phi) / math.sin(entering_phi))
            entering_ll = raw_entering_triangle_edge - new_entering_edge

        delta_phi = (entering_phi - leaving_phi) / 2

        # solve min_lmbda
        if abs(entering_phi) > abs(leaving_phi):
            min_lmbda = inverseFresnelDeTNewton(safe_heading_diff, math.tan(entering_phi))
        else:
            min_lmbda = inverseFresnelDeTNewton(safe_heading_diff, math.tan(leaving_phi))

        if min_lmbda >= min(input_max_lmbda, turn_max_lmbda):
            return None

    entering_point = np.array([
        raw_entering_pose[0] + entering_ll * math.cos(raw_entering_pose[2]),
        raw_entering_pose[1] + entering_ll * math.sin(raw_entering_pose[2]),
    ], dtype=np.float64)
    leaving_point = np.array([
        raw_leaving_pose[0] - leaving_ll * math.cos(raw_leaving_pose[2]),
        raw_leaving_pose[1] - leaving_ll * math.sin(raw_leaving_pose[2]),
    ], dtype=np.float64)
    chord_middle_point = (entering_point + leaving_point) / 2
    triangle_t_direction = leaving_point - entering_point
    triangle_t = np.linalg.norm(triangle_t_direction) / 2
    triangle_t_direction /= 2 * triangle_t
    triangle_n_direction = triangle_vertex - chord_middle_point
    triangle_n = np.linalg.norm(triangle_n_direction)
    triangle_n_direction /= triangle_n

    if delta > 0:
        triangle_n *= -1
        triangle_n_direction *= -1

    return TrianlgeParams(
        delta                = delta,
        entering_ll          = entering_ll,
        leaving_ll           = leaving_ll,
        entering_point       = entering_point,
        leaving_point        = leaving_point,
        chord_middle_point   = chord_middle_point,
        triangle_vertex_point= triangle_vertex,
        delta_phi            = delta_phi,
        min_lmbda            = min_lmbda,
        triangle_t           = float(triangle_t),
        triangle_n           = float(triangle_n),
        triangle_t_direction = triangle_t_direction,
        triangle_n_direction = triangle_n_direction,
        turn_max_lmbda       = turn_max_lmbda
    )

@dataclass_json
@dataclass
class ClothoidArcTurnParams:
    """Parameters of the clothoid arc turn"""
    entering_ls: float
    """Length of the entering clothoid segment."""
    entering_lc: float
    """Length of the entering arc arc segment."""
    leaving_lc: float
    """Length of the leaving arc arc segment."""
    leaving_ls: float
    """Length of the leaving clothoid segment."""
    entering_sharpness: float
    """Curvature change rate (a.k.a sharpness) of the entering clothoid segment."""
    leaving_sharpness: float
    """Curvature change rate (a.k.a sharpness) of the leaving clothoid segment."""
    entering_circle_n: float
    """Distance between the circle and the entering part path chord."""
    leaving_circle_n: float
    """Distance between the circle and the leaving part path chord."""
    curvature: float
    """Curvature of the arc arc segment."""
    delta_delta: float
    """Skewness of the entering part and the leaving part. Ratio of the turning angle is 1+rho vs 1-rho"""
def solveClothoidArcTurnParams(delta: float, triangle_t: float, lmbda: float, delta_phi: float = 0) -> ClothoidArcTurnParams:
    """_summary_

    Args:
        delta: half turn
        triangle_t: half chord length
        lmbda: Turn ratio between the clothoid segment and the arc arc segment. lmbda vs 1-lmbda
        half_phi_diff: Half of the difference of phi

    Returns:
        Parameters of the clothoid arc turn
    """
    # fallback case1: pure straight line
    if delta == 0:
        return ClothoidArcTurnParams(
            entering_ls=0,
            entering_lc=0,
            leaving_lc=0,
            leaving_ls=0,
            entering_sharpness=0,
            leaving_sharpness=0,
            entering_circle_n=0,
            leaving_circle_n=0,
            curvature=0,
            delta_delta=0
        )

    # fallback case2: pure circle
    if lmbda == 0:
        # when half_phi_diff != 0, there're no feasible solution under lmbda == 0
        return ClothoidArcTurnParams(
            entering_ls=0,
            entering_lc=triangle_t * delta / math.sin(delta),
            leaving_lc=triangle_t * delta / math.sin(delta),
            leaving_ls=0,
            entering_sharpness=np.sign(delta)*float('inf'),
            leaving_sharpness=-np.sign(delta)*float('inf'),
            entering_circle_n=-(math.tan(delta/2)) * triangle_t,
            leaving_circle_n=-(math.tan(delta/2)) * triangle_t,
            curvature=math.sin(delta)/triangle_t,
            delta_delta=0
        )

    # helper elements
    fdse, fdce = fresnelDe(delta, lmbda)
    fdte = fdse / fdce

    if delta_phi == 0:
        # symmetric case
        ls = lmbda * (2 * delta) * triangle_t / fdce
        lc = (1 - lmbda) / 2 * (2 * delta) * triangle_t / fdce
        curvature = fdce / triangle_t
        sharpness = curvature / ls
        circle_n = - fdte * triangle_t
        return ClothoidArcTurnParams(
            entering_ls=ls, entering_lc=lc,
            leaving_lc=lc, leaving_ls=ls,
            entering_sharpness=sharpness, leaving_sharpness=-sharpness,
            entering_circle_n=circle_n, leaving_circle_n=circle_n,
            curvature=curvature, delta_delta=0
        )
    else:
        delta_delta = solveDeltaDeltaNewton(delta, lmbda, delta_phi)

        if abs(delta_delta) >= abs(delta):
            delta_delta = (abs(delta) - 1e-6) * (1 if delta_delta > 0 else -1)

        entering_delta = delta + delta_delta
        leaving_delta = delta - delta_delta

        # print(entering_delta, leaving_delta)

        # helper params
        entering_fdse, entering_fdce = fresnelDe(entering_delta, lmbda)
        leaving_fdse, leaving_fdce = fresnelDe(leaving_delta, lmbda)

        t_sum = 2 * triangle_t * math.cos(delta_phi - delta_delta)
        entering_t_fdce = t_sum / (entering_fdce + leaving_fdce)
        leaving_t_fdce = t_sum / (entering_fdce + leaving_fdce)

        entering_ls = lmbda * (2 * entering_delta) * entering_t_fdce
        entering_lc = (1 - lmbda) / 2 * (2 * entering_delta) * entering_t_fdce
        entering_curvature = 1 / entering_t_fdce
        entering_sharpness = entering_curvature / entering_ls
        entering_circle_n = - (entering_fdse) * entering_t_fdce

        leaving_ls = lmbda * (2 * leaving_delta) * leaving_t_fdce
        leaving_lc = (1 - lmbda) / 2 * (2 * leaving_delta) * leaving_t_fdce
        leaving_curvature = 1 / leaving_t_fdce
        leaving_sharpness = -leaving_curvature / leaving_ls
        leaving_circle_n = - (leaving_fdse) * leaving_t_fdce

        return ClothoidArcTurnParams(
            entering_ls        = entering_ls,
            entering_lc        = entering_lc,
            leaving_lc         = leaving_lc,
            leaving_ls         = leaving_ls,
            entering_sharpness = entering_sharpness,
            leaving_sharpness  = leaving_sharpness,
            entering_circle_n  = entering_circle_n,
            leaving_circle_n   = leaving_circle_n,
            curvature          = entering_curvature,
            delta_delta        = delta_delta
        )

@dataclass_json
@dataclass
class EstimatedClothoidArcTurnParams:
    """Estimated parameters with its derivatives."""
    circle_n: float
    """Estimated distance between the circle and the input chord. Not applicable for unsymmetry case."""
    curvature: float
    """Estimated curvature of the path."""
    circle_n_d_lmbda: float
    """dy / dlambda"""
    curvature_d_lmbda: float
    """dcurvature / dlambda"""
def solveEstimatedClothoidArcTurnParams(delta: float, triangle_t: float, triangle_n: float, delta_phi: float, lmbda: float) -> EstimatedClothoidArcTurnParams:
    """Estimate parameters of the path with its derivatives.

    Args:
        delta: half turn
        triangle_t: half chord length
        triangle_n: distance between middle point and vertex point
        lmbda: Turn ratio between the clothoid segment and the arc arc segment. lmbda vs 1-lmbda

    Returns:
        Estimated parameters with its derivatives.
    """
    efdse, efdce, efdse_d_lmbda, efdce_d_lmbda = fresnelDeWithDerivatives(delta, lmbda)
    curvature = 1 / triangle_t * efdce
    curvature_d_lmbda = 1 / triangle_t * efdce_d_lmbda

    efdte = efdse / efdce
    efdte_d_lmbda = (-efdse * efdce_d_lmbda + efdce * efdse_d_lmbda) / (efdce ** 2)

    scaling_factor = math.cos((math.pi / 2 - (1 - lmbda)**2) * delta_phi / delta)
    scaling_factor_d_lmbda = -math.sin((math.pi / 2 - (1 - lmbda)**2) * delta_phi / delta) * 2 * (1 - lmbda) * delta_phi / delta


    circle_n = triangle_n / math.tan(delta) * efdte * scaling_factor
    circle_n_d_lmbda = triangle_n / math.tan(delta) * (efdte_d_lmbda * scaling_factor + efdte * scaling_factor_d_lmbda)

    return EstimatedClothoidArcTurnParams(
        circle_n = circle_n, curvature = curvature,
        circle_n_d_lmbda=circle_n_d_lmbda, curvature_d_lmbda=curvature_d_lmbda)

@dataclass_json
@dataclass
class VehicleGeometryRelatedParams:
    """Inner distance and outer distance of the coverage region."""
    inner_n: float
    """Distance between the coverage region inner point and the input chord."""
    outer_n: float
    """Distance between the coverage region outer point and the input chord."""
def solveVehicleGeometryRelatedParams(curvature: float, circle_n: float, vehicle_half_width: float, vehicle_base_front: float) -> VehicleGeometryRelatedParams:
    """Generate inner distance and outer distance of the coverage region.

    Args:
        curvature: Max curvature of the path
        circle_n: Distance between the circle and the input chord. Not applicable for unsymmetry case.
        vehicle_half_width: Half width of the vehicle.
        vehicle_base_front: Distance between the rear axle to the front hang.

    Returns:
        Inner distance and outer distance of the coverage region.
    """
    sign = 1 if curvature > 0 else -1
    inner_n = circle_n + sign * vehicle_half_width
    outer_n = circle_n - (sign * math.sqrt((1/ curvature + sign * vehicle_half_width)**2 + vehicle_base_front**2) - 1/curvature)
    return VehicleGeometryRelatedParams(inner_n=inner_n, outer_n=outer_n)

def solveLmbdaByCurvatureNewton(delta: float, triangle_t: float, delta_phi: float, min_lmbda: float, curvature: float) -> float:
    """Solve lambda parameter to fit curvature

    Args:
        delta: half turn
        triangle_t: half chord length
        curvature: Target curvature

    Returns:
        Estimated lambda parameter
    """
    kt = triangle_t * curvature
    current_lmbda = inverseFresnelDeCNewton(delta, kt)
    if delta_phi == 0 or current_lmbda == 0:
        return current_lmbda
    else:
        # SOLVE r1: mC - kt * cos(delta_delta - delta_phi) = 0
        # SOLVE r2: dS - kt * sin(delta_delta - delta_phi) = 0

        res = delta_phi
        fds, fdc, fdse, fdce = fresnelDAndDe(delta, current_lmbda)
        dres = 1 / (fdce / fds / current_lmbda)
        current_delta_delta = -res / dres
        lmbda_change = 1
        delta_delta_change = 1
        iters = 0
        while abs(lmbda_change) > 1e-3 or abs(delta_delta_change) > 1e-3:
            mC, dS, mC_lmbda, dS_lmbda, mC_delta, dS_delta = fresnelMcDsWithLmbdaAndDeltaDeltaDerivatives(delta, current_delta_delta, current_lmbda)
            r1 = mC - kt * math.cos(current_delta_delta - delta_phi)
            r2 = dS - kt * math.sin(current_delta_delta - delta_phi)
            r1l = mC_lmbda
            r2l = dS_lmbda
            r1d = mC_delta + kt * math.sin(current_delta_delta - delta_phi)
            r2d = dS_delta - kt * math.cos(current_delta_delta - delta_phi)
            # solution for inverse Jacobian
            new_lmbda = current_lmbda + (r1d*r2 - r1*r2d)/(r1l*r2d - r1d*r2l)
            new_delta_delta = current_delta_delta + (r1l*r2 - r1*r2l)/(-(r1l*r2d) + r1d*r2l)
            if new_lmbda > 1:
                new_lmbda = 1
            if new_lmbda < min_lmbda:
                new_lmbda = min_lmbda
            if abs(new_delta_delta) > abs(delta):
                if new_delta_delta * delta > 0:
                    new_delta_delta = delta
                else:
                    new_delta_delta = -delta
            # print(f'lmbda={current_lmbda}, delta={delta}, delta_delta={current_delta_delta}, r1={r1}, r2={r2}')
            # print(r1, r2, '{{' + f'{r1l}, {r1d}' + '},{' + f'{r2l}, {r2d}' + '}}')
            # print(f'lmbda: {current_lmbda} -> {new_lmbda}')
            # print(f'delta: {current_delta_delta} -> {new_delta_delta}')
            delta_delta_change = new_delta_delta - current_delta_delta
            lmbda_change = new_lmbda - current_lmbda
            current_delta_delta = new_delta_delta
            current_lmbda = new_lmbda

            iters += 1
            if iters > 100:
                raise Exception()

        return current_lmbda
def solveLmbdaByCircleNNewton(delta: float, triangle_t: float, triangle_n: float, delta_phi: float, min_lmbda: float, circle_n: float) -> float:
    """Solve lambda parameter to fit circle_n. Not applicable for unsymmetry case.

    Args:
        delta: half turn
        t: half chord length
        helper_params: Helper parameters to make the estimation
        circle_n: Target circle distance to chord point.

    Returns:
        Estimated lambda parameter.
    """
    # circle_n = triangle_n / math.tan(delta) * efdte * scaling_factor
    # scaling_factor = math.cos((math.pi / 2 - (1 - lmbda)**2) * delta_phi / delta)
    if triangle_n == 0:
        return 1.0
    current_lmbda = inverseFresnelDeTNewton(delta, circle_n / triangle_n * math.tan(delta))
    if delta_phi == 0:
        return current_lmbda
    else:
        # SOLVE circle_n = triangle_n / math.tan(delta) * efdte * scaling_factor
        # KNOWN circle_n, triangle_n, delta
        k = triangle_n / math.tan(delta)
        lmbda_change = 1
        while abs(lmbda_change) > 1e-3:
            estimated_params = solveEstimatedClothoidArcTurnParams(delta, triangle_t, triangle_n, delta_phi, current_lmbda)
            if current_lmbda >= 1 and abs(circle_n) > abs(estimated_params.circle_n):
                return 1
            residual = circle_n - estimated_params.circle_n
            residual_d_lmbda = - estimated_params.circle_n_d_lmbda
            lmbda_change = - residual / residual_d_lmbda
            new_lmbda = current_lmbda + lmbda_change
            if new_lmbda < min_lmbda:
                new_lmbda = min_lmbda
            if new_lmbda > 1:
                new_lmbda = 1
            lmbda_change = new_lmbda - current_lmbda
            current_lmbda = new_lmbda
        return current_lmbda
