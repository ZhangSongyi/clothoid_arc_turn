from .scene import Scene
import numpy as np
import numpy.typing as npt
import math

def findMidControlPose(raw_entering_pose: npt.NDArray[np.float64],
                       raw_leaving_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Find middle control pose for a dual-clothoid arc turn curve.

    Args:
        raw_entering_pose: the overall entering pose.
        raw_leaving_pose: the overall leaving pose.

    Raises:
        AssertionError: raw_entering_pose & raw_leaving_pose are not suitable for the case.

    Returns:
        The middle control pose between two clothoid arc turn.
    """
    x_diff = raw_leaving_pose[0] - raw_entering_pose[0]
    y_diff = raw_leaving_pose[1] - raw_entering_pose[1]
    dual_t = 0.25 * math.sqrt(x_diff**2 + y_diff**2)
    dual_theta  = math.atan2(y_diff, x_diff)

    dual_xi_a = raw_entering_pose[2] - dual_theta
    dual_xi_b = raw_leaving_pose[2] - dual_theta

    while dual_xi_a >= math.pi:
        dual_xi_a -= 2 * math.pi
    while dual_xi_a < -math.pi:
        dual_xi_a += 2 * math.pi
    while dual_xi_b >= math.pi:
        dual_xi_b -= 2 * math.pi
    while dual_xi_b < -math.pi:
        dual_xi_b += 2 * math.pi

    if dual_xi_a * dual_xi_b < 0:
        raise AssertionError('Entering & leaving poses are not suitable for dual scene case')

    dual_mean_xi = 0.5 * (dual_xi_a + dual_xi_b)
    dual_delta_xi = 0.5 * (dual_xi_a - dual_xi_b)

    if dual_delta_xi == 0:
        # full symmetry
        mid_control_pose = np.array([
            raw_entering_pose[0] + x_diff / 2,
            raw_entering_pose[1] + y_diff / 2,
            dual_theta - dual_mean_xi
        ], dtype=np.float64)
    else:
        # ignore O(delta_xi^2)
        mean_delta = -dual_mean_xi

        # solve delta_a and delta_b
        delta_a = mean_delta - 0.5 * dual_delta_xi
        delta_b = -mean_delta - 0.5 * dual_delta_xi

        # solve t_a and t_b
        t_a = -2 * dual_t * math.sin(mean_delta + dual_mean_xi - dual_delta_xi / 2) / math.sin(dual_delta_xi)
        t_b = 2 * dual_t * math.sin(mean_delta + dual_mean_xi + dual_delta_xi / 2) / math.sin(dual_delta_xi)

        mid_control_pose = np.array([
            raw_entering_pose[0] + 2 * t_a * math.cos(mean_delta + dual_mean_xi + dual_delta_xi / 2 + dual_theta),
            raw_entering_pose[1] + 2 * t_a * math.sin(mean_delta + dual_mean_xi + dual_delta_xi / 2 + dual_theta),
            raw_entering_pose[2] + 2 * delta_a
        ], dtype=np.float64)
    return mid_control_pose

def solveDualScene(raw_entering_pose: npt.NDArray[np.float64],
                   raw_leaving_pose: npt.NDArray[np.float64],
                   lmbda: float,
                   vehicle_half_width: float = 1,
                   vehicle_base_front: float = 4,
                   same_lmbda: bool = False,
                   use_first_curvature: bool = False) -> tuple[Scene, Scene]:
    """Generate fully controled dual-clothoid arc turn curve.

    Args:
        raw_entering_pose: the overall entering pose.
        raw_leaving_pose: the overall leaving pose.
        lmbda: lmbda parameter for at least one of the clothoid arc turn.
        vehicle_half_width: Half width of the vehicle.
        vehicle_base_front: Distance between the rear axle to the front hang.
        same_lmbda: If set to True, then two clothoid arc turns will have same lmbda. Defaults to False.
        use_first_curvature: Only valid if same_lmbda = False. If set to True, then lmbda is used to control the first curve, and the second curve will keep the same curvature with it. If set to False, then lmbda is used to control the second curve, and the first curve will keep the same curvature with it. Defaults to False.

    Returns:
        Two Sence object, representing the two clothoid arc turns.
    """
    mid_control_pose = findMidControlPose(raw_entering_pose, raw_leaving_pose)
    scene_a = Scene(raw_entering_pose, mid_control_pose, lmbda, vehicle_half_width, vehicle_base_front, True)
    scene_b = Scene(mid_control_pose, raw_leaving_pose, lmbda, vehicle_half_width, vehicle_base_front, True)
    if not same_lmbda:
        if use_first_curvature:
            scene_b.setCurvature(-scene_a.curvature())
        else:
            scene_a.setCurvature(-scene_b.curvature())
    return scene_a, scene_b
