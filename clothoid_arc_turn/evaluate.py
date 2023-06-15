"""Evaluate clothoid arc turn

This module samples the clothoid arc turn curve states based on the exported parameters.
By providing the initial state, the length of each segment and the curvature, the module will represent the whole turn curve in state vectors with position, orientation and curvature.
Meanwhile, the module can generate the boundary of the vehicle's sweeping area by giving the vehicle shape.
"""

import scipy.special as sc
import math
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import scipy.optimize

def segmentSample(initial: npt.NDArray[np.float64], sharpness: float, l: float) -> npt.NDArray[np.float64]:
    """Get a state at specific station in a clothoid segment

    Args:
        initial: Initial state of the segment, in the form (x, y, orientation, curvature).
        sharpness: Curvature change rate (a.k.a sharpness) of the segment.
                   For arc arc and straight line, this value should be 0.
        l: Station to sample the state.

    Returns:
        State at station l, in the form (x, y, orientation, curvature).
    """
    if sharpness != 0:
        # Clothoid
        curvature = initial[3] + sharpness * l
        t = initial[2] - initial[3] * initial[3] / 2 / sharpness
        eta1 = curvature / math.sqrt(math.pi * abs(sharpness))
        eta0 = initial[3] / math.sqrt(math.pi * abs(sharpness))
        s1, c1 = sc.fresnel(eta1)
        s0, c0 = sc.fresnel(eta0)
        if sharpness < 0:
            s1 *= -1
            s0 *= -1
        dx = ((c1-c0) * math.cos(t) -
              (s1-s0) * math.sin(t)) * math.sqrt(math.pi / abs(sharpness))
        dy = ((c1-c0) * math.sin(t) +
              (s1-s0) * math.cos(t)) * math.sqrt(math.pi / abs(sharpness))
        if sharpness < 0:
            dx *= -1
            dy *= -1
        return np.array([
            initial[0] + dx,
            initial[1] + dy,
            initial[2] + initial[3] * l + sharpness * l * l / 2,
            initial[3] + sharpness * l
        ], dtype=np.float64)

    else:
        # Straight or arc arc
        if initial[3] != 0:
            orientation = initial[2] + l * initial[3]
            return np.array([
                initial[0] + (math.sin(orientation) - math.sin(initial[2])) / initial[3],
                initial[1] - (math.cos(orientation) - math.cos(initial[2])) / initial[3],
                orientation,
                initial[3]
            ], dtype=np.float64)
        else:
            # Straight segment
            return np.array([
                initial[0] + l * math.cos(initial[2]),
                initial[1] + l * math.sin(initial[2]),
                initial[2],
                0
            ], dtype=np.float64)

def segmentFullSample(initial: npt.NDArray[np.float64], sharpness: float, length: float, margin: float) -> npt.NDArray[np.float64]:
    """Sample all the states in a clothoid segment

    Args:
        initial: Initial state of the segment, in the form (x, y, orientation, curvature)
        sharpness: Curvature change rate (a.k.a sharpness) of the segment.
                           For arc arc and straight line, this value should be 0.
        length: Length of this segment.
        margin: Margin between the sampled points. It will be slightly adjust to ensure the first
                        and the last sampled states are exactly on station 0 and length.

    Returns:
        Sampled points, each in the form (x, y, orientation, curvature).
    """
    if length == 0:
        return np.array([initial], dtype=np.float64)
    else:
        states = []
        states.append(initial)
        nsegments = math.ceil(length / margin)
        real_margin = length / nsegments
        for i in range(nsegments):
            states.append(segmentSample(initial, sharpness, (i+1) * real_margin))
        return np.array(states, dtype=np.float64)

@dataclass_json
@dataclass
class CenterCurveSegments:
    """Parts of the generated curve."""
    entering_straight_part : npt.NDArray[np.float64]
    """States in the entering straight segment."""
    entering_clothoid_part   : npt.NDArray[np.float64]
    """States in the entering clothoid segment."""
    entering_arc_part : npt.NDArray[np.float64]
    """States in the entering arc arc segment."""
    arc_part          : npt.NDArray[np.float64]
    """States in both entering and leaving arc arc segment."""
    leaving_arc_part  : npt.NDArray[np.float64]
    """States in the leaving arc arc segment."""
    leaving_clothoid_part    : npt.NDArray[np.float64]
    """States in the leaving clothoid segment."""
    leaving_straight_part  : npt.NDArray[np.float64]
    """States in the leaving straight segment."""
    full_curve             : npt.NDArray[np.float64]
    """States in the whole path."""
    arc_entering_pose_k: npt.NDArray[np.float64]
    """First point of the entering arc segment."""
    arc_middle_pose_k : npt.NDArray[np.float64]
    """Junction point of the two arc segments."""
    curcular_leaving_pose_k: npt.NDArray[np.float64]
    """Last point of the leaving arc segment."""
    arc_center_point  : npt.NDArray[np.float64]
    """Circle center of the arc segments"""
def getCenterCurve(entering_pose: npt.NDArray[np.float64],
             entering_ll: float, entering_ls: float, entering_lc: float,
             leaving_lc: float, leaving_ls: float, leaving_ll: float,
             max_curvature: float, margin: float) -> CenterCurveSegments:
    """Generate full sample of a clothoid-arc turn

    Args:
        entering_pose: The entering pose of the turn.
        entering_ll: Length of the entering straight segment.
        entering_ls: Length of the entering clothoid segment.
        entering_lc: Length of the entering arc arc segment.
        leaving_lc: Length of the leaving arc arc segment.
        leaving_ls: Length of the leaving clothoid segment.
        leaving_ll: Length of the leaving straight segment.
        max_curvature: Curvature of the arc arc segment.
        margin: Sampled margin.

    Returns:
        Properties of the generated path.
    """
    # first half points
    initial_state = np.append(entering_pose[:3], 0)
    entering_straight_part = segmentFullSample(initial_state, 0, entering_ll, margin)
    entering_straight_part_end = np.append(entering_straight_part[-1][:3], 0)
    if entering_ls == 0:
        entering_clothoid_part = np.array([entering_straight_part_end], dtype=np.float64)
    else:
        entering_clothoid_part = segmentFullSample(entering_straight_part_end, max_curvature / entering_ls, entering_ls, margin)
    entering_clothoid_part_end = np.append(entering_clothoid_part[-1][:3], max_curvature)
    entering_arc_part = segmentFullSample(entering_clothoid_part_end, 0, entering_lc, margin)
    leaving_arc_part = segmentFullSample(entering_arc_part[-1], 0, leaving_lc, margin)
    if leaving_ls == 0:
        leaving_clothoid_part = np.array([leaving_arc_part[-1]], dtype=np.float64)
    else:
        leaving_clothoid_part = segmentFullSample(leaving_arc_part[-1], -max_curvature / leaving_ls, leaving_ls, margin)
    leaving_clothoid_part_end = np.append(leaving_clothoid_part[-1][:3], 0)
    leaving_straight_part = segmentFullSample(leaving_clothoid_part_end, 0, leaving_ll, margin)

    full_path = np.concatenate([
            entering_straight_part[:-1],
            entering_clothoid_part[:-1],
            entering_arc_part[:-1],
            leaving_arc_part[:-1],
            leaving_clothoid_part[:-1],
            leaving_straight_part
        ])

    arc_center_point = entering_arc_part[-1][0:2] + 1 / entering_arc_part[-1][3] * np.array([
            math.cos(entering_arc_part[-1][2] + math.pi / 2),
            math.sin(entering_arc_part[-1][2] + math.pi / 2)
        ], dtype=np.float64)

    return CenterCurveSegments(
        entering_straight_part = entering_straight_part,
        entering_clothoid_part   = entering_clothoid_part,
        entering_arc_part = entering_arc_part,
        arc_part          = np.concatenate([entering_arc_part[:-1], leaving_arc_part]),
        leaving_arc_part  = leaving_arc_part,
        leaving_clothoid_part    = leaving_clothoid_part,
        leaving_straight_part  = leaving_straight_part,
        full_curve              = full_path,
        arc_entering_pose_k= entering_arc_part[0],
        arc_middle_pose_k  = entering_arc_part[-1],
        curcular_leaving_pose_k = leaving_arc_part[-1],
        arc_center_point  = arc_center_point
    )

@dataclass_json
@dataclass
class BorderSegments:
    """Trails of special points of the vehicle when following the clothoid arc turn."""
    left_front_border  : npt.NDArray[np.float64]
    """Trail of the left front point of the vehicle. Generally, it is the outer boundary of the vehicle's coverage region when turning right."""
    right_front_border : npt.NDArray[np.float64]
    """Trail of the right front point of the vehicle. Generally, it is the outer boundary of the vehicle's coverage region when turning left."""
    left_middle_border : npt.NDArray[np.float64]
    """Trail of the left rear wheel point of the vehicle. Generally, it is the inner boundary of the vehicle's coverage region when turning left."""
    right_middle_border: npt.NDArray[np.float64]
    """Trail of the right rear wheel point of the vehicle. Generally, it is the inner boundary of the vehicle's coverage region when turning right."""
def getBorderSegments(center_path: npt.NDArray[np.float64], vehicle_half_width: float, vehicle_base_front: float) -> BorderSegments:
    """Generate border trails for the vehicle when following the center path.

    Args:
        center_path: Reference path for the vehicle, aligned with the center of its rear axle.
        vehicle_half_width: Half width of the vehicle.
        vehicle_base_front: Distance between the rear axle to the front hang.

    Returns:
        Border trails for the vehicle when following center_path.
    """
    lf_points = []
    rf_points = []
    lm_points = []
    rm_points = []
    for pose in center_path:
        lf_points.append([
            pose[0] + vehicle_base_front * math.cos(pose[2]) + vehicle_half_width * math.cos(pose[2] + math.pi / 2),
            pose[1] + vehicle_base_front * math.sin(pose[2]) + vehicle_half_width * math.sin(pose[2] + math.pi / 2)
        ])
        rf_points.append([
            pose[0] + vehicle_base_front * math.cos(pose[2]) - vehicle_half_width * math.cos(pose[2] + math.pi / 2),
            pose[1] + vehicle_base_front * math.sin(pose[2]) - vehicle_half_width * math.sin(pose[2] + math.pi / 2)
        ])
        lm_points.append([
            pose[0] + vehicle_half_width * math.cos(pose[2] + math.pi / 2),
            pose[1] + vehicle_half_width * math.sin(pose[2] + math.pi / 2)
        ])
        rm_points.append([
            pose[0] - vehicle_half_width * math.cos(pose[2] + math.pi / 2),
            pose[1] - vehicle_half_width * math.sin(pose[2] + math.pi / 2)
        ])
    return BorderSegments(
        left_front_border=np.array(lf_points),
        right_front_border=np.array(rf_points),
        left_middle_border=np.array(lm_points),
        right_middle_border=np.array(rm_points)
    )

class ProjectionCalculator:
    def __init__(self,
                 delta: float, triangle_t: float,delta_phi: float,
                 entering_ls: float, entering_lc: float,
                 leaving_lc: float, leaving_ls: float,
                 max_curvature: float) -> None:
        self.base_x = triangle_t
        self.midline_direction = math.atan2(math.cos(2 * delta) - math.cos(2 * delta_phi), -math.sin(2 * delta_phi))
        phi_0 = delta + delta_phi
        phi_1 = delta - delta_phi
        theta_0 = -phi_0
        theta_1 = phi_1
        self.s0 = np.array([0, 0, theta_0, 0])
        self.has_entering_part = entering_ls != 0
        self.has_leaving_part = leaving_ls != 0
        if self.has_entering_part:
            self.entering_sharpness = max_curvature / entering_ls
            self.sm1 = segmentSample(self.s0, self.entering_sharpness, entering_ls)
        else:
            self.sm1 = self.s0
            self.sm1[3] = max_curvature
            self.entering_sharpness = float('inf')
        self.sm2 = segmentSample(self.sm1, 0, entering_lc)
        self.sm3 = segmentSample(self.sm2, 0, leaving_lc)
        if self.has_leaving_part:
            self.leaving_sharpness = -max_curvature / leaving_ls
            self.s1 = segmentSample(self.sm3, self.leaving_sharpness, leaving_ls)
        else:
            self.s1 = self.sm3
            self.s1[3] = 0
            self.leaving_sharpness = -float('inf')
        self.stations = [
            entering_ls,
            entering_ls + entering_lc,
            entering_ls + entering_lc + leaving_lc,
            entering_ls + entering_lc + leaving_lc + leaving_ls
        ]
        self.max_station = self.stations[-1]

    def stateAtStation(self, station: float) -> npt.NDArray[np.float64]:
        if station <= 0:
            return self.s0
        if station >= self.stations[3]:
            return self.s1
        if self.has_leaving_part and station > self.stations[2]:
            return segmentSample(self.sm3, self.leaving_sharpness, station - self.stations[2])
        if station > self.stations[1]:
            return segmentSample(self.sm2, 0, station - self.stations[1])
        if station > self.stations[0]:
            return segmentSample(self.sm1, 0, station - self.stations[0])
        if self.has_entering_part:
            return segmentSample(self.s0, self.entering_sharpness, station)
        return self.s0

    def nErrAtStation(self, station: float) -> float:
        state = self.stateAtStation(station)
        err_x = state[0] - self.base_x
        err_y = state[1]
        dot = err_x * (-math.sin(self.midline_direction)) + err_y * math.cos(self.midline_direction)
        return dot

    def distanceAtStation(self, station: float) -> float:
        state = self.stateAtStation(station)
        err_x = state[0] - self.base_x
        err_y = state[1]
        dis = math.sqrt(err_x**2 + err_y**2)
        return dis

def getCircleN(delta: float, triangle_t: float,delta_phi: float,
               entering_ls: float, entering_lc: float,
               leaving_lc: float, leaving_ls: float,
               max_curvature: float) -> float:
    calculator = ProjectionCalculator(delta, triangle_t, delta_phi, entering_ls, entering_lc, leaving_lc, leaving_ls, max_curvature)
    if delta_phi == 0:
        optim_station = calculator.max_station / 2
    else:
        optim_station, _ = scipy.optimize.bisect(lambda s: calculator.nErrAtStation(s), 0, calculator.max_station, xtol=1e-6, full_output=True)
    raw_distance = calculator.distanceAtStation(optim_station)
    if delta > 0:
        raw_distance *= -1
    return raw_distance
