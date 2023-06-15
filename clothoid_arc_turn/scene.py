import numpy as np
from .solve import *
from .evaluate import *
import inspect

class Scene:
    """A scene to storage all the parameters of the clothoid arc turn."""

    # =============
    # PROPS | INPUT
    # =============
    def rawEnteringPose(self) -> npt.NDArray[np.float64]:
        return self.raw_entering_pose
    def rawLeavingPose (self) -> npt.NDArray[np.float64]:
        return self.raw_leaving_pose
    # ===========================
    # PROPS | SPECIAL PATH POINTS
    # ===========================
    def enteringPoint        (self) -> npt.NDArray[np.float64]:
        return self.triangle_params.entering_point
    def arcEnteringPoseK(self) -> npt.NDArray[np.float64]:
        return self.center_curve.arc_entering_pose_k
    def arcMiddlePoseK  (self) -> npt.NDArray[np.float64]:
        return self.center_curve.arc_middle_pose_k
    def arcLeavingPoseK (self) -> npt.NDArray[np.float64]:
        return self.center_curve.curcular_leaving_pose_k
    def leavingPoint         (self) -> npt.NDArray[np.float64]:
        return self.triangle_params.leaving_point
    # ==================
    # PROPS | AUX POINTS
    # ==================
    def triangleVertex       (self) -> npt.NDArray[np.float64]:
        return self.triangle_params.triangle_vertex_point
    def chordMiddlePoint     (self) -> npt.NDArray[np.float64]:
        return self.triangle_params.chord_middle_point
    def arcCenterPoint  (self) -> npt.NDArray[np.float64]:
        return self.center_curve.arc_center_point
    # ==============
    # PROPS | PARAMS
    # ==============
    def isInferiorArc  (self) -> bool:
        return abs(self.triangle_params.delta) <= math.pi / 2
    def delta          (self) -> float:
        return self.triangle_params.delta
    def triangleT      (self) -> float:
        return self.triangle_params.triangle_t
    def triangleN      (self) -> float:
        return self.triangle_params.triangle_n
    def circleN        (self) -> float:
        return self.turn_params.entering_circle_n
    def curvature      (self) -> float:
        return self.turn_params.curvature
    def minCurvature   (self) -> float:
        if abs(self.estimated_params1.curvature) >= abs(self.estimated_params0.curvature):
            return self.estimated_params0.curvature
        else:
            return self.estimated_params1.curvature
    def maxCurvature   (self) -> float:
        if abs(self.estimated_params1.curvature) >= abs(self.estimated_params0.curvature):
            return self.estimated_params1.curvature
        else:
            return self.estimated_params0.curvature
    def minLmbda       (self) -> float:
        return self.triangle_params.min_lmbda
    def maxLmbda       (self) -> float:
        return min(self.max_lmbda, self.triangle_params.turn_max_lmbda)
    # ============
    # PROPS | PATH
    # ============
    def fullCurve      (self) -> npt.NDArray[np.float64]:
        return self.center_curve.full_curve
    def outerBorder    (self) -> npt.NDArray[np.float64]:
        if self.delta() > 0:
            return self.border_segments.right_front_border
        else:
            return self.border_segments.left_front_border
    def innerBorder    (self) -> npt.NDArray[np.float64]:
        if self.delta() > 0:
            return self.border_segments.left_middle_border
        else:
            return self.border_segments.right_middle_border
    # ================
    # PROPS | ESTIMATE
    # ================
    def estimatedCircleN     (self) -> float:
        return self.estimated_params.circle_n
    def estimatedCurvature   (self) -> float:
        return self.estimated_params.curvature
    def estimatedCirclePoint (self) -> npt.NDArray[np.float64]:
        return self.triangle_params.chord_middle_point + self.estimated_params.circle_n * self.triangle_params.triangle_n_direction
    def estinatedCirclePointRange (self) -> npt.NDArray[np.float64]:
        return np.array([
            self.triangle_params.chord_middle_point + self.estimated_params0.circle_n * self.triangle_params.triangle_n_direction,
            self.triangle_params.chord_middle_point + self.estimated_params1.circle_n * self.triangle_params.triangle_n_direction
        ], dtype=np.float64)
    def estimatedCircleTangent(self)-> npt.NDArray[np.float64]:
        return np.array([
            self.estimatedCirclePoint() + self.triangleT() * (1-self.estimatedCircleN() / self.triangleN()) * self.triangle_params.triangle_t_direction,
            self.estimatedCirclePoint() - self.triangleT() * (1-self.estimatedCircleN() / self.triangleN()) * self.triangle_params.triangle_t_direction
        ], dtype=np.float64)
    def estimatedOuterPoint  (self) -> npt.NDArray[np.float64]:
        return self.triangle_params.chord_middle_point + self.geometry_params.outer_n * self.triangle_params.triangle_n_direction
    def estimatedInnerPoint  (self) -> npt.NDArray[np.float64]:
        return self.triangle_params.chord_middle_point + self.geometry_params.inner_n * self.triangle_params.triangle_n_direction
    def estimatedInnerN      (self) -> float:
        return self.geometry_params.inner_n
    def estimatedOuterN      (self) -> float:
        return self.geometry_params.outer_n
    def outerCircleBorder    (self) -> npt.NDArray[np.float64]:
        circle_center_point = self.arcCenterPoint()
        radius = abs(1 / self.curvature() + self.turn_params.leaving_circle_n - self.geometry_params.outer_n)
        heading_margin = abs(0.2 / radius)
        circle_points: list[npt.NDArray[np.float64]] = []
        for t in np.arange(0, 2 * math.pi, heading_margin):
            circle_points.append(circle_center_point + radius * np.array([math.cos(t), math.sin(t)], dtype=np.float64))
        return np.array(circle_points)
    def curvatureProfile     (self) -> npt.NDArray[np.float64]:
        return np.array([
            [0, 0],
            [self.turn_params.entering_ls, self.turn_params.curvature],
            [self.turn_params.entering_ls + self.turn_params.entering_lc, self.turn_params.curvature],
            [self.turn_params.entering_ls + self.turn_params.entering_lc + self.turn_params.leaving_lc, self.turn_params.curvature],
            [self.turn_params.entering_ls + self.turn_params.entering_lc + self.turn_params.leaving_lc + self.turn_params.leaving_ls, 0],
            [self.turn_params.entering_ls + self.turn_params.entering_lc + self.turn_params.leaving_lc + self.turn_params.leaving_ls + self.triangle_params.leaving_ll, 0]
        ], dtype=np.float64)

    # ======
    # EVENTS
    # ======
    def onPoseUpdated(self) -> None:
        self.onAuxUpdated()
    def onAuxUpdated (self) -> None:
        self.turn_params = solveClothoidArcTurnParams(
                                    self.triangle_params.delta,
                                    self.triangle_params.triangle_t,
                                    self.lmbda,
                                    self.triangle_params.delta_phi)
        self.estimated_params = solveEstimatedClothoidArcTurnParams(
                                    self.triangle_params.delta,
                                    self.triangle_params.triangle_t,
                                    self.triangle_params.triangle_n,
                                    self.triangle_params.delta_phi,
                                    self.lmbda)
        self.estimated_params0 = solveEstimatedClothoidArcTurnParams(
                                    self.triangle_params.delta,
                                    self.triangle_params.triangle_t,
                                    self.triangle_params.triangle_n,
                                    self.triangle_params.delta_phi,
                                    self.triangle_params.min_lmbda)
        self.estimated_params1 = solveEstimatedClothoidArcTurnParams(
                                    self.triangle_params.delta,
                                    self.triangle_params.triangle_t,
                                    self.triangle_params.triangle_n,
                                    self.triangle_params.delta_phi,
                                    min(self.max_lmbda, self.triangle_params.turn_max_lmbda))
        self.geometry_params = solveVehicleGeometryRelatedParams(
                                    self.estimated_params.curvature,
                                    self.estimated_params.circle_n,
                                    self.vehicle_half_width,
                                    self.vehicle_base_front)
        self.center_curve = getCenterCurve(
                                    self.raw_entering_pose,
                                    self.triangle_params.entering_ll,
                                    self.turn_params.entering_ls,
                                    self.turn_params.entering_lc,
                                    self.turn_params.leaving_lc,
                                    self.turn_params.leaving_ls,
                                    self.triangle_params.leaving_ll,
                                    self.turn_params.curvature,
                                    0.2)
        self.border_segments = getBorderSegments(self.center_curve.full_curve, self.vehicle_half_width, self.vehicle_base_front)
    # =======
    # METHODS
    # =======
    def setEnteringPose(self, new_entering_pose: npt.NDArray[np.float64])-> bool:
        new_trianlge_params = solveTrianlgeParams(new_entering_pose, self.raw_leaving_pose, self.symmetry, self.max_lmbda)
        if not new_trianlge_params:
            return False
        if self.lmbda < new_trianlge_params.min_lmbda:
            self.lmbda = new_trianlge_params.min_lmbda
        if self.lmbda > min(self.max_lmbda, new_trianlge_params.turn_max_lmbda):
            self.lmbda = min(self.max_lmbda, new_trianlge_params.turn_max_lmbda)
        self.triangle_params = new_trianlge_params
        self.raw_entering_pose = new_entering_pose
        self.onPoseUpdated()
        return True
    def setLeavingPose(self, new_leaving_pose: npt.NDArray[np.float64])  -> bool:
        new_trianlge_params = solveTrianlgeParams(self.raw_entering_pose, new_leaving_pose, self.symmetry, self.max_lmbda)
        if not new_trianlge_params:
            return False
        if self.lmbda < new_trianlge_params.min_lmbda:
            self.lmbda = new_trianlge_params.min_lmbda
        if self.lmbda > min(self.max_lmbda, new_trianlge_params.turn_max_lmbda):
            self.lmbda = min(self.max_lmbda, new_trianlge_params.turn_max_lmbda)
        self.triangle_params = new_trianlge_params
        self.raw_leaving_pose = new_leaving_pose
        self.onPoseUpdated()
        return True
    def setVertexPoint(self, new_vertex_point: npt.NDArray[np.float64], keep_delta: float = False) -> bool:
        new_entering_dx = new_vertex_point[0] - self.raw_entering_pose[0]
        new_entering_dy = new_vertex_point[1] - self.raw_entering_pose[1]
        new_entering_theta = math.atan2(new_entering_dy, new_entering_dx)
        new_leaving_dx = self.raw_leaving_pose[0] - new_vertex_point[0]
        new_leaving_dy = self.raw_leaving_pose[1] - new_vertex_point[1]
        new_leaving_theta = math.atan2(new_leaving_dy, new_leaving_dx)
        new_entering_pose = self.raw_entering_pose
        new_leaving_pose = self.raw_leaving_pose
        if keep_delta:
            delta_change = new_entering_theta - new_entering_pose[2]
            new_entering_pose[2] = new_entering_theta
            new_leaving_pose[2] += delta_change
        else:
            new_entering_pose[2] = new_entering_theta
            new_leaving_pose[2] = new_leaving_theta
        new_trianlge_params = solveTrianlgeParams(new_entering_pose, new_leaving_pose, self.symmetry, self.max_lmbda)
        if not new_trianlge_params:
            return False
        if self.lmbda < new_trianlge_params.min_lmbda:
            self.lmbda = new_trianlge_params.min_lmbda
        if self.lmbda > min(self.max_lmbda, new_trianlge_params.turn_max_lmbda):
            self.lmbda = min(self.max_lmbda, new_trianlge_params.turn_max_lmbda)
        self.triangle_params = new_trianlge_params
        self.raw_entering_pose = new_entering_pose
        self.raw_leaving_pose = new_leaving_pose
        self.onPoseUpdated()
        return True
    def setLmbda      (self, new_lmbda: float)              -> bool:
        if new_lmbda < self.triangle_params.min_lmbda or new_lmbda > min(self.max_lmbda, self.triangle_params.turn_max_lmbda):
            return False
        self.lmbda = new_lmbda
        self.onAuxUpdated()
        return True
    def setMaxLmbda   (self, new_max_lmbda: float)          -> bool:
        new_trianlge_params = solveTrianlgeParams(
            self.raw_entering_pose,
            self.raw_leaving_pose,
            self.symmetry,
            new_max_lmbda)
        if not new_trianlge_params:
            return False
        if self.lmbda < new_trianlge_params.min_lmbda:
            self.lmbda = new_trianlge_params.min_lmbda
        if self.lmbda > min(new_max_lmbda, new_trianlge_params.turn_max_lmbda):
            self.lmbda = min(new_max_lmbda, new_trianlge_params.turn_max_lmbda)
        self.triangle_params = new_trianlge_params
        self.max_lmbda = new_max_lmbda
        self.onPoseUpdated()
        return True

    def setCircleN    (self, new_circle_n: float)   -> bool:
        new_lmbda = solveLmbdaByCircleNNewton(
            self.triangle_params.delta,
            self.triangle_params.triangle_t,
            self.triangle_params.triangle_n,
            self.triangle_params.delta_phi,
            self.triangle_params.min_lmbda,
            new_circle_n)
        result = self.setLmbda(new_lmbda)
        return result
    def setCurvature  (self, new_curvature: float)  -> bool:
        new_lmbda = solveLmbdaByCurvatureNewton(
            self.triangle_params.delta,
            self.triangle_params.triangle_t,
            self.triangle_params.delta_phi,
            self.triangle_params.min_lmbda,
            new_curvature)
        result = self.setLmbda(new_lmbda)
        return result

    def __init__(self, entering_pose: npt.NDArray[np.float64], leaving_pose: npt.NDArray[np.float64],
                 lmbda: float,
                 vehicle_half_width: float = 1, vehicle_base_front: float = 4,
                 symmetry: bool = True, max_lmbda = 1) -> None:
        """Initialize of the scene

        Args:
            entering_pose: Entering position and orientation
            leaving_pose: Leaving position and orientation
            lmbda: Turn ratio between the clothoid segment and the arc arc segment. lmbda vs 1-lmbda
            vehicle_half_width: Half width of the vehicle.
            vehicle_base_front: Distance between the rear axle to the front hang.
            symmetry: Generate symmetry turn or not. Defaults to True.
        """
        self.raw_entering_pose = np.array(entering_pose, dtype=np.float64)
        self.raw_leaving_pose = np.array(leaving_pose, dtype=np.float64)
        self.lmbda = lmbda
        self.vehicle_half_width = vehicle_half_width
        self.vehicle_base_front = vehicle_base_front
        self.symmetry = symmetry
        self.max_lmbda = max_lmbda
        new_trianlge_params = solveTrianlgeParams(self.raw_entering_pose, self.raw_leaving_pose, self.symmetry, self.max_lmbda)
        if not new_trianlge_params:
            raise AssertionError("can't find feasible config under initial config")
        if self.lmbda < new_trianlge_params.min_lmbda:
            self.lmbda = new_trianlge_params.min_lmbda
        if self.lmbda > min(max_lmbda, new_trianlge_params.turn_max_lmbda):
            self.lmbda = min(max_lmbda, new_trianlge_params.turn_max_lmbda)
        self.triangle_params = new_trianlge_params
        self.onPoseUpdated()
