import tkinter as tk
import tkinter.messagebox
import tkinter.ttk
import math
import numpy as np
import numpy.typing as npt
from typing import Optional
from .scene import Scene
from .dump_scene import dumpSceneDescription

class InteractiveVisualizer:
    """An interactive visualizer to manipuate the clothoid arc turn scene
    """
    SCALE = 25
    ORIENTATION_LINE_LENGTH = 1.0

    def moveElement(self, element_id: int, new_center_point: npt.NDArray[np.float64]) -> None:
        elem_x1, elem_y1, elem_x2, elem_y2 = self.cv.bbox(element_id)
        old_x = (elem_x1 + elem_x2) / 2
        old_y = (elem_y1 + elem_y2) / 2
        new_x = new_center_point[0] * InteractiveVisualizer.SCALE + 50
        new_y = 550 - new_center_point[1] * InteractiveVisualizer.SCALE
        # [0,10]x[0,10] -> [50,550]x[50,550]
        self.cv.move(element_id, new_x - old_x, new_y - old_y)
    def replaceLine(self, element_id: int, new_path: Optional[npt.NDArray[np.float64]], **kwargs) -> int:
        if element_id != -1:
            self.cv.delete(element_id)
        if new_path is None:
            return -1
        line_params: list[float] = []
        for state in new_path:
            line_params.append(state[0] * InteractiveVisualizer.SCALE + 50)
            line_params.append(550 - state[1] * InteractiveVisualizer.SCALE)
        return self.cv.create_line(line_params, **kwargs)
    def updateSliderInterval(self, slider: tk.Scale, variable: tk.DoubleVar, slider_min: float, slider_max: float, slider_margin: float = 0.001):
        # ensure min and max reachable for slider
        ticks = math.ceil((slider_max - slider_min) / slider_margin)
        real_slider_margin = (slider_max - slider_min) / ticks
        tick_interval = (slider_max - slider_min) / 5

        # tkinter slider will automaticly round to 0.01
        if math.floor(variable.get() * 100) < math.ceil(slider_min * 100):
            # print(f'variable set to {math.ceil(slider_min * 100) / 100} ({slider_min}, {slider_max})')
            variable.set(math.ceil(slider_min * 100) / 100)
        if math.ceil(variable.get() * 100) > math.floor(slider_max * 100):
            # print(f'variable set to {math.floor(slider_max * 100) / 100} ({slider_min}, {slider_max})')
            variable.set(math.floor(math.floor(slider_max * 100) / 100))
        slider.config(from_=slider_min, to=slider_max + 1e-6, resolution=real_slider_margin, tickinterval=tick_interval)
        # print(f'{slider_min}, {slider_max}, {variable.get()}')
        pass

    def initEnteringPoseElement(self) -> None:
        self.CONTROL_ENTERING_POINT       = self.cv.create_rectangle(0, 0, 10, 10, activefill='red')
        self.CONTROL_ENTERING_ORIENTATION = self.cv.create_oval(0, 0, 10, 10, activefill='yellow')
        aux_circle_radius = InteractiveVisualizer.ORIENTATION_LINE_LENGTH * InteractiveVisualizer.SCALE
        self.AUX_ENTERING_CIRCLE          = self.cv.create_oval(0, 0, aux_circle_radius * 2, aux_circle_radius * 2, dash=(1,1), state=tk.DISABLED, outline='gray50')
        self.AUX_ENTERING_LINE            = -1
    def refreshEnteringPoseElement(self) -> None:
        entering_pose = self.scene.rawEnteringPose()
        entering_pose_control_point = np.array([
            entering_pose[0] + math.cos(entering_pose[2]) * InteractiveVisualizer.ORIENTATION_LINE_LENGTH,
            entering_pose[1] + math.sin(entering_pose[2]) * InteractiveVisualizer.ORIENTATION_LINE_LENGTH
        ], dtype=np.float64)
        self.moveElement(self.CONTROL_ENTERING_POINT, entering_pose)
        self.moveElement(self.CONTROL_ENTERING_ORIENTATION, entering_pose_control_point)
        self.moveElement(self.AUX_ENTERING_CIRCLE, entering_pose)
        self.AUX_ENTERING_LINE = self.replaceLine(self.AUX_ENTERING_LINE, np.array([
            entering_pose[0:2], entering_pose_control_point
        ], dtype=np.float64), state=tk.DISABLED, fill='gray50')

    def initLeavingPoseElement(self) -> None:
        self.CONTROL_LEAVING_POINT        = self.cv.create_rectangle(0, 0, 10, 10, activefill='red')
        self.CONTROL_LEAVING_ORIENTATION  = self.cv.create_oval(0, 0, 10, 10, activefill='yellow')
        aux_circle_radius = InteractiveVisualizer.ORIENTATION_LINE_LENGTH * InteractiveVisualizer.SCALE
        self.AUX_LEAVING_CIRCLE           = self.cv.create_oval(0, 0, aux_circle_radius * 2, aux_circle_radius * 2, dash=(1,1), state=tk.DISABLED, outline='gray50')
        self.AUX_LEAVING_LINE             = -1
    def refreshLeavingPoseElement(self) -> None:
        leaving_pose = self.scene.rawLeavingPose()
        leaving_pose_control_point = np.array([
            leaving_pose[0] + math.cos(leaving_pose[2]) * InteractiveVisualizer.ORIENTATION_LINE_LENGTH,
            leaving_pose[1] + math.sin(leaving_pose[2]) * InteractiveVisualizer.ORIENTATION_LINE_LENGTH
        ], dtype=np.float64)
        self.moveElement(self.CONTROL_LEAVING_POINT, leaving_pose)
        self.moveElement(self.CONTROL_LEAVING_ORIENTATION, leaving_pose_control_point)
        self.moveElement(self.AUX_LEAVING_CIRCLE, leaving_pose)
        self.AUX_LEAVING_LINE = self.replaceLine(self.AUX_LEAVING_LINE, np.array([
            leaving_pose[0:2], leaving_pose_control_point
        ], dtype=np.float64), state=tk.DISABLED, fill='gray50')

    def initTrianlgeTurnElement(self) -> None:
        self.AUX_TRIANGLE   = -1
        self.AUX_CHORD      = -1
        self.AUX_VERTEX     = -1
        self.CONTROL_VERTEX = self.cv.create_oval(0, 0, 10, 10, activefill='yellow')
    def refreshTrianlgeTurnElement(self) -> None:
        self.AUX_TRIANGLE = self.replaceLine(self.AUX_TRIANGLE, np.array([
            self.scene.enteringPoint(),
            self.scene.triangleVertex(),
            self.scene.leavingPoint()
        ], dtype=np.float64), dash=(1,1), state=tk.DISABLED, fill='gray50')
        self.AUX_CHORD = self.replaceLine(self.AUX_CHORD, np.array([
            self.scene.enteringPoint(),
            self.scene.leavingPoint()
        ], dtype=np.float64), dash=(1,1), state=tk.DISABLED, fill='gray50')
        self.AUX_VERTEX = self.replaceLine(self.AUX_VERTEX, np.array([
            self.scene.chordMiddlePoint(),
            self.scene.triangleVertex()
        ], dtype=np.float64), dash=(1,1), state=tk.DISABLED, fill='gray50')
        if self.scene.isInferiorArc():
            self.moveElement(self.CONTROL_VERTEX, self.scene.triangleVertex())
        else:
            # TODO: the element could be removed here
            self.moveElement(self.CONTROL_VERTEX, np.array([-1000,-1000]))

    def initPathElement(self) -> None:
        self.PATH              = -1
        self.AUX_CIRCLE_RANGE  = -1
        self.AUX_CIRCLE_CENTER = self.cv.create_oval(0, 0, 6, 6, state=tk.DISABLED, fill='purple', outline='')
        self.AUX_CIRCLE_PRE    = self.cv.create_oval(0, 0, 6, 6, state=tk.DISABLED, fill='purple', outline='')
        self.AUX_CIRCLE_MIDDLE = self.cv.create_oval(0, 0, 6, 6, state=tk.DISABLED, fill='purple', outline='')
        self.AUX_CIRCLE_POST   = self.cv.create_oval(0, 0, 6, 6, state=tk.DISABLED, fill='purple', outline='')
        self.AUX_LINE_PRE      = self.cv.create_oval(0, 0, 6, 6, state=tk.DISABLED, fill='purple', outline='')
        self.AUX_LINE_POST     = self.cv.create_oval(0, 0, 6, 6, state=tk.DISABLED, fill='purple', outline='')
        self.AUX_CIRCLE_TANGENT= -1
        self.AUX_CONTROL_CIRCLE_RANGE = -1
        self.CONTROL_CIRCLE_N  = self.cv.create_oval(0, 0, 10, 10, activefill='yellow')
        # self.PATH_OUT          = -1
        # self.PATH_IN           = -1
        # self.AUX_CIRCLE_OUT    = -1
        # self.CONTROL_OUTER_N   = self.cv.create_oval(0, 0, 10, 10, activefill='yellow')
        # self.CONTROL_INNER_N   = self.cv.create_oval(0, 0, 10, 10, activefill='yellow')
    def refreshPathElement(self) -> None:
        self.PATH = self.replaceLine(self.PATH, self.scene.fullCurve(), state=tk.DISABLED, fill='purple')
        self.AUX_CIRCLE_RANGE = self.replaceLine(self.AUX_CIRCLE_RANGE, np.array([
            self.scene.arcEnteringPoseK()[0:2],
            self.scene.arcCenterPoint(),
            self.scene.arcLeavingPoseK()[0:2]
        ], dtype=np.float64), state=tk.DISABLED, fill='purple', dash=(1,1))
        if self.scene.isInferiorArc():
            self.AUX_CIRCLE_TANGENT = self.replaceLine(self.AUX_CIRCLE_TANGENT,
                self.scene.estimatedCircleTangent(), state=tk.DISABLED, fill='gray', dash=(1,1))
        else:
            circle_tangent = self.scene.estimatedCircleTangent()
            full_aux_line = np.array([
                self.scene.leavingPoint(),
                circle_tangent[0], circle_tangent[1],
                self.scene.enteringPoint()
            ])
            self.AUX_CIRCLE_TANGENT = self.replaceLine(self.AUX_CIRCLE_TANGENT,
                full_aux_line, state=tk.DISABLED, fill='gray', dash=(1,1))
        self.AUX_CONTROL_CIRCLE_RANGE = self.replaceLine(self.AUX_CONTROL_CIRCLE_RANGE,
            self.scene.estinatedCirclePointRange(), state=tk.DISABLED, fill='black', width=2)
        self.moveElement(self.AUX_CIRCLE_CENTER, self.scene.arcCenterPoint())
        self.moveElement(self.AUX_CIRCLE_PRE,    self.scene.arcEnteringPoseK())
        self.moveElement(self.AUX_CIRCLE_MIDDLE, self.scene.arcMiddlePoseK())
        self.moveElement(self.AUX_CIRCLE_POST,   self.scene.arcLeavingPoseK())
        self.moveElement(self.AUX_LINE_PRE,      self.scene.enteringPoint())
        self.moveElement(self.AUX_LINE_POST,     self.scene.leavingPoint())
        self.moveElement(self.CONTROL_CIRCLE_N,  self.scene.estimatedCirclePoint())
        # self.PATH_OUT = self.replaceLine(self.PATH_OUT, self.scene.outerBorder(), state=tk.DISABLED, fill='blue')
        # self.PATH_IN = self.replaceLine(self.PATH_IN, self.scene.innerBorder(), state=tk.DISABLED, fill='blue')
        # self.moveElement(self.CONTROL_OUTER_N,   self.scene.estimatedOuterPoint())
        # self.moveElement(self.CONTROL_INNER_N,   self.scene.estimatedInnerPoint())
        # if self.scene.symmetry:
        #     self.AUX_CIRCLE_OUT = self.replaceLine(self.AUX_CIRCLE_OUT, self.scene.outerCircleBorder(), state=tk.DISABLED, fill='blue', dash=(1,1))
        # else:
        #     self.AUX_CIRCLE_OUT = self.replaceLine(self.AUX_CIRCLE_OUT, None)
    def refereshSlider(self) -> None:
        if (self.var_radius.get() <= 1/abs(self.scene.maxCurvature()) or self.var_radius.get() >= 1/abs(self.scene.minCurvature())):
            self.curvature_slider_lock = True
        self.updateSliderInterval(self.slider_radius, self.var_radius, 1/abs(self.scene.maxCurvature()), 1/abs(self.scene.minCurvature()))
        self.updateSliderInterval(self.slider_lmbda, self.var_lmbda, self.scene.minLmbda(), self.scene.maxLmbda())
        self.var_lmbda.set(self.scene.lmbda)
        self.var_radius.set(1/abs(self.scene.curvature()))
        self.var_max_lmbda.set(self.scene.max_lmbda)
        self.label.config(text=dumpSceneDescription(self.scene))

    def refreshSceneFull(self) -> None:
        self.refreshEnteringPoseElement()
        self.refreshLeavingPoseElement()
        self.refreshTrianlgeTurnElement()
        self.refreshPathElement()
        self.refereshSlider()
    def refreshPathElementWithSlider(self) -> None:
        self.refreshPathElement()
        self.refereshSlider()

    def updateControlElement(self, element_id: int, dx: float, dy: float) -> None:
        move_vec = np.array([dx, dy], dtype=np.float64)
        if element_id == self.CONTROL_ENTERING_POINT:
            entering_pose = self.scene.rawEnteringPose()
            result = self.scene.setEnteringPose(entering_pose + np.array([dx, dy, 0], dtype=np.float64))
            if not result:
                element_id = -1
            self.refreshSceneFull()
        if element_id == self.CONTROL_LEAVING_POINT:
            leaving_pose = self.scene.rawLeavingPose()
            result = self.scene.setLeavingPose(leaving_pose + np.array([dx, dy, 0], dtype=np.float64))
            if not result:
                element_id = -1
            self.refreshSceneFull()
        if element_id == self.CONTROL_ENTERING_ORIENTATION:
            entering_pose = self.scene.rawEnteringPose()
            dot_result = move_vec.dot(np.array([
                math.cos(entering_pose[2] + math.pi / 2),
                math.sin(entering_pose[2] + math.pi / 2)
            ], dtype=np.float64))
            dt = dot_result / InteractiveVisualizer.ORIENTATION_LINE_LENGTH
            result = self.scene.setEnteringPose(entering_pose + np.array([0, 0, dt], dtype=np.float64))
            if not result:
                element_id = -1
            self.refreshSceneFull()
        if element_id == self.CONTROL_LEAVING_ORIENTATION:
            leaving_pose = self.scene.rawLeavingPose()
            dot_result = move_vec.dot(np.array([
                math.cos(leaving_pose[2] + math.pi / 2),
                math.sin(leaving_pose[2] + math.pi / 2)
            ], dtype=np.float64))
            dt = dot_result / InteractiveVisualizer.ORIENTATION_LINE_LENGTH
            result = self.scene.setLeavingPose(leaving_pose + np.array([0, 0, dt], dtype=np.float64))
            if not result:
                element_id = -1
            self.refreshSceneFull()
        if element_id == self.CONTROL_CIRCLE_N:
            original_y = self.scene.estimatedCircleN()
            dot_result = move_vec.dot(self.scene.triangle_params.triangle_n_direction)
            result = self.scene.setCircleN(original_y + dot_result)
            if not result:
                element_id = -1
            self.refreshPathElementWithSlider()
        # if element_id == self.CONTROL_OUTER_N:
        #     original_y = self.scene.estimatedOuterN()
        #     dot_result = move_vec.dot(self.scene.triangle_params.triangle_n_direction)
        #     result = self.scene.setOuterN(original_y + dot_result)
        #     if not result:
        #         element_id = -1
        #     self.refreshPathElementWithSlider()
        # if element_id == self.CONTROL_INNER_N:
        #     original_y = self.scene.estimatedInnerN()
        #     dot_result = move_vec.dot(self.scene.triangle_params.triangle_n_direction)
        #     result = self.scene.setInnerN(original_y + dot_result)
        #     if not result:
        #         element_id = -1
        #     self.refreshPathElementWithSlider()
        if element_id == self.CONTROL_VERTEX:
            original_vertex = self.scene.triangleVertex()
            new_vertex = original_vertex + move_vec
            result = self.scene.setVertexPoint(new_vertex)
            if not result:
                element_id = -1
            self.refreshSceneFull()
    def updateLmbdaBySlider(self, event: str) -> None:
        self.scene.setLmbda(self.slider_lmbda.get())
        self.refreshPathElementWithSlider()
    def updateMaxLmbdaBySlider(self, event: str) -> None:
        self.scene.setMaxLmbda(self.slider_max_lmbda.get())
        self.refreshSceneFull()
    def updateCurvatureBySlider(self, event: str) -> None:
        if self.curvature_slider_lock:
            self.curvature_slider_lock = False
            return
        curvature = 1 / self.slider_radius.get()
        if curvature < abs(self.scene.estimated_params0.curvature) or curvature > abs(self.scene.estimated_params1.curvature):
            # The curvature radius slider is spanned with the curvature at lambda_min and lambda_max,
            # therefore a feasible solution can exceed this range.
            # In such cases the curve will not be updated.
            return
        if self.scene.delta() < 0:
            curvature *= -1
        self.scene.setCurvature(curvature)
        self.refreshPathElementWithSlider()

    def flip(self):
        current_entering_pose = self.scene.rawEnteringPose()
        current_leaving_pose = self.scene.rawLeavingPose()
        new_entering_pose = np.array([
            current_entering_pose[0], current_leaving_pose[1], -current_entering_pose[2]
        ], dtype=np.float64)
        new_leaving_pose = np.array([
            current_leaving_pose[0], current_entering_pose[1], -current_leaving_pose[2]
        ], dtype=np.float64)
        self.scene = Scene(
            new_entering_pose, new_leaving_pose, self.scene.lmbda, self.scene.vehicle_half_width, self.scene.vehicle_base_front, self.scene.symmetry
        )
        self.refreshSceneFull()
    def flipSymmetry(self):
        try:
            flip_scene = Scene(
                self.scene.raw_entering_pose, self.scene.raw_leaving_pose, self.scene.lmbda, self.scene.vehicle_half_width, self.scene.vehicle_base_front, not self.scene.symmetry
            )
            self.scene = flip_scene
            self.refreshSceneFull()
        except:
            tkinter.messagebox.showerror(title='Error' , message='Cannot toggle to non-symmetry mode')

    def enteringMove(self, event: tk.Event) -> None:
        all_id = self.cv.find_closest(event.x, event.y)
        for item in all_id:
            if item in self.control_elements:
                self.picked_element = item
                self.last_x = event.x
                self.last_y = event.y
                return
    def stopMove(self, event: tk.Event) -> None:
        self.picked_element = -1
    def onMotion(self, event: tk.Event) -> None:
        if self.picked_element == -1:
            return
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.updateControlElement(self.picked_element, dx / InteractiveVisualizer.SCALE, -dy / InteractiveVisualizer.SCALE)
        self.last_x = event.x
        self.last_y = event.y

    def __init__(self, scene: Scene) -> None:
        """Initialize of the visualizer

        Args:
            scene: backend scene
        """

        self.root = tk.Tk()
        self.root.geometry('800x800')

        # ============
        # BACKEND DATA
        # ============
        self.scene = scene

        # ===========
        # MAIN CANVAS
        # ===========
        self.cv = tk.Canvas(self.root, height=800, width=560, bd=0, background='white')
        self.cv.pack(side=tk.LEFT)

        # =============
        # CONTROL FRAME
        # =============
        self.curvature_slider_lock = True
        command_frame = tk.Frame(self.root,height=800, width=240)
        command_frame.pack(side=tk.RIGHT)
        self.label = tk.Label(command_frame,font='TkFixedFont')
        self.label.pack()
        self.var_max_lmbda = tk.DoubleVar(value=self.scene.max_lmbda)
        self.slider_max_lmbda = tk.Scale(command_frame,
                                         variable=self.var_max_lmbda,
                                         length=240, orient=tk.HORIZONTAL, digits = 3,
                                         label="Max Lambda",
                                         command=self.updateMaxLmbdaBySlider)
        self.updateSliderInterval(self.slider_max_lmbda, self.var_max_lmbda, 0, 1)
        self.slider_max_lmbda.pack()
        self.var_lmbda = tk.DoubleVar(value=self.scene.lmbda)
        self.slider_lmbda = tk.Scale(command_frame,
                                     variable=self.var_lmbda,
                                     length=240, orient=tk.HORIZONTAL, digits = 3,
                                     label="Lambda",
                                     command=self.updateLmbdaBySlider)
        self.slider_lmbda.pack()
        self.var_radius = tk.DoubleVar(value=1/abs(self.scene.curvature()))
        self.slider_radius = tk.Scale(command_frame,
                                      variable=self.var_radius,
                                      length=240, orient=tk.HORIZONTAL, digits = 3,
                                      label="Curvature Radius",
                                      command=self.updateCurvatureBySlider)
        self.slider_radius.pack()
        self.curvature_slider_lock = False
        btn = tk.Button(command_frame, text = 'Flip', width=240, command=self.flip)
        btn.pack()
        btn2 = tk.Button(command_frame, text = 'Toggle Symmetry', width=240, command=self.flipSymmetry)
        btn2.pack()

        # ==================
        # ELEMENTS IN CANVAS
        # ==================
        self.initEnteringPoseElement()
        self.initLeavingPoseElement()
        self.initTrianlgeTurnElement()
        self.initPathElement()
        self.refreshSceneFull()

        # ================
        # MANIPULATE LOGIC
        # ================
        self.picked_element = -1
        self.last_x = 0
        self.last_y = 0
        self.control_elements = [
            self.CONTROL_ENTERING_POINT,
            self.CONTROL_ENTERING_ORIENTATION,
            self.CONTROL_LEAVING_POINT,
            self.CONTROL_LEAVING_ORIENTATION,
            self.CONTROL_CIRCLE_N,
            self.CONTROL_VERTEX
            # self.CONTROL_OUTER_N,
            # self.CONTROL_INNER_N,
        ]
        self.cv.bind("<ButtonPress-1>", self.enteringMove)
        self.cv.bind("<ButtonRelease-1>", self.stopMove)
        self.cv.bind("<B1-Motion>", self.onMotion)

    def spin(self):
        self.root.mainloop()
