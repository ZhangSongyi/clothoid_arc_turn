import math
import numpy as np
import tkinter as tk
from clothoid_arc_turn import InteractiveVisualizer, Scene

scene = Scene(
            np.array([5, 18, 0]),
            np.array([12, 5, -math.pi * 1 / 2]),
            lmbda=1.0, symmetry=True
        )
interactive_visualizer = InteractiveVisualizer(scene)
interactive_visualizer.spin()
