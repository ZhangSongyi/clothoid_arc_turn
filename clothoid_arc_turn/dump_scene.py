from .scene import Scene
import numpy as np
import numpy.typing as npt

def dumpItem(description: str, item: npt.NDArray[np.float64], postfix: str = '') -> str:
    np.set_printoptions(suppress=True)
    return f'{description}{postfix} = ' + np.array2string(item, separator=",").replace('[','{').replace(']','}').replace('\n','') + ';'

def dumpScene(scene: Scene, postfix: str = '') -> str:
    """dump a scene to Mathematica compatible code

    Args:
        scene: scene that want to dump.
        postfix: postfix for Mathematica variables. Defaults to ''.

    Returns:
        generated mathematica code.
    """
    return dumpItem('enteringStraight', scene.center_curve.entering_straight_part       , postfix) + '\n' \
        +  dumpItem('enteringClothoid', scene.center_curve.entering_clothoid_part       , postfix) + '\n' \
        +  dumpItem('enteringArc'     , scene.center_curve.entering_arc_part            , postfix) + '\n' \
        +  dumpItem('leavingArc'      , scene.center_curve.leaving_arc_part             , postfix) + '\n' \
        +  dumpItem('leavingClothoid' , scene.center_curve.leaving_clothoid_part        , postfix) + '\n' \
        +  dumpItem('leavingStraight' , scene.center_curve.leaving_straight_part        , postfix) + '\n' \
        +  dumpItem('circCenter'      , scene.center_curve.arc_center_point             , postfix) + '\n' \
        +  dumpItem('triangleVertex'  , scene.triangle_params.triangle_vertex_point     , postfix) + '\n' \
        +  dumpItem('chordMiddle'     , scene.triangle_params.chord_middle_point        , postfix) + '\n' \
        +  dumpItem('curvatures'      , scene.curvatureProfile()                        , postfix) + '\n' \
        +  dumpItem('circlePoint'     , scene.estimatedCirclePoint()                    , postfix) + '\n' \
        +  dumpItem('circleRange'     , scene.estinatedCirclePointRange()               , postfix)

def dumpSceneDescription(scene: Scene) -> str:
    return 'INPUT -------------------'                                         + '\n' \
        + f'  triangle T    : {scene.triangle_params.triangle_t        :7.3f}' + '\n' \
        + f'  triangle N    : {scene.triangle_params.triangle_n        :7.3f}' + '\n' \
        + f'  delta         : {scene.triangle_params.delta             :7.3f}' + '\n' \
        + f'  delta phi     : {scene.triangle_params.delta_phi         :7.3f}' + '\n' \
        +  'PROPS -------------------'                                         + '\n' \
        + f'  curvature     : {scene.turn_params.curvature             :7.3f}' + '\n' \
        + f'  est curvature : {scene.estimated_params.curvature        :7.3f}' + '\n' \
        + f'  ent sharpness : {scene.turn_params.entering_sharpness    :7.3f}' + '\n' \
        + f'  lea sharpness : {scene.turn_params.leaving_sharpness     :7.3f}' + '\n' \
        + f'  ent circle n  : {scene.turn_params.entering_circle_n     :7.3f}' + '\n' \
        + f'  lea circle n  : {scene.turn_params.leaving_circle_n      :7.3f}' + '\n' \
        + f'  half del diff : {scene.turn_params.delta_delta           :7.3f}' + '\n' \
        + f'  est n         : {scene.estimated_params.circle_n         :7.3f}' + '\n' \
        +  'LEN   -------------------'                                         + '\n' \
        + f'  ent Ll        : {scene.triangle_params.entering_ll       :7.3f}' + '\n' \
        + f'  ent Ls        : {scene.turn_params.entering_ls           :7.3f}' + '\n' \
        + f'  ent Lc        : {scene.turn_params.entering_lc           :7.3f}' + '\n' \
        + f'  lea Lc        : {scene.turn_params.leaving_lc            :7.3f}' + '\n' \
        + f'  lea Ls        : {scene.turn_params.leaving_ls            :7.3f}' + '\n' \
        + f'  lea Ll        : {scene.triangle_params.leaving_ll        :7.3f}' + '\n' \
        +  '-------------------------'
