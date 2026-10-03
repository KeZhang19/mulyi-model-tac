import numpy as np
from ....common import Logger as Logger
from ....common.exceptions import InvalidOffsetError as InvalidOffsetError, InvalidParameterError as InvalidParameterError, InvalidWarpParametersError as InvalidWarpParametersError
from ...utils.image_transformation import warpPerspective as warpPerspective
from _typeshed import Incomplete

logger: Incomplete

class GFImageProcessor:
    _warp_src: Incomplete
    _warp_src_in: Incomplete
    _input_size: Incomplete
    _warp_dsize: Incomplete
    _warp_scale: float
    _warp_paddings: Incomplete
    _is_auto_warp_mode: bool
    _need_warpPerspective: bool
    _corner_detector: str
    def __init__(self, input_size: tuple[int, int], output_size: tuple[int, int]) -> None: ...
    def set_warp_params(self, corner_points: list[list[int]] = [], offset: list[int] = [0, 0, 0, 0], scale: float = 1.0, mode: str = 'auto', need_warpPerspective: bool = True, corner_detector: str = 'default') -> None: ...
    def process(self, image: np.ndarray) -> np.ndarray: ...
    def _rotate_image(self, image: np.ndarray) -> np.ndarray: ...
