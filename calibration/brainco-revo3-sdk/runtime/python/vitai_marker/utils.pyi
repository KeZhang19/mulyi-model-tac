import numpy as np
from .exceptions import InvalidFrameError as InvalidFrameError, InvalidFrameTypeError as InvalidFrameTypeError, InvalidParameterError as InvalidParameterError

YELLOW_DISPLACEMENT: float

def draw_flow(frame: np.ndarray, flow, enable_debounce: bool = True, debounce_threshold: float = 3.0, arrow_gain: float = 3.0, arrow_thickness: int = 2, arrow_tip_length: float = 0.3) -> None: ...
