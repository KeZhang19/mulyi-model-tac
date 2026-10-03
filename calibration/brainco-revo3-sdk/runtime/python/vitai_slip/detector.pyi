import numpy as np
from .profile import SlipState as SlipState
from _typeshed import Incomplete

logger: Incomplete

class SlipDetector:
    _slip_state: Incomplete
    _marker_history: Incomplete
    _history_len: Incomplete
    _last_state: Incomplete
    def __init__(self, history_len: int = 5) -> None: ...
    @property
    def slip_state(self) -> SlipState: ...
    def update_marker_history(self, markers: np.ndarray): ...
    def slip_detect(self, origin_markers: np.ndarray, current_markers: np.ndarray): ...
