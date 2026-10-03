import numpy as np
from .utils import draw_flow as draw_flow
from _typeshed import Incomplete
from dataclasses import dataclass

logger: Incomplete

@dataclass
class ReferenceFrame:
    image: np.ndarray
    flow: np.ndarray
    timestamp: float

class OpticalFlowTracker:
    image_size: Incomplete
    dis_flow: Incomplete
    initial_reference: np.ndarray | None
    def __init__(self, image_size: tuple[int, int]) -> None: ...
    def initialize(self, initial_image: np.ndarray): ...
    def compute_flow(self, current_image: np.ndarray, timestamp: float = 0.0) -> np.ndarray: ...

class AdaptiveOpticalFlowTracker:
    image_size: Incomplete
    add_reference_threshold: Incomplete
    loop_closure_threshold: Incomplete
    downsample_factor: Incomplete
    coarse_size: Incomplete
    dis_flow: Incomplete
    coarse_flow: Incomplete
    reference_stack: list[ReferenceFrame]
    initial_reference: np.ndarray | None
    prev_dis_flow: np.ndarray | None
    prev_coarse_flow: np.ndarray | None
    stats: Incomplete
    use_dis_optical_flow: Incomplete
    farneback_params: Incomplete
    def __init__(self, image_size: tuple[int, int], add_reference_threshold: float = 10.0, loop_closure_threshold: float = 5.0, downsample_factor: int = 10, use_dis_optical_flow: bool = False) -> None: ...
    def initialize(self, initial_image: np.ndarray): ...
    def compute_flow(self, current_image: np.ndarray, timestamp: float = 0.0) -> np.ndarray: ...
    def _warp_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray: ...
    def get_stats(self) -> dict: ...
    def get_initial_reference(self) -> np.ndarray | None: ...
    def get_current_reference(self) -> np.ndarray | None: ...
    def reset(self) -> None: ...

def visualize_optical_flow(flow: np.ndarray) -> np.ndarray: ...
def main() -> None: ...
