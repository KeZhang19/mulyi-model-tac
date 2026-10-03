import numpy as np
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Deque

__all__ = ['CornerStabilizer', 'corner_point_detection_gf220']

Line = tuple[float, float, float]
Bounds = tuple[int, int, int, int]

@dataclass(frozen=True)
class EdgeBand:
    name: str
    normalized_bounds: tuple[float, float, float, float]
    scan_axis: str
    gradient_sign: int
    max_slope: float

@dataclass
class EdgeFit:
    line: Line
    bounds: Bounds
    candidates: np.ndarray
    inliers: np.ndarray
    inlier_ratio: float
    rmse: float
    median_strength: float

class CornerStabilizer:
    _history: Deque[np.ndarray]
    _min_samples: Incomplete
    _max_deviation: Incomplete
    _max_jump: Incomplete
    _stable: np.ndarray | None
    def __init__(self, window_size: int = 10, min_samples: int = 6, max_deviation: float = 3.0, max_jump: float = 8.0) -> None: ...
    def update(self, corners: list[list[int]] | None) -> list[list[int]] | None: ...
    def reset(self) -> None: ...
    @staticmethod
    def _as_list(corners: np.ndarray | None) -> list[list[int]] | None: ...

def corner_point_detection_gf220(image: np.ndarray, valid_region: list[tuple[int, int]] | None = None, debug: bool = False, min_gradient_strength: float = 40.0) -> list[list[int]] | None: ...
