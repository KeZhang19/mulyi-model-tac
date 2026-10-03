import numpy as np
from .estimator_interface import IDisplacementEstimator as IDisplacementEstimator
from _typeshed import Incomplete
from scipy.spatial import KDTree as KDTree

logger: Incomplete

class RobustBilinearEstimator(IDisplacementEstimator):
    expected_shape: Incomplete
    grid_initial: Incomplete
    row_ref: Incomplete
    col_ref: Incomplete
    query_points: Incomplete
    dx: Incomplete
    dy: Incomplete
    max_displacement: float
    boundary_margin: int
    def __init__(self, expected_shape=(9, 9)) -> None: ...
    def initialize(self, initial_points: np.ndarray): ...
    def set_query_points(self, query_points: np.ndarray): ...
    def get_query_points(self): ...
    grid_current: Incomplete
    def update_displacements(self, current_points: np.ndarray): ...
    def estimate(self, query_points: np.ndarray = None): ...
    @staticmethod
    def visualize(frame: np.ndarray, pts: np.ndarray, displacement: np.ndarray, enable_debounce: bool = True): ...
