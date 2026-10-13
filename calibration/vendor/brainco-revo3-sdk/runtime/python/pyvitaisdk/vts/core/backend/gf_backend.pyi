import numpy as np
from ....common import Logger as Logger
from _typeshed import Incomplete
from enum import IntEnum
from threading import Thread
from typing import Callable

logger: Incomplete

class ThreadLifeCycle(IntEnum):
    STARTING = 0
    RUNNING = 1
    ENDING = 2

class GFBackend:
    _is_running: bool
    _thread: Thread | None
    _state: Incomplete
    _backend_name: Incomplete
    def __init__(self, backend_name: str = 'GF BackEnd') -> None: ...
    def start(self, process_func: Callable, on_frame_callback: Callable[[float, np.ndarray], None]) -> None: ...
    def stop(self) -> None: ...
    def _backend_loop(self, process_func: Callable, on_frame_callback: Callable[[float, np.ndarray], None]) -> None: ...
