#!/usr/bin/env python3
# coding=utf-8
'''
Description  :  
'''

import logging

import colorlog

_handler = logging.StreamHandler()
_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)
logging.basicConfig(level=logging.INFO, handlers=[_handler])

from .dfs_tracking import DFSMarkerTracking as MarkerTracking
from .opticalflow_marker import OpticalFlowMarkerGenerator
from .adaptive_optical_flow import AdaptiveOpticalFlowTracker, OpticalFlowTracker
from .exceptions import MarkerError
from .utils import draw_flow

__all__ = [
    "AdaptiveOpticalFlowTracker",
    "MarkerError",
    "MarkerTracking",
    "OpticalFlowMarkerGenerator",
    "OpticalFlowTracker",
    "draw_flow",
]
