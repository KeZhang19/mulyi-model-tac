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

from .reconstruct3d import Reconstruction3D
from .reconstruct3d import RKNNInference
from .resources import get_depth_model_path
from .exceptions import DepthError

__all__ = ["DepthError", "Reconstruction3D", "RKNNInference", "get_depth_model_path"]
