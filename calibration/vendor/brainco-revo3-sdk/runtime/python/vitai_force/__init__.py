#!/usr/bin/env python3
# coding=utf-8
"""Six-axis force estimation module."""

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

from .inference_mlp import MLPForceEstimator
from .estimator import ForceEstimator
from .exceptions import ForceError
from .resources import get_force_model_path
from .utils import validate_force_model

__all__ = [
    "ForceEstimator",
    "ForceError",
    "MLPForceEstimator",
    "get_force_model_path",
    "validate_force_model",
]
