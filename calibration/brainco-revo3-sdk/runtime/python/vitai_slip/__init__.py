#!/usr/bin/env python3
# coding=utf-8
"""Marker-based slip detection."""

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

from .detector import SlipDetector
from .profile import SlipState

__all__ = ["SlipDetector", "SlipState"]
