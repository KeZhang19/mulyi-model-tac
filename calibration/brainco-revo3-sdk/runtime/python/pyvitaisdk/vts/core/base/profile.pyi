from _typeshed import Incomplete
from dataclasses import dataclass
from enum import Enum, IntEnum

@dataclass(frozen=True)
class VideoStreamFmt:
    width: int
    height: int
    fps: int
    fourcc: str

@dataclass(frozen=True)
class MarkerFmt:
    center_distance: float
    diameter: float
    rows: int
    columns: int

@dataclass(frozen=True)
class GelFmt:
    width: float
    height: float
    thickness: float

@dataclass(frozen=True)
class OutputFmt:
    width: int
    height: int

class GF225VideoStreamProfile:
    MJPG_640_360_30: Incomplete
    MJPG_320_240_120: Incomplete

class GF225MarkerProfile:
    NORMARL_PROFILE: Incomplete
    SMALL_PROFILE: Incomplete

class GF225GelProfile:
    NORMAL: Incomplete

class GF225OutputProfile:
    W240_H240: Incomplete

class GF220VideoStreamProfile:
    MJPG_640_480_30: Incomplete
    MJPG_320_240_30: Incomplete

class GF220MarkerProfile:
    NORMARL_PROFILE: Incomplete

class GF220GelProfile:
    NORMAL: Incomplete

class GF220OutputProfile:
    W200_H300: Incomplete

class GF210VideoStreamProfile:
    MJPG_640_360_30: Incomplete

class GF210MarkerProfile:
    NORMARL_PROFILE: Incomplete

class GF210GelProfile:
    NORMAL: Incomplete

class GF210OutputProfile:
    W320_H180: Incomplete

class GF515VideoStreamProfile:
    MJPG_1280_720_30: Incomplete
    MJPG_1280_720_60: Incomplete
    MJPG_640_480_60: Incomplete
    MJPG_640_480_30: Incomplete
    MJPG_640_360_60: Incomplete
    MJPG_640_360_30: Incomplete
    MJPG_320_240_60: Incomplete
    MJPG_320_240_30: Incomplete

class GF515MarkerProfile:
    NORMARL_PROFILE: Incomplete
    SMALL_PROFILE: Incomplete

class GF515OutputProfile:
    W160_H90: Incomplete
    W320_H180: Incomplete
    W240_H180: Incomplete
    W240_H240: Incomplete

class GF515GelProfile:
    NORMAL: Incomplete

class GFTGVideoStreamProfile:
    MJPG_1280_720_30: Incomplete
    MJPG_640_480_30: Incomplete
    MJPG_640_360_30: Incomplete
    MJPG_320_240_30: Incomplete

class GFTGMarkerProfile:
    NORMARL_PROFILE: Incomplete
    SMALL_PROFILE: Incomplete

class GFTGOutputProfile:
    W210_H180: Incomplete
    W220_H180: Incomplete

class GFTGGelProfile:
    NORMAL: Incomplete

class GFBCVideoStreamProfile:
    MJPG_640_480_60: Incomplete
    MJPG_640_480_30: Incomplete
    MJPG_640_360_30: Incomplete
    MJPG_320_240_120: Incomplete
    MJPG_320_240_60: Incomplete
    MJPG_320_240_30: Incomplete

class GFBCMarkerProfile:
    NORMARL_PROFILE: Incomplete
    SMALL_PROFILE: Incomplete

class GFBCOutputProfile:
    W240_H180: Incomplete
    W210_H180: Incomplete
    W320_H180: Incomplete
    W280_H140: Incomplete
    W320_H240: Incomplete
    W100_H75: Incomplete
    W120_H90: Incomplete
    W105_H90: Incomplete

class GFBCGelProfile:
    NORMAL: Incomplete

class VTSensorType(Enum):
    GF225 = 'GF225'
    GF220 = 'GF220'
    GF210 = 'GF210'
    GF515T = 'GF515T'
    GF515I = 'GF515I'
    GFBCT = 'GFBCT'
    GFBCI = 'GFBCI'
    GFTGI = 'GFTGI'
    GFTGT = 'GFTGT'

class VTSDataType(IntEnum):
    TIME_STAMP = 0
    CALIBRATE_IMG = 1
    RAW_IMG = 2
    WARPED_IMG = 3
    DIFF_IMG = 4
    DEPTH_MAP = 5
    MARKER_IMG = 6
    MARKER_ORIGIN_VECTOR = 7
    MARKER_CURRENT_VECTOR = 8
    MARKER_OFFSET_VECTOR = 9
    XYZ_VECTOR = 10
    SLIP_STATE = 11
    FORCE6D_VECTOR = 12
    @classmethod
    def get_all_types(cls) -> set: ...
    @classmethod
    def is_valid(cls, data_type: int) -> bool: ...
