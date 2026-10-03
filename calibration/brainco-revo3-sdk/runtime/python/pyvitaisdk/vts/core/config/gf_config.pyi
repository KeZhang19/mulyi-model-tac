from ....common import Logger as Logger, hash_crc16 as hash_crc16
from ....common.exceptions import InvalidConfigError as InvalidConfigError, InvalidDataTypeError as InvalidDataTypeError, InvalidParameterError as InvalidParameterError
from ....devices import VTSDeviceConfig as VTSDeviceConfig
from ..base.profile import GF225OutputProfile as GF225OutputProfile, GF225VideoStreamProfile as GF225VideoStreamProfile, OutputFmt as OutputFmt, VideoStreamFmt as VideoStreamFmt
from _typeshed import Incomplete
from dataclasses import dataclass

logger: Incomplete

@dataclass
class GFConfig:
    device_config: VTSDeviceConfig
    model_path: str
    marker_size: list[int]
    marker_offsets: list[int]
    stream_format: VideoStreamFmt
    output_format: OutputFmt
    device_id: int = ...
    @classmethod
    def create(cls, device_config: VTSDeviceConfig = None, model_path: str = None, marker_size: int | list[int] = [9, 9], marker_offsets: list[int] | None = None, stream_format: VideoStreamFmt = ..., output_format: OutputFmt = ..., device_id: int = 0) -> GFConfig: ...
    @staticmethod
    def _validate_sn(sn: str) -> bool: ...
    @staticmethod
    def _validate_marker_size(marker_size: int | list[int], output_format: OutputFmt) -> list[int]: ...
    @staticmethod
    def _validate_marker_offsets(marker_offsets: list[int] | None, output_format: OutputFmt) -> list[int]: ...
