from ...common import Logger as Logger
from ...common.exceptions import InvalidParameterError as InvalidParameterError
from ...devices import VTSDeviceConfig as VTSDeviceConfig
from .base.gf_sensor_base import GFSensorBase as GFSensorBase
from .base.profile import VTSensorType as VTSensorType
from .gf210 import GF210 as GF210
from .gf220 import GF220 as GF220
from .gf225 import GF225 as GF225
from .gf515i import GF515I as GF515I
from .gf515t import GF515T as GF515T
from .gfbci import GFBCI as GFBCI
from .gfbct import GFBCT as GFBCT
from .gftgi import GFTGI as GFTGI
from .gftgt import GFTGT as GFTGT
from _typeshed import Incomplete

logger: Incomplete

class VTSensor:
    def __new__(cls, config: VTSDeviceConfig | None = None, marker_size: int | list[int] | None = None, model_path: str | None = None, force_model_path: str | None = None, marker_offsets: list[int] | None = None, sensor_type: VTSensorType | None = None, device_id: int = 0) -> GFSensorBase: ...
    @staticmethod
    def _detect_product(sn: str) -> str: ...
