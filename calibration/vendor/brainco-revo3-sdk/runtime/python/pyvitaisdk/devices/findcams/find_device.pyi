from ...common import hash_crc16 as hash_crc16
from ...common.exceptions import DeviceNotFoundError as DeviceNotFoundError, InvalidOperationError as InvalidOperationError
from _typeshed import Incomplete
from collections import namedtuple as namedtuple
from comtypes import IPersist, IUnknown

def _build_camera_info(name: str, raw_sn: str, index: int): ...
def getUdevadmProperty(devName): ...
def find_all_cameras(): ...
def _get_usb_serial_map_mac(): ...

VITAI_SN_PREFIXES: Incomplete

def _is_vitai_registry_device(device_info: dict[str, object]) -> bool: ...
def _filter_vitai_registry_devices(devices: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]: ...

class ISequentialStream(IUnknown):
    _case_insensitive_: bool
    _iid_: Incomplete
    _idlflags_: Incomplete

class IStream(ISequentialStream):
    _case_insensitive_: bool
    _iid_: Incomplete
    _idlflags_: Incomplete

class IBindCtx(IUnknown):
    _case_insensitive_: bool
    _iid_: Incomplete
    _idlflags_: Incomplete

class IPersistStream(IPersist):
    _case_insensitive_: bool
    _iid_: Incomplete
    _idlflags_: Incomplete

class IMoniker(IPersistStream):
    _case_insensitive_: bool
    _iid_: Incomplete
    _idlflags_: Incomplete

IMONIKER: Incomplete

class IEnumMoniker(IUnknown):
    _case_insensitive_: bool
    _iid_: Incomplete
    _idlflags_: Incomplete

class ICreateDevEnum(IUnknown):
    _case_insensitive_: bool
    _iid_: Incomplete
    _idlflags_: Incomplete

def get_moniker_name(moniker) -> str: ...
def get_device_path(moniker) -> str: ...
def get_current_devices_via_registry(): ...
def _extract_vid_pid_iid(text): ...
def link_parent_child_devices(devices): ...
def _normalize_device_path(path: str) -> str: ...
def _enumerate_msmf_devices() -> list: ...
