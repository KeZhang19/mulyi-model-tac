from _typeshed import Incomplete
from typing import Any

__all__ = ['VTSError', 'VTSConfigError', 'InvalidConfigError', 'InvalidParameterError', 'InvalidDataTypeError', 'VTSDeviceError', 'DeviceNotFoundError', 'DeviceConnectionError', 'CameraDisconnectedError', 'VTSCalibrationError', 'CalibrationRequiredError', 'CalibrationFailedError', 'BackgroundNotInitializedError', 'MarkerNotInitializedError', 'VTSDataError', 'InvalidFrameError', 'InvalidFrameSizeError', 'InvalidFrameTypeError', 'FrameNotAvailableError', 'VTSBackendError', 'BackendNotStartedError', 'BackendInitializationError', 'VTSOperationError', 'InvalidOperationError', 'ResourceNotAvailableError', 'OfflineModeError', 'OnlineModeError', 'VTSImageProcessingError', 'InvalidWarpParametersError', 'InvalidOffsetError', 'VTSModelError', 'ModelNotFoundError', 'ModelLoadError', 'ModelInitializationError', 'ModelNotInitializedError', 'ModelInferenceError', 'ForceModelPathRequiredError', 'VTSMarkerTrackingError', 'InvalidMarkerCountError', 'MarkerDetectionError', 'InvalidTrackingMethodError', 'VTSFEMError', 'FEMMeshNotInitializedError', 'FEMStiffnessMatrixError', 'FEMNodeCountError', 'FEMInvalidFaceError', 'create_exception_from_std']

class VTSError(Exception):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str
    message: Incomplete
    error_code: Incomplete
    suggestion: Incomplete
    context: Incomplete
    def __init__(self, message: str, error_code: int | None = None, suggestion: str | None = None, context: dict[str, Any] | None = None) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

class VTSConfigError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidConfigError(VTSConfigError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidParameterError(VTSConfigError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidDataTypeError(VTSConfigError, TypeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSDeviceError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class DeviceNotFoundError(VTSDeviceError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class DeviceConnectionError(VTSDeviceError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class CameraDisconnectedError(VTSDeviceError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSCalibrationError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class CalibrationRequiredError(VTSCalibrationError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class CalibrationFailedError(VTSCalibrationError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class BackgroundNotInitializedError(VTSCalibrationError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class MarkerNotInitializedError(VTSCalibrationError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSDataError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidFrameError(VTSDataError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidFrameSizeError(VTSDataError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidFrameTypeError(VTSDataError, TypeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class FrameNotAvailableError(VTSDataError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSBackendError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class BackendNotStartedError(VTSBackendError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class BackendInitializationError(VTSBackendError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSOperationError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidOperationError(VTSOperationError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class ResourceNotAvailableError(VTSOperationError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class OfflineModeError(VTSOperationError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class OnlineModeError(VTSOperationError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSImageProcessingError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidWarpParametersError(VTSImageProcessingError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidOffsetError(VTSImageProcessingError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSModelError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class ModelNotFoundError(VTSModelError, FileNotFoundError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class ModelLoadError(VTSModelError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class ModelInitializationError(VTSModelError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class ModelNotInitializedError(VTSModelError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class ModelInferenceError(VTSModelError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class ForceModelPathRequiredError(VTSModelError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSMarkerTrackingError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidMarkerCountError(VTSMarkerTrackingError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class MarkerDetectionError(VTSMarkerTrackingError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class InvalidTrackingMethodError(VTSMarkerTrackingError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class VTSFEMError(VTSError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class FEMMeshNotInitializedError(VTSFEMError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class FEMStiffnessMatrixError(VTSFEMError, RuntimeError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class FEMNodeCountError(VTSFEMError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

class FEMInvalidFaceError(VTSFEMError, ValueError):
    ERROR_CODE: int
    DEFAULT_SUGGESTION: str

def create_exception_from_std(std_exception: Exception, error_class: type = ..., suggestion: str | None = None) -> VTSError: ...
