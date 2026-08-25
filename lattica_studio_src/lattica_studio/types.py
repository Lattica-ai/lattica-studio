from typing import Any, NotRequired, TypeAlias, TypedDict
from enum import Enum


ModelId: TypeAlias = str
WorkerSessionId: TypeAlias = str
Token: TypeAlias = str


class ModelInfo(TypedDict, total=False):
    modelId: str
    modelName: str
    description: str
    status: str
    visibility: str

    instanceTypeId: str
    numDevices: int

    is_compiled: bool
    compilation_error: str

    autoRestart: bool
    inputType: str
    outputType: str


class WorkerStatus(TypedDict, total=False):
    workerSessionId: str
    status: str
    modelId: str


class ActiveWorkerGroup(TypedDict, total=False):
    workers: list[WorkerStatus]


class TokenInfo(TypedDict, total=False):
    tokenStatus: str
    tokenName: str
    tokenExpiration: str

    modelName: str
    modelStatus: str
    workerStatus: str
    evalKeyCreatedAt: str


JsonDict: TypeAlias = dict[str, Any]

class InstanceType(str, Enum):
    """Enum representing available instance types for model deployment"""
    G4DN_XLARGE = "G4DN_XLARGE"
    G5_2XLARGE = "G5_2XLARGE"
    G6E_2XLARGE = "G6E_2XLARGE"
    G7E_2XLARGE = "G7E_2XLARGE"
    G7E_12XLARGE = "G7E_12XLARGE"
    TPU_MEDIUM = "TPU_MEDIUM"
    CPU_C7G_XLARGE = "CPU_C7G_XLARGE"
    CPU_C7G_2XLARGE = "CPU_C7G_2XLARGE"

class SchemeType(str, Enum):
    LATTICA = "LATTICA"       # FHE
    SUNSCREEN = "SUNSCREEN"   # TFHE
