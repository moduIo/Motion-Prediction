from enum import Enum


class ModelEnum(Enum):
    SPATIO_TEMPORAL_TRANSFORMER = "spatio-temporal-transformer"


class TargetEnum(Enum):
    AUTO_REGRESSIVE = "auto-regressive"
    DEFAULT = "default"
