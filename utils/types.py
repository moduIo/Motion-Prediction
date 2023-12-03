from enum import Enum


class ModelEnum(Enum):
    SPATIO_TEMPORAL_TRANSFORMER = "spatio-temporal-transformer"
    BIDIRECTIONAL_TRANSFORMER = "bi-drectional-transformer"
    LSTM_SEQ2SEQ = "lstm_seq2seq"

class TargetEnum(Enum):
    AUTO_REGRESSIVE = "auto-regressive"
    PRE_TRAIN = "pre-train"
    DEFAULT = "default"
