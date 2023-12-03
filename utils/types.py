from enum import Enum


class ModelEnum(Enum):
    SPATIO_TEMPORAL_TRANSFORMER = "spatio-temporal-transformer"
    BIDIRECTIONAL_TRANSFORMER = "bi-directional-transformer"
    LSTM_SEQ2SEQ = 'lstm_seq2seq'
    LSTM_SEQ2SEQ_ATT = 'lstm_seq2seq_att'


class TargetEnum(Enum):
    AUTO_REGRESSIVE = "auto-regressive"
    PRE_TRAIN = "pre-train"
    DEFAULT = "default"
