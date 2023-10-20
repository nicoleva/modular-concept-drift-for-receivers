from enum import Enum

HALF = 0.5


class ChannelModes(Enum):
    SISO = 'SISO'
    MIMO = 'MIMO'


class ChannelModels(Enum):
    DistortedMIMO = 'DistortedMIMO'


class DetectorType(Enum):
    black_box = 'black_box'
    model = 'model'
