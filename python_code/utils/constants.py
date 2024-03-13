from enum import Enum

HALF = 0.5


class ChannelModes(Enum):
    SISO = 'SISO'
    MIMO = 'MIMO'


class ChannelModels(Enum):
    DistortedMIMO = 'DistortedMIMO'
    OneUserDistortedMIMOChannel = 'OneUserDistortedMIMOChannel'
    SEDChannel = 'SEDChannel'
    Cost2100 = 'Cost2100'


class DetectorType(Enum):
    black_box = 'black_box'
    model = 'model'
