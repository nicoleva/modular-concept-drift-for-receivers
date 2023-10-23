from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import ChannelModes, DetectorType, ChannelModels


class PlotType(Enum):
    SingleUserDistortedMIMODeepSIC = 'SingleUserDistortedMIMODeepSIC'
    ModularSingleUserDistortedMIMODeepSIC = 'ModularSingleUserDistortedMIMODeepSIC'
    DistortedMIMODeepSIC = 'DistortedMIMODeepSIC'


def get_config(label_name: str) -> Tuple[List[Dict], list, list, str, str, str]:
    drift_detection_methods = None
    if label_name == PlotType.SingleUserDistortedMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 5, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.3, 'delta_pht': 0.6, 'lambda_pht': 0.15},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.01},
             },
            {'drift_detection_method': 'PST',
             'drift_detection_method_hp': {'posterior_threshold': 0.96},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.ModularSingleUserDistortedMIMODeepSIC.name:
        methods_list = [
            # 'PERIODIC',
            # 'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 5, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.3, 'delta_pht': 0.6, 'lambda_pht': 0.15},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.01},
             },
            {'drift_detection_method': 'PST',
             'drift_detection_method_hp': {'posterior_threshold': 0.96},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.DistortedMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 5500, 'pilot_size': 500, 'drift_detection_method': None,
             'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 7, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.2, 'delta_pht': 0.8, 'lambda_pht': 0.1},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.005},
             },
            {'drift_detection_method': 'PST',
             'drift_detection_method_hp': {'posterior_threshold': 0.9},
             },
        ]
        values = list(range(50))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    else:
        raise ValueError('No such plot mechanism_type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel, plot_type, drift_detection_methods
