from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import ChannelModes, DetectorType, ChannelModels


class PlotType(Enum):
    SingleUserDistortedMIMODeepSIC = 'SingleUserDistortedMIMODeepSIC'
    ModularSingleUserDistortedMIMODeepSIC = 'ModularSingleUserDistortedMIMODeepSIC'
    CostMIMODeepSICSNR = 'CostMIMODeepSICSNR'
    ModularCostMIMODeepSICSNR = 'ModularCostMIMODeepSICSNR'
    CostMIMODNNSNR = 'CostMIMODNNSNR'
    SingleUserDistortedMIMODeepSICSNR = 'SingleUserDistortedMIMODeepSICSNR'
    SingleUserDistortedMIMODNN = 'SingleUserDistortedMIMODNN'
    DistortedMIMODeepSIC = 'DistortedMIMODeepSIC'
    ModularDistortedMIMODeepSIC = 'ModularDistortedMIMODeepSIC'
    LinearSISO = 'LinearSISO'
    LinearSISO_RNN = 'LinearSISO_RNN'
    HARD_SER_BLOCK_LINEAR_COST = 'HARD_SER_BLOCK_LINEAR_COST'
    SOFT_SER_BLOCK_LINEAR_COST = 'SOFT_SER_BLOCK_LINEAR_COST'
    SISO_BER_By_SNR = 'SISO_BER_By_SNR'
    SISO_BER_By_SNR_RNN = 'SISO_BER_By_SNR_RNN'
    MIMO_BER_By_SNR = 'MIMO_BER_By_SNR'
    CostMIMODeepSIC = 'CostMIMODeepSIC'
    CostMIMODeepSICQPSK = 'CostMIMODeepSICQPSK'
    ModularCostMIMODeepSIC = 'ModularCostMIMODeepSIC'
    CostMIMODNN = 'CostMIMODNN'
    MultiDistortedMIMODeepSIC = 'MultiDistortedMIMODeepSIC'
    ModularMultiDistortedMIMODeepSIC = 'ModularMultiDistortedMIMODeepSIC'
    MultiDistortedMIMODNN = 'MultiDistortedMIMODNN'
    CostMIMODeepSICQAM16 = 'CostMIMODeepSICQAM16'


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
             'drift_detection_method_hp': {'ht_threshold': 0.005},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.ModularSingleUserDistortedMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
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
             'drift_detection_method_hp': {'ht_threshold': 0.005},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.SingleUserDistortedMIMODNN.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.black_box.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
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
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.CostMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 7.5, 'beta_ddm': 0.2},
             },
            # {'drift_detection_method': 'PHT',
            #  'drift_detection_method_hp': {'beta_pht': 0.9, 'delta_pht': 1, 'lambda_pht': 0.08},
            #  },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.015},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.CostMIMODeepSICQPSK.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name, 'modulation_type': 'QPSK',
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None, }
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 7.5, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.015},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.CostMIMODeepSICQAM16.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 16, 'detector_type': DetectorType.model.name, 'modulation_type': 'QAM16',
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 7.5, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.015},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.ModularCostMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 2, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.9, 'delta_pht': 0.08, 'lambda_pht': 0.03},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.003},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.CostMIMODNN.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.black_box.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 7.5, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.07},
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
            {'snr': 14, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.SEDChannel.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 7, 'beta_ddm': 0.3},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.7, 'delta_pht': 0.6, 'lambda_pht': 0.01},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.001},
             },
            {'drift_detection_method': 'PST',
             'drift_detection_method_hp': {'posterior_threshold': 0.0001},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.ModularDistortedMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 15, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.3, 'delta_pht': 0.6, 'lambda_pht': 0.15},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.05},
             },
            {'drift_detection_method': 'PST',
             'drift_detection_method_hp': {'posterior_threshold': 0.85},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.MultiDistortedMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 30, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.5},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.7, 'delta_pht': 1.3, 'lambda_pht': 0.15},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.ModularMultiDistortedMIMODeepSIC.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 8, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.07},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.7, 'delta_pht': 1, 'lambda_pht': 0.05},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.MultiDistortedMIMODNN.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.black_box.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 15, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.8},
             },
            # {'drift_detection_method': 'PHT',
            #  'drift_detection_method_hp': {'beta_pht': 0.8, 'delta_pht': 1, 'lambda_pht': 0.08},
            #  },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.SOFT_SER_BLOCK_LINEAR_COST.name:
        methods_list = [
            'PERIODIC',
            # 'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'fading_in_channel': False,
             'modular': True, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 5, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.3, 'delta_pht': 0.6, 'lambda_pht': 0.1},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.00001},
             },
            {'drift_detection_method': 'PST',
             'drift_detection_method_hp': {'posterior_threshold': 0.0005},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.LinearSISO.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9500, 'pilot_size': 500, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 3.5, 'beta_ddm': 0.2},  # 4;0.2
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.7, 'delta_pht': 1.3, 'lambda_pht': 0.02},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.LinearSISO_RNN.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.black_box.name,
             'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9000, 'pilot_size': 1000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None}
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 3, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.7, 'delta_pht': 1.2, 'lambda_pht': 0.02},
             },

        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.SISO_BER_By_SNR.name:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9500, 'pilot_size': 500, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9500, 'pilot_size': 500, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9500, 'pilot_size': 500, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9500, 'pilot_size': 500, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9500, 'pilot_size': 500, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
        ]
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 3.9, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.7, 'delta_pht': 1.3, 'lambda_pht': 0.02},
             },
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    elif label_name == PlotType.SISO_BER_By_SNR_RNN.name:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9000, 'pilot_size': 1000, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9000, 'pilot_size': 1000, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9000, 'pilot_size': 1000, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9000, 'pilot_size': 1000, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
            {'snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'block_length': 9000, 'pilot_size': 1000, 'drift_detection_method': None,
             'fading_in_channel': True, 'from_scratch': False, 'modular': False, },
        ]
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 3, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.7, 'delta_pht': 1.2, 'lambda_pht': 0.02},
             },
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    elif label_name == PlotType.MIMO_BER_By_SNR.name:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 10, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 11, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 13, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.DistortedMIMO.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None}
        ]
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 8, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'beta_pht': 0.3, 'delta_pht': 0.6, 'lambda_pht': 0.01},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.05},
             },
            {'drift_detection_method': 'PST',
             'drift_detection_method_hp': {'posterior_threshold': 0.85},
             },
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    elif label_name == PlotType.CostMIMODeepSICSNR.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None},
            {'snr': 10, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None},
            {'snr': 11, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None},
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None},
            {'snr': 13, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None},

        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 7.5, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.014},
             },
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    elif label_name == PlotType.ModularCostMIMODeepSICSNR.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 10, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 11, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},
            {'snr': 13, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': True, 'drift_detection_method_hp': None},

        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'alpha_ddm': 2, 'beta_ddm': 0.2},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.003},
             },
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    elif label_name == PlotType.CostMIMODNNSNR.name:
        methods_list = [
            # 'PERIODIC',
            # 'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.black_box.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'modular': False, 'drift_detection_method_hp': None},
            # {'snr': 10, 'detector_type': DetectorType.black_box.name,
            #  'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
            #  'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
            #  'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
            #  'modular': False, 'drift_detection_method_hp': None},
            # {'snr': 11, 'detector_type': DetectorType.black_box.name,
            #  'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
            #  'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
            #  'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
            #  'modular': False, 'drift_detection_method_hp': None},
            # {'snr': 12, 'detector_type': DetectorType.black_box.name,
            #  'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
            #  'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
            #  'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
            #  'modular': False, 'drift_detection_method_hp': None},
            # {'snr': 13, 'detector_type': DetectorType.black_box.name,
            #  'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
            #  'channel_model': ChannelModels.Cost21002nd.name, 'fading_in_channel': False,
            #  'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
            #  'modular': False, 'drift_detection_method_hp': None},

        ]
        drift_detection_methods = [
            # {'drift_detection_method': 'DDM',
            #  'drift_detection_method_hp': {'alpha_ddm': 7.5, 'beta_ddm': 0.2},
            #  },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 0.07},
             },
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    elif label_name == PlotType.SingleUserDistortedMIMODeepSICSNR.name:
        methods_list = [
            'PERIODIC',
            'ALWAYS',
            'DRIFT',
        ]
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'drift_detection_method_hp': None},
            {'snr': 10, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'drift_detection_method_hp': None},
            {'snr': 11, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'drift_detection_method_hp': None},
            {'snr': 12, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
             'block_length': 7000, 'pilot_size': 2000, 'drift_detection_method': None,
             'drift_detection_method_hp': None},
            {'snr': 13, 'detector_type': DetectorType.model.name,
             'channel_type': ChannelModes.MIMO.name, 'blocks_num': 100,
             'channel_model': ChannelModels.OneUserDistortedMIMOChannel.name,
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
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    else:
        raise ValueError('No such plot mechanism_type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel, plot_type, drift_detection_methods
