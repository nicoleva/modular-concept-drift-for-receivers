import os
from collections import namedtuple
from typing import Tuple, List, Dict, Union

from dir_definitions import CONFIG_RUNS_DIR
from python_code.detectors.trainer import Trainer
from python_code.drift_mechanisms.drift_mechanism_wrapper import TRAINING_TYPES
from python_code.evaluate import CHANNEL_TYPE_TO_TRAINER_DICT
from python_code.plotters.plotter_utils import get_ser_plot, get_channel_h
from python_code.utils.config_singleton import Config

RunParams = namedtuple(
    "RunParams",
    "run_over trial_num",
    defaults=[False, 1]
)


def set_method_name(conf: Config, method_name: str, params_dict: Dict[str, Union[int, str]]) -> str:
    """
    Set values of params dict to current config. And return the field and their respective values as the name of the run,
    used to save as pkl file for easy access later.
    :param conf: config file.
    :param method_name: the desired augmentation scheme name
    :param params_dict: the run params
    :return: name of the run
    """
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        if field == 'drift_detection_method_hp':
            continue  # dont include hp in file name, otherwise the file name is too long
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)
    return name


def add_ser(all_curves: List[Tuple[List[float], str]], conf: Config, method_name: str, name: str, run_over: bool,
            trial_num: int, trainer: Trainer):
    """
    Run the experiments #trial_num times, averaging over the whole run's aggregated ser.
    """
    total_ser = []
    total_idn_trained = []
    for trial in range(trial_num):
        conf.set_value('seed', 1 + trial)
        trainer.__init__()
        ser, idn_trained = get_ser_plot(trainer, run_over=run_over,
                                        method_name=method_name + name,
                                        trial=trial)
        total_ser.append(ser)
        total_idn_trained.append(idn_trained)
    all_curves.append((total_ser, method_name, total_idn_trained))
    get_channel_h(trainer)


def compute_ser_for_method(all_curves: List[Tuple[float, str]], method: str, params_dict: Dict[str, Union[int, str]],
                           run_params_obj: RunParams):
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, params_dict['channel_type'], f'{method}.yaml'))
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[params_dict['channel_type']][params_dict['detector_type']]()
    full_method_name = f'{trainer.__str__()} - {method}'
    print(full_method_name)
    name = set_method_name(conf, full_method_name, params_dict)
    if method == TRAINING_TYPES.DRIFT.name:
        full_method_name += f'{params_dict["drift_detection_method"]}'
    add_ser(all_curves, conf, full_method_name, name, run_params_obj.run_over, run_params_obj.trial_num, trainer)
