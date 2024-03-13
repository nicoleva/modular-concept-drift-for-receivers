import datetime
import os
from typing import List, Tuple, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dir_definitions import FIGURES_DIR, PLOTS_DIR
from python_code.detectors.trainer import Trainer, alarms_dict
from python_code.drift_mechanisms.drift_mechanism_wrapper import TRAINING_TYPES
from python_code.utils.config_singleton import Config
from python_code.utils.constants import DetectorType
from python_code.utils.python_utils import load_pkl, save_pkl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

conf = Config()

h_channel = []


def get_linestyle(method_name: str) -> str:
    model_bb = str(conf.detector_type)
    if DetectorType.model.name in model_bb:
        return 'solid'
    elif DetectorType.black_box.name in model_bb:
        return 'dotted'
    else:
        raise ValueError('No such detector!!!')


def get_marker(method_name: str) -> str:
    if TRAINING_TYPES.ALWAYS.name in method_name:
        return '.'
    elif TRAINING_TYPES.PERIODIC.name in method_name:
        return 's'
    elif 'DDM' in method_name:
        return 'X'
    elif 'PHT' in method_name:
        return 'x'
    elif 'HT' in method_name:
        return 'P'
    elif 'PST' in method_name:
        return 'P'
    else:
        raise ValueError('No such method!!!')


def get_color(method_name: str) -> str:
    if TRAINING_TYPES.ALWAYS.name in method_name:
        return 'b'
    elif TRAINING_TYPES.PERIODIC.name in method_name:
        return 'green'
    elif 'DDM' in method_name:
        return 'red'
    elif 'PHT' in method_name:
        return 'chocolate'
    elif 'HT' in method_name:
        return 'orange'
    elif 'PST' in method_name:
        return 'pink'
    else:
        raise ValueError('No such method!!!')


LABELS_DICT = {
    'ALWAYS': 'Always',
    'PERIODIC': 'Periodic',
    'DRIFTPST': 'PST',
    'DRIFTHT': 'HT',
    'DRIFTDDM': 'DDM',
    'DRIFTPHT': 'PHT',
}


def get_ser_plot(dec: Trainer, run_over: bool, method_name: str, trial=None):
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = method_name
    if trial is not None:
        file_name = file_name + '_' + str(trial)
    plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path) and not run_over:
        print("Loading plots")
        ser_total, block_idn_trained = load_pkl(plots_path)
    else:
        # otherwise - run again
        print("calculating fresh")
        ser_total, block_idn_trained = dec.evaluate()
        save_pkl(plots_path, (ser_total, block_idn_trained))
    return ser_total, block_idn_trained


def plot_by_values(all_curves: List[Tuple[np.ndarray, str, np.ndarray]], values: List[float], xlabel: str,
                   ylabel: str, plot_type: str):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    names = []
    for i in range(len(all_curves)):
        if TRAINING_TYPES.DRIFT.name in all_curves[i][1]:
            name = all_curves[i][1].split(" - ")[1].split("_")[0]
            if conf.modular:
                name = name + " - Modular"
            names.append(name)
        elif TRAINING_TYPES.PERIODIC.name in all_curves[i][1]:
            names.append(all_curves[i][1].split(" - ")[1] + f' ({conf.period})')
        else:
            names.append(all_curves[i][1].split(" - ")[1])

    cur_name, sers_dict, train_annotation_dict, total_actions_dict = populate_sers_dict(all_curves, names, plot_type)
    if plot_type == 'plot_ber_aggregated':
        plot_ber_aggregated(names, sers_dict, train_annotation_dict, values, folder_name, total_actions_dict)
    elif plot_type == 'plot_by_snrs':
        plot_ber_vs_snr(names, sers_dict, train_annotation_dict, values, folder_name, total_actions_dict, cur_name)
    else:
        raise ValueError("No such plot type!")

    plot_channel()


def populate_sers_dict(all_curves: List[Tuple[float, str]], names: List[str], plot_type: str) -> Tuple[
    str, Dict[str, List[np.ndarray]]]:
    sers_dict = {}
    total_actions_dict = {}
    train_annotation_dict = {}
    for method_name in names:
        sers_list = []
        method_name_reduced = method_name.split(" ")[0]
        for ser, cur_name, train_idn in all_curves:
            if method_name_reduced not in cur_name:
                continue
            if method_name_reduced == TRAINING_TYPES.ALWAYS.name:
                train_annotation_dict[method_name] = list(range(0, len(train_idn[0]), 2))
            else:
                train_annotation_dict[method_name] = np.argwhere(train_idn[0])[:, 0]
            # depending on plot, add the relevant ser manipulations: (aggregated/ with instantaneous etc)
            avg_ser_trials = np.sum(ser, axis=0) / len(ser)
            if plot_type == 'plot_ber_aggregated':
                agg_ser = (np.cumsum(avg_ser_trials) / np.arange(1, len(ser[0]) + 1))
                sers_dict[method_name] = agg_ser
                total_actions_dict[method_name] = float(np.sum(train_idn) / len(train_idn))
            elif plot_type == 'plot_by_snrs':
                mean_ser = np.mean(ser)
                sers_list.append(mean_ser)
                sers_dict[method_name] = sers_list
                total_actions_dict[method_name] = float(np.sum(train_idn) / len(train_idn))
            else:
                raise ValueError("No such plot mechanism_type!")

    return cur_name, sers_dict, train_annotation_dict, total_actions_dict


def plot_common_aggregated(names: List[str], sers_dict: Dict[str, np.ndarray], annotation_dict: Dict[str, np.ndarray],
                           values: List[float], total_actions_dict: Dict[str, List]):
    for method_name in names:
        if 'Modular' in method_name:
            modular_label = "Modular "
        else:
            modular_label = ""
        plt.plot(values, sers_dict[method_name],
                 label=modular_label + LABELS_DICT[method_name.split(' ')[0]] + " " + f'[{total_actions_dict[method_name]}]',
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=16,
                 linestyle=get_linestyle(method_name), linewidth=3.2,
                 markevery=annotation_dict[method_name])
        marker_idx = annotation_dict[method_name]
        marker_annotation = [count + 1 for count, x in enumerate(marker_idx)]
        if 'Always' in method_name:
            continue
        elif 'DeepSIC' in method_name:
            if 'DriftDetectionDriven' in method_name:
                drift_method = method_name.split('- ')[2].split(' {')[0]
                num_users = len(alarms_dict[drift_method])
                count_alarms = 0
                for i in range(num_users):
                    count_alarms = count_alarms + alarms_dict[drift_method][i].count(1)
                marker_annotation[-1] = count_alarms
            elif 'Periodic' in method_name:
                try:
                    marker_annotation[-1] = marker_annotation[-1] * num_users
                except:
                    print("loaded from pkl")
    plt.yscale('log')
    plt.legend(loc='upper left', prop={'size': 18})
    plt.ylabel("Agg. BER")
    plt.xlabel("Block Index")
    major_ticks = np.arange(0, conf.blocks_num, 20)
    major_ticks = np.append(major_ticks, conf.blocks_num)
    plt.xticks(major_ticks)
    plt.grid(alpha=0.6, axis='both', visible=True)


def plot_ber_aggregated(names: List[str], sers_dict: Dict[str, np.ndarray], annotation_dict: Dict[str, np.ndarray],
                        values: List[float], folder_name: str, total_actions_dict: Dict[str, List]):
    plt.figure()
    plot_common_aggregated(names, sers_dict, annotation_dict, values, total_actions_dict)
    method_name = names[0]
    trainer_name = method_name.split(' ')[0]
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'aggr_inst_{trainer_name}_snr={str(conf.snr)}'
                                                       f'_pilots={str(conf.pilot_size)}'
                                                       f'_blocklen={str(conf.block_length)}.png'), bbox_inches='tight')
    plt.show()

def plot_ber_vs_snr(names: List[str], sers_dict: Dict[str, np.ndarray], annotation_dict: Dict[str, np.ndarray],
                        values: List[float], folder_name: str, total_actions_dict: Dict[str, List], cur_name: str):
    plt.figure()
    MARKER_EVERY = 1
    x_ticks = values
    x_labels = values
    # plots all methods
    unique_names = list(set(names))

    for method_name in unique_names:
        if 'Modular' in method_name:
            modular_label = "Modular "
        else:
            modular_label = ""
        plt.plot(values, sers_dict[method_name],
                 label=modular_label + LABELS_DICT[method_name.split(' ')[0]] + " " + f'[{total_actions_dict[method_name]}]',
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2,
                 markevery=MARKER_EVERY)

    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.yscale('log')
    trainer_name = cur_name.split(' ')[0]
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_snrs_{trainer_name}.png'),
                bbox_inches='tight')
    plt.show()
def get_channel_h(dec: Trainer):
    transmitted_words, received_words, hs = dec.channel_dataset.__getitem__(snr_list=[conf.snr])
    global h_channel
    h_channel = hs

def plot_channel():
    mimo = 0
    if mimo:
        h_channel_np = np.array(h_channel)
        fig, ax = plt.subplots(nrows=4, ncols=1)
        user = 0
        for row in ax:
            row.plot(h_channel_np[:, :, user])
            user = user+1
    else: #siso channel
        h_channel_np = np.array(h_channel)
        plt.figure()
        for i in range(h_channel_np.shape[2]):
            plt.plot(h_channel_np[:, :, i])

    plt.xlabel("Block Index")
    plt.ylabel("Magnitude")
    plt.ylim(top=1.0)
    plt.grid()

    plt.show()