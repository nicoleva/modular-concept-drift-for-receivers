from python_code.drift_mechanisms.drift_mechanism_wrapper import TRAINING_TYPES
from python_code.plotters.plotter_config import get_config, PlotType
from python_code.plotters.plotter_methods import compute_ser_for_method, RunParams
from python_code.plotters.plotter_utils import plot_by_values

if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results  of multiple runs
    run_params_obj = RunParams(run_over=run_over,
                               trial_num=trial_num)
    #label_name = PlotType.MIMO_BER_By_SNR
    #label_name = PlotType.SingleUserDistortedMIMODeepSIC
    #label_name = PlotType.ModularSingleUserDistortedMIMODeepSIC
    #label_name = PlotType.SingleUserDistortedMIMODNN
    #label_name = PlotType.DistortedMIMODeepSIC
    #label_name = PlotType.SOFT_SER_BLOCK_LINEAR_COST
    #label_name = PlotType.SISO_BER_By_SNR
    label_name = PlotType.SISO_BER_By_SNR_RNN
    #label_name = PlotType.LinearSISO
    #label_name = PlotType.LinearSISO_RNN
    #label_name = PlotType.CostMIMODeepSIC
    #label_name = PlotType.ModularCostMIMODeepSIC
    #label_name = PlotType.CostMIMODNN
    #label_name = PlotType.ModularSingleUserDistortedMIMODeepSICSNR
    #label_name = PlotType.ModularCostMIMODeepSICSNR
    #label_name = PlotType.CostMIMODeepSICSNR
    #label_name = PlotType.CostMIMODNNSNR
    #label_name = PlotType.SingleUserDistortedMIMODeepSICSNR
    #label_name = PlotType.MultiDistortedMIMODeepSIC
    #label_name = PlotType.ModularMultiDistortedMIMODeepSIC
    #label_name = PlotType.MultiDistortedMIMODNN


    print(label_name.name)
    params_dicts, methods_list, values, xlabel, ylabel, plot_type, drift_methods_list = get_config(label_name.name)
    all_curves = []

    for method in methods_list:
        print(method)
        for params_dict in params_dicts:
            print(params_dict)
            if method == TRAINING_TYPES.DRIFT.name:
                for drift_detection_params in drift_methods_list:
                    # set the drift detection method
                    params_dicts_drift = params_dict
                    params_dicts_drift['drift_detection_method'] = drift_detection_params['drift_detection_method']
                    params_dicts_drift['drift_detection_method_hp'] = \
                        drift_detection_params['drift_detection_method_hp']
                    compute_ser_for_method(all_curves, method, params_dicts_drift, run_params_obj)
            else:
                compute_ser_for_method(all_curves, method, params_dict, run_params_obj)
    plot_by_values(all_curves, values, xlabel, ylabel, plot_type)
