import torch


class DriftPosterior:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def check_drift(self, tx_pilot: torch.Tensor, probs_vec: torch.Tensor):
        vec_0 = 1 - probs_vec[tx_pilot == 0]
        vec_1 = probs_vec[tx_pilot == 1]
        threshold_0 = self.calc_threshold(vec_0)
        threshold_1 = self.calc_threshold(vec_1)
        total_threshold = (vec_0.shape[0] * threshold_0 + vec_1.shape[0] * threshold_1) / probs_vec.shape[0]
        print(f'Threshold: {total_threshold}')
        if total_threshold < self.threshold:
            return True
        return False

    def calc_threshold(self, s_probs_vec):
        mu = s_probs_vec.mean()
        return mu
