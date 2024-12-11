import numpy as np

from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ModulationType

conf = Config()

CHANNEL_FLIPS_EVERY = {0: [6, 22, 46, 61, 85], 1: [], 2: [], 3: []}
C = 2


class OneUserDistortedMIMOChannel:
    def __init__(self):
        # Antennas (rows) X Users (columns)
        self.h = np.array([[0.8, 0.4, 0.2, 0.1],
                           [0.4, 0.8, 0.4, 0.2],
                           [0.2, 0.4, 0.8, 0.4],
                           [0.1, 0.2, 0.4, 0.8]])
        self.users_factors = C * np.ones(N_USER)

    def calculate_channel(self, n_user: int, frame_ind: int) -> np.ndarray:
        for user in range(n_user):
            if frame_ind in CHANNEL_FLIPS_EVERY[user]:
                self.h[:, user] *= self.users_factors[user]
                self.users_factors[user] = 1 / self.users_factors[user]
        return self.h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO COST2100 Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel coefficients
        :return: received word
        """
        conv = OneUserDistortedMIMOChannel._compute_channel_signal_convolution(h, s)
        sigma = 10 ** (-0.1 * snr)
        if conf.modulation_type == ModulationType.BPSK.name:
            w = np.sqrt(sigma) * np.random.randn(N_ANT, s.shape[1])
        else:
            w_real = np.sqrt(sigma) / 2 * np.random.randn(N_ANT, s.shape[1])
            w_imag = np.sqrt(sigma) / 2 * np.random.randn(N_ANT, s.shape[1]) * 1j
            w = w_real + w_imag
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
