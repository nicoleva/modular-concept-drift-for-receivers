import numpy as np

from python_code.channel.channels_hyperparams import N_ANT
from python_code.utils.config_singleton import Config

conf = Config()

CHANNEL_FLIPS_EVERY = {0: 7, 1: 18, 2: 22, 3: 28}
C = 2

class DistortedMIMOChannel:
    def __init__(self):
        # Antennas (rows) X Users (columns)
        self.h = np.array([[0.8, 0.4, 0.2, 0.1],
                           [0.4, 0.8, 0.4, 0.2],
                           [0.2, 0.4, 0.8, 0.4],
                           [0.1, 0.2, 0.4, 0.8]])

    def calculate_channel(self, n_user: int, frame_ind: int) -> np.ndarray:
        for user in range(n_user):
            if frame_ind > 0 and frame_ind % CHANNEL_FLIPS_EVERY[user] == 0:
                self.h[:, user] *= C ** ((-1) ** (frame_ind // CHANNEL_FLIPS_EVERY[user]))
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
        conv = DistortedMIMOChannel._compute_channel_signal_convolution(h, s)
        sigma = 10 ** (-0.1 * snr)
        w = np.sqrt(sigma) * np.random.randn(N_ANT, s.shape[1])
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
