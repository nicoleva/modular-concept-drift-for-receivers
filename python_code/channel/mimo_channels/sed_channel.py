import numpy as np

from python_code.channel.channels_hyperparams import N_ANT, N_USER

H_COEF = 0.8


class SEDChannel:
    def __init__(self):
        h_row = np.array([i for i in range(N_ANT)])
        h_row = np.tile(h_row, [N_USER, 1]).T
        h_column = np.array([i for i in range(N_USER)])
        h_column = np.tile(h_column, [N_ANT, 1])
        self.h = np.exp(-np.abs(h_row - h_column))

    def calculate_channel(self, n_ant: int, frame_ind: int) -> np.ndarray:
        cur_h = SEDChannel._add_fading(self.h, n_ant, frame_ind)
        return cur_h

    @staticmethod
    def _add_fading(h: np.ndarray, n_ant: int, frame_ind: int) -> np.ndarray:
        degs_array = np.array([83, 76, 55, 32])
        fade_mat = H_COEF + (1 - H_COEF) * np.cos(2 * np.pi * frame_ind / degs_array)
        fade_mat = np.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return h * fade_mat

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel function
        :return: received word
        """
        conv = SEDChannel._compute_channel_signal_convolution(h, s)
        var = 10 ** (-0.1 * snr)
        w = np.sqrt(var) * np.random.randn(N_ANT, s.shape[1])
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
