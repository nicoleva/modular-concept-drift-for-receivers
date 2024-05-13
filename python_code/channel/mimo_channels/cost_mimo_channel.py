import os

import numpy as np
import scipy.io

from dir_definitions import MIMO_COST2100_DIR
from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF

conf = Config()

MAX_FRAMES = 25
gain = {1:0.75, 2:0.5, 3:0.5, 4:0.25}

class Cost2100MIMOChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, fading: bool) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        main_folder = 1 + (frame_ind // MAX_FRAMES)
        for i in range(1, n_user + 1):
            path_to_mat = os.path.join(MIMO_COST2100_DIR, f'{main_folder}', f'h_{i}.mat')
            h_user = scipy.io.loadmat(path_to_mat)['norm_channel'][frame_ind % MAX_FRAMES, :N_USER]
            total_h[i - 1] = HALF * h_user

        total_h[np.arange(n_user), np.arange(n_user)] = 1
        return total_h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO COST2100 Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel coefficients
        :return: received word
        """
        conv = Cost2100MIMOChannel._compute_channel_signal_convolution(h, s)
        sigma = 10 ** (-0.1 * snr)
        w = np.sqrt(sigma) * np.random.randn(N_ANT, s.shape[1])
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv

class Cost2100MIMOChannel2nd:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, fading: bool) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        main_folder = 'cost2100'
        room = "_room_2" if (frame_ind >= 33 and frame_ind <= 66) else ""
        for i in range(1, n_user + 1):
            h_user = []
            for ant in range(1, n_user + 1):
                los = ant #if frame_ind > 51 else (ant+0)%n_user + 1
                path_to_mat = os.path.join(MIMO_COST2100_DIR, f'{main_folder}', f'h_user_{i}_ant_{ant}{room}.mat')
                h_user_ant = np.squeeze(np.array(scipy.io.loadmat(path_to_mat)['out']))
                h_user_ant = h_user_ant[::1]
                # if (i == los): # antenna pointing at user
                #     h_user_ant = h_user_ant + 10
                # else:
                #     h_user_ant = h_user_ant - 5 - los
                h_user_ant = np.array([10**(h/10) for h in h_user_ant])
                max_val = h_user_ant.max()
                if (i == los): # antenna pointing at user
                    h_user_ant = np.array(h_user_ant) / max_val
                    # if i == 3 and (frame_ind <33 or frame_ind >66):
                    #     h_user_ant = np.array(h_user_ant)/ max_val + 0.8
                    # else:
                    #     h_user_ant = np.array(h_user_ant) / max_val
                else:
                    h_user_ant = np.array(h_user_ant)*0.33/max_val
                h_user_ant = [i if i<1 else 1 for i in h_user_ant]
                h_user.append(h_user_ant)

            total_h[i - 1] = np.squeeze(np.array(h_user))[:, frame_ind%33]
            #total_h[i - 1] = np.squeeze(np.array(h_user))[:,frame_ind]

        #total_h[np.arange(n_user), np.arange(n_user)] = 1
        return total_h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO COST2100 Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel coefficients
        :return: received word
        """
        conv = Cost2100MIMOChannel._compute_channel_signal_convolution(h, s)
        sigma = 10 ** (-0.1 * snr)
        w = np.sqrt(sigma) * np.random.randn(N_ANT, s.shape[1])
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
