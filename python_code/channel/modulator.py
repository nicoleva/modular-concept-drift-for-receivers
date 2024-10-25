import numpy as np
import torch

from python_code.utils.constants import HALF


class BPSKModulator:
    constellation_size = 2

    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        BPSK modulation 0->1, 1->-1
        :param c: the binary codeword
        :return: binary modulated signal
        """
        x = 1 - 2 * c
        return x

    @staticmethod
    def demodulate(s: np.ndarray) -> np.ndarray:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """
        return HALF * (s + 1)


class QPSKModulator:
    constellation_size = 4

    # Define the QPSK constellation mapping
    SYMBOLS = {
        0: (1 / np.sqrt(2)) + (1 / np.sqrt(2)) * 1j,
        1: (1 / np.sqrt(2)) - (1 / np.sqrt(2)) * 1j,
        2: -(1 / np.sqrt(2)) + (1 / np.sqrt(2)) * 1j,
        3: -(1 / np.sqrt(2)) - (1 / np.sqrt(2)) * 1j
    }

    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        QPSK modulation using mapping vectors
        :param c: input matrix of shape (n_user, batch_size) with integer values (0, 1, 2, 3)
        :return: modulated signal matrix of shape (n_user, batch_size) with complex symbols
        """
        # Map each integer in `c` to the corresponding QPSK symbol
        modulated_signal = np.vectorize(QPSKModulator.SYMBOLS.get)(c)
        return modulated_signal

    @staticmethod
    def demodulate(s: np.ndarray) -> np.ndarray:
        """
        QPSK demodulation using inverse mapping
        :param s: modulated signal array of shape (n_user, batch_size) with complex symbols
        :return: demodulated matrix of shape (n_user, batch_size) with integer values (0, 1, 2, 3)
        """
        # Create an array of constellation points for vectorized distance calculation
        constellation_points = np.array(list(QPSKModulator.SYMBOLS.values()))

        # Calculate distance from each point in s to each constellation point
        distances = np.abs(s[:, :, np.newaxis] - constellation_points)

        # Find the index of the nearest constellation point
        demodulated = np.argmin(distances, axis=2)

        return demodulated

class EightPSKModulator:
    constellation_size = 8

    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        8PSK modulation
        [0,0,0] -> [-1,0]
        [0,0,1] -> [-1/sqrt(2),-1/sqrt(2)]
        [0,1,0] -> [-1/sqrt(2),1/sqrt(2)]
        [0,1,1] -> [0,-1]
        [1,0,0] -> [0,1]
        [1,0,1] -> [1/sqrt(2),-1/sqrt(2)]
        [1,1,0] -> [1/sqrt(2),1/sqrt(2)]
        [1,1,1] -> [1,0]
        :param c: the binary codeword
        :return: modulated signal
        """
        deg = (c[:, ::3] / 4 + c[:, 1::3] / 2 + c[:, 2::3]) * math.pi
        x = np.exp(1j * deg)
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(s.imag, s.real) / np.pi
        theta[theta < 0] += 2
        c1 = torch.div(theta, 0.25, rounding_mode='floor') % 2
        c2 = torch.div(theta, 0.5, rounding_mode='floor') % 2
        c3 = torch.div(theta, 1, rounding_mode='floor') % 2
        concat_cs = torch.zeros(3 * s.shape[0], s.shape[1]).to(DEVICE)
        concat_cs[::3, :] = c1
        concat_cs[1::3, :] = c2
        concat_cs[2::3, :] = c3
        return concat_cs


MODULATION_DICT = {
    'BPSK': BPSKModulator,
    'QPSK': QPSKModulator,
    'EightPSK': EightPSKModulator
}

MODULATION_NUM_MAPPING = {
    'BPSK': 2,
    'QPSK': 4,
    'EightPSK': 8
}

MODULATION_BASE_SIZE_MAPPING = {
    'BPSK': 32,
    'QPSK': 16,
}
