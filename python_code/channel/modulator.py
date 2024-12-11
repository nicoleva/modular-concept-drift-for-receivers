import numpy as np
import torch

from python_code import DEVICE
from python_code.utils.constants import HALF


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - HALF)


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
        1: -(1 / np.sqrt(2)) + (1 / np.sqrt(2)) * 1j,
        2: (1 / np.sqrt(2)) - (1 / np.sqrt(2)) * 1j,
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
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        """
        QPSK demodulation using inverse mapping
        :param s: modulated signal tensor of shape (n_user, batch_size) with complex symbols
        :return: demodulated matrix of shape (n_user, batch_size) with integer values (0, 1, 2, 3)
        """
        # Define reverse mapping based on distance to constellation points
        symbols = torch.tensor(list(QPSKModulator.SYMBOLS.values()), dtype=torch.complex64).to(DEVICE)
        distances = torch.abs(s.unsqueeze(-1) - symbols)
        demodulated = torch.argmin(distances, dim=-1)

        return demodulated


class QAM16Modulator:
    constellation_size = 16

    # Define the 16-QAM constellation mapping
    SYMBOLS = {
        0: -3 - 3j, 1: -3 - 1j, 2: -3 + 3j, 3: -3 + 1j,
        4: -1 - 3j, 5: -1 - 1j, 6: -1 + 3j, 7: -1 + 1j,
        8: 3 - 3j, 9: 3 - 1j, 10: 3 + 3j, 11: 3 + 1j,
        12: 1 - 3j, 13: 1 - 1j, 14: 1 + 3j, 15: 1 + 1j
    }

    # Normalize constellation points
    SYMBOLS = {k: v / np.sqrt(10) for k, v in SYMBOLS.items()}

    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        16-QAM modulation using mapping vectors.
        :param c: input matrix of shape (n_user, batch_size) with integer values (0 to 15)
        :return: modulated signal matrix of shape (n_user, batch_size) with complex symbols
        """
        # Map each integer in `c` to the corresponding 16-QAM symbol
        modulated_signal = np.vectorize(QAM16Modulator.SYMBOLS.get)(c)
        return modulated_signal

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        """
        16-QAM demodulation using inverse mapping.
        :param s: modulated signal tensor of shape (n_user, batch_size) with complex symbols
        :return: demodulated matrix of shape (n_user, batch_size) with integer values (0 to 15)
        """
        # Define reverse mapping based on distance to constellation points
        symbols = torch.tensor(list(QAM16Modulator.SYMBOLS.values()), dtype=torch.complex64).to(s.device)
        distances = torch.abs(s.unsqueeze(-1) - symbols)
        demodulated = torch.argmin(distances, dim=-1)

        return demodulated


MODULATION_DICT = {
    'BPSK': BPSKModulator,
    'QPSK': QPSKModulator,
    'QAM16': QAM16Modulator
}

MODULATION_NUM_MAPPING = {
    'BPSK': 2,
    'QPSK': 4,
    'QAM16': 16
}
