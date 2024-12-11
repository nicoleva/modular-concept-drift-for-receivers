import math
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.channel.modulator import MODULATION_NUM_MAPPING, MODULATION_DICT, prob_to_BPSK_symbol
from python_code.detectors.deepsic.deep_sic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.drift_mechanisms.drift_mechanism_wrapper import TRAINING_TYPES
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF, ModulationType
from python_code.utils.hotelling_test_utils import run_hotelling_test

conf = Config()
ITERATIONS = 3
EPOCHS = 250


class DeepSICTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        self.lr = 1e-3
        self.ht = [0] * N_USER
        self.prev_ht = {symbol: [[[] for _ in range(ITERATIONS)] for _ in range(self.n_user)] for symbol in
                        range(MODULATION_NUM_MAPPING[conf.modulation_type])}
        self.constellation_bits = int(math.log2(MODULATION_NUM_MAPPING[conf.modulation_type]))
        self.data_size = int((conf.block_length - conf.pilot_size) / self.constellation_bits)
        self.pilot_size = int(conf.pilot_size / self.constellation_bits)
        super().__init__()

    def __str__(self):
        name = 'DeepSIC'
        if conf.mechanism == TRAINING_TYPES.DRIFT.name and conf.modular:
            name = 'Modular ' + name
        return name

    def _initialize_probs_for_infer(self, rx):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(rx.shape).to(DEVICE).float()
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.QAM16.name]:
            probs_vec = (1 / MODULATION_NUM_MAPPING[conf.modulation_type]) * torch.ones(rx.shape).to(DEVICE).unsqueeze(
                -1)
            probs_vec = probs_vec.repeat([1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1]).float()
        else:
            raise ValueError("No such constellation!")
        return probs_vec

    def _initialize_detector(self):
        self.detector = [[DeepSICDetector().to(DEVICE) for _ in range(ITERATIONS)] for _ in
                         range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=tx.long())

    @staticmethod
    def preprocess(rx: torch.Tensor) -> torch.Tensor:
        if conf.modulation_type == ModulationType.BPSK.name:
            return rx.float()
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.QAM16.name]:
            y_input = torch.view_as_real(rx[:, :conf.n_ant]).float().reshape(rx.shape[0], -1)
            return torch.cat([y_input, rx[:, conf.n_ant:].float()], dim=1)
        else:
            raise ValueError("No such constellation type!")

    def train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        loss = 0
        y_total = self.preprocess(rx)
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total)
            current_loss = self.run_train_loop(soft_estimation, tx)
            loss += current_loss

    def _initialize_probs_for_training(self, tx):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(tx.shape).to(DEVICE)
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.QAM16.name]:
            probs_vec = (1 / MODULATION_NUM_MAPPING[conf.modulation_type]) * torch.ones(tx.shape).to(DEVICE).unsqueeze(
                -1).repeat([1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1])
        else:
            raise ValueError("No such constellation!")
        return probs_vec

    def train_models(self, model: List[List[DeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(self.n_user):
            if not self.train_users_list[user]:
                continue
            self.train_model(model[user][i], tx_all[user], rx_all[user])

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if conf.mechanism == TRAINING_TYPES.DRIFT:
            self._initialize_detector()
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(ITERATIONS):
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, tx_all, rx_all)
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)

    def compute_output(self, probs_vec):
        if conf.modulation_type == ModulationType.BPSK.name:
            symbols_word = prob_to_BPSK_symbol(probs_vec.float())
            detected_words = MODULATION_DICT[conf.modulation_type].demodulate(symbols_word)
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.QAM16.name]:
            first_user_probs = 1 - torch.sum(probs_vec, dim=2).unsqueeze(-1)
            all_probs = torch.cat([first_user_probs, probs_vec], dim=2)
            detected_words = torch.argmax(all_probs, dim=2)
        else:
            raise ValueError("No such constellation!")
        return detected_words

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # detect and decode
            probs_vec = self._initialize_probs_for_infer(rx)
            for i in range(ITERATIONS):
                probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
            detected_words = self.compute_output(probs_vec)
            return detected_words

    def prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != k]
            current_y_train = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all

    def calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                             rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self.preprocess(input)
            with torch.no_grad():
                output = self.softmax(model[user][i - 1](preprocessed_input))
            next_probs_vec[:, user] = output[:, 1:]
        return next_probs_vec

    def forward_pilot(self, rx: torch.Tensor, tx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.pilots_probs_vec = self._initialize_probs_for_infer(rx)
        constellation_size = MODULATION_NUM_MAPPING[conf.modulation_type]

        # Dynamically create structures based on constellation size
        ht_t_0 = {symbol: [[[] for _ in range(ITERATIONS)] for _ in range(self.n_user)] for symbol in
                  range(MODULATION_NUM_MAPPING[conf.modulation_type])}
        ht_mat = [[[] for _ in range(ITERATIONS)] for _ in range(self.n_user)]

        for i in range(ITERATIONS):
            # Calculate posterior probabilities for each iteration
            self.pilots_probs_vec = self.calculate_posteriors(self.detector, i + 1, self.pilots_probs_vec, rx)

            for user in range(self.n_user):
                # Separate received indices based on symbols in the constellation
                for symbol in range(constellation_size):
                    rx_symbol_idx = [i for i, x in enumerate(tx[:, user]) if x == symbol]
                    # BPSK case
                    if constellation_size == 2:
                        ht_t_0[symbol][user][i] = self.pilots_probs_vec[rx_symbol_idx, user].cpu().numpy()
                    # Higher order modulation case - extract the symbolwise probs vector from the total probs vector
                    # if it is the last symbol in the constellation - its probs is 1 - (all other probs)
                    else:
                        # first symbol takes its prob as 1 - all other probs
                        if symbol == 0:
                            all_other_probs = np.sum(self.pilots_probs_vec[rx_symbol_idx, user].cpu().numpy(), axis=1)
                            ht_t_0[symbol][user][i] = 1 - all_other_probs
                        else:
                            ht_t_0[symbol][user][i] = self.pilots_probs_vec[rx_symbol_idx, user, symbol - 1].cpu().numpy()

                # Run hypothesis testing if previous distributions are available
                if np.shape(self.prev_ht[0][user][i])[0] != 0:
                    run_hotelling_test(
                        ht_mat, ht_t_0, self.prev_ht, i, tx, user, constellation_size
                    )

                # Save previous distributions for each symbol
                for symbol in range(constellation_size):
                    self.prev_ht[symbol][user][i] = ht_t_0[symbol][user][i].copy()

        if np.prod(np.shape(ht_mat[self.n_user - 1][ITERATIONS - 1])) != 0:
            self.ht = [row[ITERATIONS - 1] for row in ht_mat]

        # Detect symbols based on the constellation
        detected_words = self.compute_output(self.pilots_probs_vec.float())
        return detected_words, self.pilots_probs_vec
