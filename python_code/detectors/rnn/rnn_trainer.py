from random import randint

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.detectors.rnn.rnn_detector import RNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.trellis_utils import calculate_siso_states

conf = Config()
EPOCHS = 1000
BATCH_SIZE = 64


class RNNTrainer(Trainer):
    """
    Trainer for the RNNTrainer model.
    """

    def __init__(self):
        self.memory_length = MEMORY_LENGTH
        self.n_user = 1
        self.n_ant = 1
        self.lr = 5e-3
        super().__init__()

    def __str__(self):
        return 'RNN Detector'

    def _initialize_detector(self):
        """
        Loads the RNN detector
        """
        self.detector = RNNDetector()

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est: [1,transmission_length,n_states], each element is a probability
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_siso_states(self.memory_length, tx)
        loss = self.criterion(input=est, target=gt_states)
        return loss

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase='val')
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous weights, or from scratch.
        :param tx: transmitted word
        :param rx: received word
        """
        self.deep_learning_setup()
        # run training loops
        loss = 0
        for i in range(EPOCHS):
            ind = randint(a=0, b=conf.pilot_size - BATCH_SIZE)
            # pass through detector
            soft_estimation = self.detector(rx[ind: ind + BATCH_SIZE].float(), phase='train')
            current_loss = self.run_train_loop(est=soft_estimation, tx=tx[ind:ind + BATCH_SIZE])
            loss += current_loss
