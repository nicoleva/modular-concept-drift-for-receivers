import math

import torch


class DriftPHT:

    def __init__(self, beta: float, delta: float, lambda_value: float):
        self.beta = beta
        self.delta = delta
        self.lambda_value = lambda_value
        self.mu_t_prev = 0
        self.previous_distance = None

    def check_drift(self, samples_vector: torch.Tensor):
        average = abs(samples_vector).mean()
        mu_t = self.beta * average + (1 - self.beta) * self.mu_t_prev
        norm = abs(samples_vector - mu_t)
        differences_vector = norm - self.delta
        differences_vector[differences_vector < 0] = 0
        distance = differences_vector.mean().item()
        if self.previous_distance is None:
            self.previous_distance = distance
        distance_diff = abs(distance - self.previous_distance)
        print(f'Threshold: {distance_diff}')
        if distance_diff != math.inf and distance_diff > self.lambda_value:
            self.mu_t_prev = average
            self.previous_distance = None
            return 1
        self.mu_t_prev = mu_t
        self.previous_distance = distance
        return 0
