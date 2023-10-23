class DriftHT:

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.trained_on_last_block = False

    def check_drift(self, user_ht_value: float):
        # receives the ht value already
        print(f'HT value: {user_ht_value}')
        if user_ht_value > self.threshold and not self.trained_on_last_block:
            self.trained_on_last_block = True
            return 1
        self.trained_on_last_block = False
        return 0
