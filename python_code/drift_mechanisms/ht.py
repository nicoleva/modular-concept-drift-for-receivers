class DriftHT:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def check_drift(self, user_ht_value: float):
        # receives the ht value already
        print(f'HT value: {user_ht_value}')
        if user_ht_value > self.threshold:
            return 1
        return 0
