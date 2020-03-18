class Meter:
    def __init__(self, tqdm):
        self.tqdm = tqdm

        self.loss = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    