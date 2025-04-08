
from dataclasses import dataclass

@dataclass
class BaseConfig:
    def __init__(self):
        self.GPUID = 1
        self.random_seed = 42
        self.verbose = True