from dataclasses import dataclass

from typing_extensions import overload

from .base_config import BaseConfig

@dataclass
class ModelConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()

        # Model architecture
        self.input_dim = None
        self.latent_dim = 32
        self.n_prototypes = 15
        self.num_classes = 2
        self.random_seed = 42

        # Loss coefficients
        self.lambda_class = 10.0
        self.lambda_ae = 1.0
        self.lambda_1 = 1.0
        self.lambda_2 = 1.0
        self.lambda_diversity = 2.0

        # Architecture settings
        self.intermediate_dims: list[int] = [85, 42]
        self.activation = 'relu'
        self.activation_out = 'sigmoid'
        self.use_batch_norm = True

        for key in kwargs:
            if key in self.__dict__:
                self.__dict__[key] = kwargs[key]
            else:
                print(key, " doesn't exit in ModelConfig")

