
from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class TrainingConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self.batch_size = 512
        self.learning_rate = 0.001
        self.training_epochs = 20
        self.training_epochs_final = 50

        # Early stopping
        self.early_stopping = True
        self.patience = 5 # Changed from early_stop_patience
        self.early_stop_min_delta = 0.001
        self.early_stop_monitor = 'val_loss'
        self.early_stop_mode = 'min'

        # Checkpointing and saving
        self.save_step = 5,
        self.save_best_only = True
        self.save_dir = 'checkpoints' # Added save_dir

        # Learning rate scheduling
        self.use_lr_scheduling = True
        self.lr_schedule_factor = 0.5
        self.lr_schedule_patience = 3
        self.min_lr = 1e-6

        # Monitoring
        self.monitor_metrics = ['accuracy', 'loss', 'prototype_metrics']
        self.validation_split = 0.2
        self.shuffle = True

        for key in kwargs:
            if key in self.__dict__:
                self.__dict__[key] = kwargs[key]
            else:
                print(key, " doesn't exit in TrainingConfig")
