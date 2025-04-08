import tensorflow as tf
from typing import Optional, Dict, Union, Tuple


class ReconstructionLoss(tf.keras.losses.Loss):
    """Mean squared error reconstruction loss with optional feature weighting."""

    def __init__(
            self,
            feature_weights: Optional[tf.Tensor] = None,
            name: str = 'reconstruction_loss'
    ):
        super().__init__(name=name)
        self.feature_weights = feature_weights

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute weighted reconstruction loss."""
        squared_diff = tf.square(y_true - y_pred)

        if self.feature_weights is not None:
            squared_diff = squared_diff * self.feature_weights

        return tf.reduce_mean(squared_diff)


class VariationalLoss(tf.keras.losses.Loss):
    """Kullback-Leibler divergence loss for variational autoencoders."""

    def __init__(
            self,
            beta: float = 1.0,
            name: str = 'variational_loss'
    ):
        super().__init__(name=name)
        self.beta = beta

    def call(self, z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
        """Compute KL divergence loss."""
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        return self.beta * kl_loss


class ContrastiveLoss(tf.keras.losses.Loss):
    """Contrastive loss for learning similar/dissimilar encodings."""

    def __init__(
            self,
            margin: float = 1.0,
            name: str = 'contrastive_loss'
    ):
        super().__init__(name=name)
        self.margin = margin

    def call(
            self,
            labels: tf.Tensor,
            encodings_1: tf.Tensor,
            encodings_2: tf.Tensor
    ) -> tf.Tensor:
        """Compute contrastive loss between pairs of encodings."""
        # Compute Euclidean distance
        distances = tf.sqrt(tf.reduce_sum(tf.square(encodings_1 - encodings_2), axis=-1))

        # Compute loss for similar/dissimilar pairs
        similar_loss = labels * tf.square(distances)
        dissimilar_loss = (1 - labels) * tf.square(tf.maximum(0., self.margin - distances))

        return tf.reduce_mean(similar_loss + dissimilar_loss)


class CombinedAutoencoderLoss(tf.keras.losses.Loss):
    """Combines multiple autoencoder-related losses."""

    def __init__(
            self,
            reconstruction_weight: float = 1.0,
            variational_weight: float = 0.1,
            contrastive_weight: float = 0.1,
            feature_weights: Optional[tf.Tensor] = None,
            margin: float = 1.0,
            beta: float = 1.0,
            name: str = 'combined_autoencoder_loss'
    ):
        super().__init__(name=name)
        self.reconstruction_loss = ReconstructionLoss(feature_weights)
        self.variational_loss = VariationalLoss(beta)
        self.contrastive_loss = ContrastiveLoss(margin)

        self.reconstruction_weight = reconstruction_weight
        self.variational_weight = variational_weight
        self.contrastive_weight = contrastive_weight
        self.feature_weights = feature_weights
        self.margin = margin
        self.beta = beta

    def call(
            self,
            y_true,
            y_pred
    ):
        """Standard call method required by tf.keras.losses.Loss"""
        # This is a simplified version - you'll need to adapt this
        # to match your actual loss calculation logic
        reconstruction_loss = self.reconstruction_weight * self.reconstruction_loss(
            y_true,
            y_pred
        )

        # Return the main loss component
        # Note: This simplified version only handles reconstruction loss
        # You'll need to adapt it based on your actual model outputs
        return reconstruction_loss

    def compute_full_loss(
            self,
            inputs: tf.Tensor,
            outputs: Dict[str, tf.Tensor],
            training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """Compute all losses."""
        losses = {}

        # Reconstruction loss
        losses['reconstruction'] = self.reconstruction_weight * self.reconstruction_loss(
            inputs,
            outputs['decoded']
        )

        # Variational loss if applicable
        if 'z_mean' in outputs and 'z_log_var' in outputs:
            losses['variational'] = self.variational_weight * self.variational_loss(
                outputs['z_mean'],
                outputs['z_log_var']
            )

        # Contrastive loss if pairs are provided
        if 'paired_encodings' in outputs and 'pair_labels' in outputs:
            losses['contrastive'] = self.contrastive_weight * self.contrastive_loss(
                outputs['pair_labels'],
                outputs['encoded'],
                outputs['paired_encodings']
            )

        # Total loss
        losses['total'] = tf.add_n(list(losses.values()))

        return losses

    def get_config(self):
        config = super().get_config()
        config.update({
            'reconstruction_weight': self.reconstruction_weight,
            'variational_weight': self.variational_weight,
            'contrastive_weight': self.contrastive_weight,
            'margin': self.margin,
            'beta': self.beta,
            # Note: feature_weights might not be serializable directly
            # You might need to handle it separately or convert to a list
        })
        return config

    @classmethod
    def from_config(cls, config):
        # You might need to handle feature_weights separately here
        return cls(**config)


class AutoencoderLossCallback(tf.keras.callbacks.Callback):
    """Callback to monitor autoencoder losses during training."""

    def __init__(self):
        super().__init__()
        self.history = {
            'reconstruction_loss': [],
            'variational_loss': [],
            'contrastive_loss': [],
            'total_loss': []
        }

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Record losses at the end of each epoch."""
        logs = logs or {}

        for loss_name in self.history.keys():
            if loss_name in logs:
                self.history[loss_name].append(logs[loss_name])

    def plot_losses(self, save_path: Optional[str] = None):
        """Plot loss curves."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for loss_name, values in self.history.items():
            if values:  # Only plot if we have values
                plt.plot(values, label=loss_name)

        plt.title('Autoencoder Losses Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == '__main__':

    # Register custom objects with Keras
    tf.keras.utils.get_custom_objects().update({
        'CombinedAutoencoderLoss': CombinedAutoencoderLoss,
        'ReconstructionLoss': ReconstructionLoss,
        'VariationalLoss': VariationalLoss,
        'ContrastiveLoss': ContrastiveLoss
    })