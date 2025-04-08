import tensorflow as tf
from typing import Optional

from model.config.model_config import ModelConfig


class R1Loss(tf.keras.losses.Loss):
    """R1 Loss: Each prototype should be close to at least one training example."""

    def __init__(self, lambda_1: float = 0.25, name: str = 'r1_loss'):
        super().__init__(name=name)
        self.lambda_1 = lambda_1

    def call(self, prototypes: tf.Tensor, encoded_inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            prototypes: Prototype vectors [n_prototypes, latent_dim]
            encoded_inputs: Encoded training examples [batch_size, latent_dim]
        """
        # Compute distances between each prototype and all encoded inputs
        distances = tf.reduce_sum(
            tf.square(
                tf.expand_dims(prototypes, axis=1) -
                tf.expand_dims(encoded_inputs, axis=0)
            ),
            axis=-1
        )  # Shape: [n_prototypes, batch_size]

        # For each prototype, get minimum distance to any training example
        min_distances = tf.reduce_min(distances, axis=1)

        # Average over all prototypes
        return self.lambda_1 * tf.reduce_mean(min_distances)


class R2Loss(tf.keras.losses.Loss):
    """R2 Loss: Each training example should be close to at least one prototype."""

    def __init__(self, lambda_2: float = 0.25, name: str = 'r2_loss'):
        super().__init__(name=name)
        self.lambda_2 = lambda_2

    def call(self, prototypes: tf.Tensor, encoded_inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            prototypes: Prototype vectors [n_prototypes, latent_dim]
            encoded_inputs: Encoded training examples [batch_size, latent_dim]
        """
        # Compute distances between each encoded input and all prototypes
        distances = tf.reduce_sum(
            tf.square(
                tf.expand_dims(encoded_inputs, axis=1) -
                tf.expand_dims(prototypes, axis=0)
            ),
            axis=-1
        )  # Shape: [batch_size, n_prototypes]

        # For each training example, get minimum distance to any prototype
        min_distances = tf.reduce_min(distances, axis=1)

        # Average over all training examples in batch
        return self.lambda_2 * tf.reduce_mean(min_distances)


class DiversityLoss(tf.keras.losses.Loss):
    """Loss function to encourage diversity among prototypes."""

    def __init__(self, lambda_diversity: float = 2.5, name: str = 'diversity_loss'):
        super().__init__(name=name)
        self.lambda_diversity = lambda_diversity

    def call(self, y_true, prototype_distances: tf.Tensor) -> tf.Tensor:
        """
        Compute diversity loss from prototype distances.

        Args:
            y_true: Placeholder (unused)
            y_pred: Prototype distances tensor of shape [batch_size, n_prototypes]

        Returns:
            Scalar tensor representing the diversity loss
        """
        # Mask the diagonal to avoid considering distance of prototype to itself
        mask = tf.eye(tf.shape(prototype_distances)[0], dtype=tf.bool)
        masked_distances = tf.where(mask, float('inf'), prototype_distances)

        # Find the minimum distance between any pair of *different* prototypes
        min_pairwise_distance = tf.reduce_min(masked_distances)

        # Diversity loss: We want to *maximize* the minimum distance.
        # To minimize loss during training, we minimize the *negative* of the minimum distance.
        # We also divide by a scaling factor (e.g., average of distances or just a constant)
        # to control the magnitude of the loss and potentially stabilize training.
        # Using average distance as scaling factor for normalization.
        avg_distance = tf.reduce_mean(prototype_distances) + 1e-8  # Avoid division by zero
        diversity_loss = - min_pairwise_distance / avg_distance  # Normalize and negate

        return self.lambda_diversity * diversity_loss

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_diversity": self.lambda_diversity})
        return config


