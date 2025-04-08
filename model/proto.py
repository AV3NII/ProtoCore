import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Optional, Dict

from sklearn.utils.multiclass import type_of_target

from .config import ModelConfig
from .components import VariationalAutoencoder
from .losses import CombinedAutoencoderLoss, DiversityLoss, R1Loss, R2Loss
from loaders import BaseDataLoader


class ProtoModel(tf.keras.Model):
    def __init__(
            self,
            mc: ModelConfig,
            data: tf.Tensor,
            pretrain_ae: bool = True,
            prototype_strategy: str = "kmeans",
            custom_prototypes: Optional[pd.DataFrame] = None,
            additional_info: Optional[Dict] = None,
    ):
        """
        :param mc: ModelConfig - configuration for model
        :param data: tf.Tensor - data to initialise the prototypes
        :param custom_prototypes: tf.Tensor (optional) custom prototypes from raw data !!pass the additional info
        :param prototype_strategy: "random" || "kmeans" || "manual"
        :param additional_info: a
        :return:
        """
        super().__init__()
        self.model_config = mc
        self.vac = VariationalAutoencoder(
            mc.input_dim,
            mc.latent_dim,
            mc.intermediate_dims,
            mc.activation,
            mc.activation_out,
            additional_info
        )
        self.data = data

        # Conditionally pretrain autoencoder
        if pretrain_ae:
            self.pretrain_autoencoder(data)

        match prototype_strategy:
            case "manual":
                if additional_info:
                    self.proto_vectors = self.initialize_prototypes_manual(
                        custom_prototypes=custom_prototypes
                    )
                else:
                    raise ValueError("Must provide custom_prototypes for manual strategy")
            case "kmeans":
                self.proto_vectors = self.initialize_prototypes(
                    data=data,
                    strategy='kmeans',
                    n_prototypes=mc.n_prototypes
                )
            case "random":
                self.proto_vectors = self.initialize_prototypes(
                    data=data,
                    strategy='random',
                    n_prototypes=mc.n_prototypes
                )
            case _:
                raise ValueError("Invalid prototype init strategy")

        initializer = tf.keras.initializers.GlorotUniform(seed=mc.random_seed)
        self.last_layer = tf.Variable(
            initializer(shape=[mc.n_prototypes, mc.num_classes]),  # TODO: replace later with full protoclasifier
            name="classifier_head",
            trainable=True
        )

        # -- Losses --
        self.class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.r1_loss = R1Loss(mc.lambda_1)
        self.r2_loss = R2Loss(mc.lambda_2)
        self.diversity_loss = DiversityLoss(mc.lambda_diversity)


        # -- metrics --
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.diversity_metric = tf.keras.metrics.Mean(name='diversity')

    ### Autoencoder Pretrain
    def pretrain_autoencoder(self, data: tf.Tensor):
        """Integrated pretraining phase"""
        # Temporary compile for pretraining
        self.vac.compile(
            optimizer=tf.keras.optimizers.Adam(self._calc_learning_rate(data)),
            loss=CombinedAutoencoderLoss(
                reconstruction_weight=1.0,
                variational_weight=0.1
            )
        )

        # Pretrain only the autoencoder
        print("Starting autoencoder pretraining...")
        self.vac.fit(
            data,
            data,  # Autoencoder reconstruction target
            epochs=self._calc_pretrain_epochs(data),
            batch_size=self._calc_batch_size(data),
            verbose=False
        )

        print(f"Pretraining completed. Final loss: {self.vac.evaluate(data)}")
        self.pretrained = True

    #### Encoder specific
    def _calc_pretrain_epochs(self, data: tf.Tensor) -> int:
        """Calculate pretraining duration based on data complexity"""
        return max(10, int(50 * (data.shape[1] / 100)))

    def _calc_batch_size(self, data: tf.Tensor) -> int:
        """Calculate reasonable batch size based on data size"""
        return min(256, max(32, int(data.shape[0] * 0.01)))

    def _calc_learning_rate(self, data: tf.Tensor) -> float:
        """Heuristic for learning rate based on input dimension"""
        return 3e-4 / np.log(data.shape[1])

    #### Prototype init from data
    def initialize_prototypes(self, data: tf.Tensor, strategy: str, n_prototypes: int) -> tf.Variable:
        encoded_data = self.vac.encoder(data, False)

        if strategy == 'kmeans':
            from sklearn.cluster import KMeans

            # Convert to numpy for sklearn
            encoded_np = encoded_data.numpy()

            # Perform k-means clustering
            kmeans = KMeans(
                n_clusters=n_prototypes,
                random_state=self.model_config.random_seed
            ).fit(encoded_np)

            # Use cluster centers as prototypes
            prototype_vectors = tf.convert_to_tensor(
                kmeans.cluster_centers_,
                dtype=tf.float32
            )

            return tf.Variable(
                prototype_vectors,
                trainable=True,
                name="kmeans_protptypes",
                dtype=tf.float32
            )

        elif strategy == 'random':
            # Compute data statistics
            mean = tf.reduce_mean(encoded_data, axis=0)
            std = tf.math.reduce_std(encoded_data, axis=0)

            # Initialize randomly within data distribution
            prototype_vectors = tf.random.normal(
                shape=[n_prototypes, self.vac.encoder.latent_dim],
                mean=mean,
                stddev=std,
                seed=self.model_config.random_seed
            )

            return tf.Variable(
                prototype_vectors,
                trainable = True,
                name = "random_prototypes",
                dtype = tf.float32
            )

    #### Prototype init from preselection
    def initialize_prototypes_manual(
            self,
            custom_prototypes = None
    ) -> tf.Variable:
        """Handle manual prototype initialization with preprocessing"""
        preprocessed_data = self.vac.encoder.encode_raw_data(custom_prototypes)

        # Convert to tensor if it's not already
        tensor_data = tf.convert_to_tensor(preprocessed_data, dtype=tf.float32)

        # Now encode the preprocessed data
        encoded_prototypes = self.vac.encoder(tensor_data)

        # Step 3: Verify shape matches expectations
        expected_shape = (self.model_config.n_prototypes, self.model_config.latent_dim)
        if encoded_prototypes.shape != expected_shape:
            raise ValueError(f"Encoded prototypes must have shape {expected_shape}. "
                             f"Got {encoded_prototypes.shape}")

        return tf.Variable(
            encoded_prototypes,
            trainable=True,
            name="manual_prototypes",
            dtype=tf.float32
        )

    def list_of_norms(self, X):
        '''
        X is a list of vectors X = [x_1, ..., x_n], we return
            [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
        function is the squared euclidean distance.
        '''
        return tf.reduce_sum(tf.pow(X, 2), axis=1)

    def list_of_distances(self, X, Y):
        '''
        Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
        Y = [y_1, ... , y_m], we return a list of vectors
                [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
                 ...
                 [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
        where the distance metric used is the sqared euclidean distance.
        The computation is achieved through a clever use of broadcasting.
        '''
        XX = tf.reshape(self.list_of_norms(X), shape=(-1, 1))
        YY = tf.reshape(self.list_of_norms(Y), shape=(1, -1))
        output = XX + YY - 2 * tf.matmul(X, tf.transpose(Y))

        return output

    def compute_prototype_distances(self):
        return self.list_of_distances(self.proto_vectors, self.proto_vectors)

    def call(self, inputs: tf.Tensor, training = False):
        encoded = self.vac.encoder(inputs, training)
        decoded = self.vac.decoder(encoded, training)
        proto_dist = self.list_of_distances(encoded, self.proto_vectors)
        logits = tf.matmul(proto_dist, self.last_layer)
        return {
            'encoded': encoded,
            'decoded': decoded,
            'logits': logits,
            'prototype_distances': proto_dist
        }

    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            encoded = outputs['encoded']
            decoded = outputs['decoded']
            logits = outputs['logits']
            prototype_distances = outputs['prototype_distances']
            proto_vectors = self.proto_vectors

            # Losses
            class_error = tf.reduce_mean(self.class_loss(labels, logits))

            # Reconstruction (VAE) loss
            vac_error = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(inputs, decoded)
            )

            # Prototype losses
            error_1 = self.r1_loss(proto_vectors, encoded)
            error_2 = self.r2_loss(proto_vectors, encoded)

            # Diversity loss
            proto_distances_div = self.compute_prototype_distances()
            diversity_loss = self.diversity_loss(None, proto_distances_div)

            # Combined total loss (using lambda weights directly)
            total_loss = self.model_config.lambda_class * class_error + \
                         self.model_config.lambda_ae * vac_error + \
                         error_1 + error_2 + self.model_config.lambda_diversity * diversity_loss

            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Calculate accuracy
            self.accuracy_metric.update_state(labels, logits)
            self.diversity_metric.update_state(diversity_loss)

            return {
                'loss': total_loss,
                'class_error': class_error,
                'vac_error': vac_error,
                'error_1': error_1,
                'error_2': error_2,
                'diversity_loss': self.diversity_metric.result(),
                'accuracy': self.accuracy_metric.result()
            }

    def test_step(self, data):
        inputs, labels = data
        outputs = self(inputs, training=False)
        encoded = outputs['encoded']
        decoded = outputs['decoded']
        logits = outputs['logits']
        proto_vectors = self.proto_vectors

        # -- Losses --
        class_error = tf.reduce_mean(self.class_loss(labels, logits))

        vac_error = tf.reduce_mean(
            tf.keras.losses.mean_squared_error(inputs, decoded)
        )

        error_1 = self.r1_loss(proto_vectors, encoded)

        error_2 = self.r2_loss(proto_vectors, encoded)

        proto_distances = self.compute_prototype_distances()
        diversity_loss = self.diversity_loss(None, proto_distances)

        total_loss = self.model_config.lambda_class * class_error + \
                     self.model_config.lambda_ae * vac_error + \
                     error_1 + error_2 - diversity_loss # Note the negative sign due to negative values from diversity loss

        # -- Metrics --
        self.accuracy_metric.update_state(labels, logits)
        self.diversity_metric.update_state(diversity_loss)

        return {
            'loss': total_loss,
            'class_error': class_error,
            'vac_error': vac_error,
            'error_1': error_1,
            'error_2': error_2,
            'diversity_loss': self.diversity_metric.result(),
            'accuracy':  self.accuracy_metric.result()
        }


    def retrieve_prototypes(self):
        """Retrieve prototypes in readable format, returns a dataframe"""
        protos = self.vac.decoder.retrieve_prototypes(self.proto_vectors)
        input = self.vac.encoder.encode_raw_data(protos)
        protos["Class"] = tf.argmax(self(input, training=False)['logits'], 1).numpy()
        return protos









