import pandas as pd
import tensorflow as tf
from typing import Optional, Dict

class Encoder(tf.keras.layers.Layer):
    def __init__(
            self,
            latent_dim: int,
            intermediate_dims: list[int] = None,
            activation: str = 'relu',
            activation_out: str = 'sigmoid',
            name: str = "encoder",
            additional_info: Dict = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.latent_dim = latent_dim
        self.intermediate_dims = [85, 42] if intermediate_dims is None else intermediate_dims
        self.activation = activation
        self.additional_info = additional_info

        # Build encoder layers
        self.encoder_layers = []
        for dim in self.intermediate_dims:
            self.encoder_layers.append(
                tf.keras.layers.Dense(dim, activation=activation)
            )

        # Final encoding layer
        self.latent_layer = tf.keras.layers.Dense(
            latent_dim,
            activation=activation_out,
            name="feature_vector"
        )

        # Add batch normalization for better training
        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization()
            for _ in range(len(self.intermediate_dims) + 1)
        ]

    def call(self, inputs, training=None, mask=None): # Parameter name changed for clarity
        if isinstance(inputs, tuple):
            x = inputs[0] # Use only the first element if a tuple is provided.
        else:
            x = inputs

        for layer, batch_norm in zip(self.encoder_layers, self.batch_norm_layers[:-1]):
            x = layer(x)
            x = batch_norm(x, training=training)

        encoded = self.latent_layer(x)
        encoded = self.batch_norm_layers[-1](encoded, training=training)
        return encoded

    def encode_raw_data(self, data):
        """Encode raw data using the encoder."""
        if self.additional_info is None:
            raise ValueError("Additional info is required to encode raw data.")

        # Ensure data is a DataFrame
        data_df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=self.additional_info["original_column_names"])

        # Apply preprocessing steps
        if self.additional_info.keys().__contains__("label_encoders"):
            for col, encoder in self.additional_info["label_encoders"].items():
                data_df[col] = encoder.transform(data_df[col].astype(str))

        if self.additional_info.keys().__contains__("onehot_encoder"):
            cols = [col for col in data_df.columns if col not in self.additional_info["numerical_cols"]]
            res = self.additional_info["onehot_encoder"].transform(data_df[cols])
            data_df.drop(cols, axis=1, inplace=True)
            data_df = pd.concat([
                data_df,
                pd.DataFrame(res.toarray(),
                             columns=self.additional_info["onehot_encoder"].get_feature_names_out(self.additional_info["categorical_cols"])
                )
            ], axis=1)

        if self.additional_info.keys().__contains__("scaler"):
            data_df[self.additional_info["numerical_cols"]] = self.additional_info["scaler"].transform(
                data_df[self.additional_info["numerical_cols"]])

        # Ensure data is a NumPy array
        return data_df.to_numpy()




class Decoder(tf.keras.layers.Layer):
    def __init__(
            self,
            output_dim: int,
            intermediate_dims: list[int] = None,
            activation: str = 'relu',
            activation_out: str = 'sigmoid',
            name: str = "decoder",
            additional_info: Dict = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.output_dim = output_dim
        self.intermediate_dims = [42, 85] if intermediate_dims is None else intermediate_dims
        self.activation = activation

        # Build decoder layers
        self.decoder_layers = []
        for dim in self.intermediate_dims:
            self.decoder_layers.append(
                tf.keras.layers.Dense(dim, activation=activation)
            )

        # Final reconstruction layer
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation=activation_out,
            name="reconstruction"
        )

        # Add batch normalization for better training
        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization()
            for _ in range(len(self.intermediate_dims))
        ]

        self.additional_info = additional_info

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        x = inputs

        # Pass through intermediate layers
        for layer, batch_norm in zip(self.decoder_layers, self.batch_norm_layers):
            x = layer(x)
            x = batch_norm(x, training=training)

        # Final decoding
        decoded = self.output_layer(x)

        return decoded

    def retrieve_prototypes(self, prototype_data: tf.Tensor) -> pd.DataFrame:
        """Retrieve prototypes from the model and reverse preprocessing."""
        # Get decoded output from the decoder.
        decoded = self(inputs=prototype_data, training=False)

        if self.additional_info is None:
            return decoded

        # Ensure decoded is a NumPy array.
        decoded_np = decoded.numpy() if isinstance(decoded, tf.Tensor) else decoded

        # Build a DataFrame using column names from additional info,
        # assuming that the preprocessor was built using both numerical and categorical columns.
        all_cols = self.additional_info['original_column_names']
        if self.additional_info.keys().__contains__("processed_column_names"):
            all_cols = self.additional_info["processed_column_names"]

        if decoded_np.shape[1] != len(all_cols):
            raise ValueError("The number of columns in the decoded data does not match the number of columns in the preprocessed data.")

        prototypes_df = pd.DataFrame(decoded_np, columns=all_cols)

        if self.additional_info.keys().__contains__("label_encoders"):
            for col, encoder in self.additional_info["label_encoders"].items():
                prototypes_df[col] = encoder.inverse_transform(prototypes_df[col].astype(int))

        if self.additional_info.keys().__contains__("onehot_encoder"):
            cols = [col for col in prototypes_df.columns if col not in self.additional_info["numerical_cols"]]
            res = self.additional_info["onehot_encoder"].inverse_transform(prototypes_df[cols])
            prototypes_df.drop(cols, axis=1, inplace=True)
            prototypes_df = pd.concat([prototypes_df, pd.DataFrame(res, columns=self.additional_info["categorical_cols"])], axis=1)

        if self.additional_info.keys().__contains__("scaler"):
            prototypes_df[self.additional_info["numerical_cols"]] = self.additional_info["scaler"].inverse_transform(prototypes_df[self.additional_info["numerical_cols"]])

        return prototypes_df

class VariationalAutoencoder(tf.keras.Model):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 32,
            intermediate_dims: list[int] = None,
            activation: str = 'relu',
            activation_out: str = 'sigmoid',
            additional_info: Dict = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.intermediate_dims = [85, 42] if intermediate_dims is None else intermediate_dims
        self.activation = activation
        self.activation_out = activation_out

        # Define encoder and decoder
        self.encoder = Encoder(
            latent_dim=latent_dim,
            intermediate_dims=self.intermediate_dims,
            activation=self.activation,
            activation_out=self.activation_out,
            additional_info=additional_info
        )
        self.decoder = Decoder(
            output_dim=input_dim,
            intermediate_dims=[self.intermediate_dims[1], self.intermediate_dims[0]],
            activation=self.activation,
            activation_out = self.activation_out,
            additional_info=additional_info
        )

        # Add metrics for pretraining
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, mask: Optional[tf.Tensor] = None) -> Dict[
        str, tf.Tensor]:
        encoded = self.encoder(inputs = inputs, training=training)
        decoded = self.decoder(inputs = encoded, training=training)
        return {
            'encoded': encoded,
            'decoded': decoded
        }

    @tf.function
    def train_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(inputs = data, training=True)
            decoded = outputs['decoded']

            # Compute loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(data, decoded)
            )

            # Add KL divergence regularization if needed
            kl_loss = 0.0  # Add KL divergence computation if using variational AE

            total_loss = reconstruction_loss + kl_loss

        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def save_pretrained(self, filepath: str):
        """Save pretrained encoder and decoder weights"""
        self.save_weights(filepath)

    def load_pretrained(self, filepath: str):
        """Load pretrained encoder and decoder weights"""
        self.load_weights(filepath)