
from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    @abstractmethod
    def load_data(self):
        """
        Load and preprocess the dataset.

        Returns:
        - X_train, X_test: Training and testing features.
        - y_train, y_test: Training and testing labels.
        - preprocessor: Fitted preprocessor object.
        - input_dim: Dimension of the input features.
        - additional_info: Dictionary containing:
            additional_info = {
                'preprocessor': preprocessor,
                'numerical_cols': numerical_cols,
                'categorical_cols': categorical_cols,
                'feature_names': feature_names,   #(optional) This contains all column names after onehot encoding if done
                'n_classes': #(number of target classes),
                'problem_type':  #(binary, multiclass),
                 'label_encoders': #(optional) dictionary of fitted LabelEncoder objects
                                #for categorical columns if using LabelEncoder approach
            }
        """
        pass

    @abstractmethod
    def process_x(self, x, num_cols, cat_cols, **encoders):
        """
        Preprocess the input features.
        """
        pass
