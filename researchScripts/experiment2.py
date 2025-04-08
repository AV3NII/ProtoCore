#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random as python_random
import yaml
import json
import shap

from sklearn.metrics import classification_report, confusion_matrix

from loaders import AdultIncomeDataLoader
from model import TrainingConfig, ModelConfig  # Import Configs
from model.proto import ProtoModel

os.chdir("..")

# --- Experiment 1: Lambda Diversity Variation ---
EXPERIMENT_NAME = "experiment2"
EXPERIMENT_BASE_DIR = f'./experiments/{EXPERIMENT_NAME}'

# Lambda diversity values to experiment with
LAMBDA_DIVERSITY_VALUES = [0, 1, 2]

# SHAP Parameters (kept as CAPITAL_VARS for script-level config)
SHAP_BACKGROUND_SIZE = 100
SHAP_SAMPLE_SIZE = 10
RANDOM_SEED = 808

# --- Standardized Hyperparameters (moved to Configs) ---
# Define configurations using dataclasses
base_model_config = ModelConfig(n_prototypes=20, num_classes=2, latent_dim=32, intermediate_dims=[85, 42],
                               activation='relu', activation_out='sigmoid', lambda_class=10.0, lambda_ae=1.0,
                               lambda_1=1.0, lambda_2=1.0, random_seed=RANDOM_SEED)
base_training_config = TrainingConfig(training_epochs=30, learning_rate=0.0005, batch_size=32,
                                     validation_split=0.2, early_stopping=True, patience=5, shuffle=True)

prototype_examples = [
    # --- Age 20 ---
    { # 1. Young White Male, Some College, Service Job
        "age": 20,
        "workclass": "Private",
        "fnlwgt": 215646,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White", "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1579,
        "hours-per-week": 35,
        "native-country": "United-States"
    },
    { # 2. Young White Female, Some College, Clerical Job
        "age": 20,
        "workclass": "Private",
        "fnlwgt": 189229,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White", "sex": "Female",
        "capital-gain": 861,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    },
    { # 3. Young Black Male, High School, Service Job
        "age": 20,
        "workclass": "Private",
        "fnlwgt": 283461,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black", "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1408,
        "hours-per-week": 40,
        "native-country": "United-States"
    },
    { # 4. Young Asian Female, Some College, Sales
        "age": 20,
        "workclass": "Private",
        "fnlwgt": 153493,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "Asian-Pac-Islander", "sex": "Female",
        "capital-gain": 1055,
        "capital-loss": 0,
        "hours-per-week": 25,
        "native-country": "United-States"
    },
     { # 5. Young White Male, High School, Craft/Repair
        "age": 20,
        "workclass": "Private",
        "fnlwgt": 192140,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Craft-repair",
        "relationship": "Not-in-family",
        "race": "White", "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1602,
        "hours-per-week": 40,
        "native-country": "United-States"
    },

    # --- Age 35 ---
    { # 6. Mid-Age White Male, Bachelors, Professional
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 176756,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White", "sex": "Male",
        "capital-gain": 5013,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    },
    { # 7. Mid-Age White Female, Masters, Managerial
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 229249,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White", "sex": "Female",
        "capital-gain": 7298,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    },
    { # 8. Mid-Age White Male, High School, Craft/Repair (Divorced)
        "age": 35,
        "workclass": "Self-emp-not-inc",
        "fnlwgt": 123011,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Craft-repair",
        "relationship": "Not-in-family",
        "race": "White", "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1887,
        "hours-per-week": 50,
        "native-country": "United-States"
    },
     { # 9. Mid-Age Black Male, Some College, Transport
        "age": 35,
         "workclass": "Private",
         "fnlwgt": 346004,
         "education": "Some-college",
         "education-num": 10,
        "marital-status": "Married-civ-spouse",
         "occupation": "Transport-moving",
         "relationship": "Husband",
         "race": "Black", "sex": "Male",
        "capital-gain": 3103,
         "capital-loss": 0,
         "hours-per-week": 40,
         "native-country": "United-States"
    },
    { # 10. Mid-Age Black Female, Bachelors, Government Clerical
        "age": 35,
        "workclass": "Local-gov",
        "fnlwgt": 199353,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Unmarried",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 1741,
        "hours-per-week": 40,
        "native-country": "United-States"
    },

    # --- Age 50 ---
    { # 11. Older White Male, Masters, Managerial (High Income Potential)
        "age": 50,
        "workclass": "Private",
        "fnlwgt": 158790,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White", "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States"
    },
    { # 12. Older White Female, High School, Sales (Divorced)
        "age": 50,
        "workclass": "Private",
        "fnlwgt": 88137,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 1740,
        "hours-per-week": 38,
        "native-country": "United-States"
    },
    { # 13. Older Black Male, High School, Machine Operator
        "age": 50,
        "workclass": "Private",
        "fnlwgt": 209646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Machine-op-inspct",
        "relationship": "Husband",
        "race": "Black", "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    },
    { # 14. Older Asian Female, Prof School, Professional (Self-Employed)
        "age": 50,
         "workclass": "Self-emp-inc",
         "fnlwgt": 119078,
         "education": "Prof-school",
         "education-num": 15,
        "marital-status": "Married-civ-spouse",
         "occupation": "Prof-specialty",
         "relationship": "Wife",
         "race": "Asian-Pac-Islander", "sex": "Female",
        "capital-gain": 7688,
         "capital-loss": 0,
         "hours-per-week": 60,
         "native-country": "United-States"
    },
     { # 15. Older White Male, Some College, Government Job
        "age": 50,
        "workclass": "State-gov",
        "fnlwgt": 173940,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Protective-serv",
        "relationship": "Husband",
        "race": "White", "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1672,
        "hours-per-week": 40,
        "native-country": "United-States"
    },

    # --- Age 65 ---
    { # 16. Senior White Male, High School, Retired/Part-time Craft
        "age": 65,
        "workclass": "Self-emp-not-inc",
        "fnlwgt": 99357,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "White", "sex": "Male",
        "capital-gain": 4386,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    },
    { # 17. Senior White Female, Some College, Widowed, Not Working
        "age": 65,
        "workclass": "Private",
        "fnlwgt": 154806,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Widowed",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 1590,
        "hours-per-week": 8,
        "native-country": "United-States"
    },
    { # 18. Senior Black Male, Bachelors, Retired/Part-time Professional
        "age": 65,
        "workclass": "Private",
        "fnlwgt": 184681,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Black", "sex": "Male",
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 25,
        "native-country": "United-States"
    },
    { # 19. Senior Black Female, High School, Widowed, Service
        "age": 65,
        "workclass": "Private",
        "fnlwgt": 135608,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Widowed",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black", "sex": "Female",
        "capital-gain": 1173,
        "capital-loss": 0,
        "hours-per-week": 30,
        "native-country": "United-States"
    },
    { # 20. Senior Amer-Indian-Eskimo Male, High School, Farming/Fishing
        "age": 65,
        "workclass": "Private",
        "fnlwgt": 198124,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Farming-fishing",
        "relationship": "Husband",
        "race": "Amer-Indian-Eskimo",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1485,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
]

# Convert the list of dictionaries to a NumPy array
prototype_df = pd.DataFrame(prototype_examples)

def run_lambda_diversity_experiment(lambda_diversity_value, base_model_config, base_training_config):
    """
    Runs a single experiment varying lambda_diversity, using configurations.
    """

    experiment_dir = f'{EXPERIMENT_BASE_DIR}/lambda_diversity_{lambda_diversity_value}'
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f'{experiment_dir}/model', exist_ok=True)
    os.makedirs(f'{experiment_dir}/plots', exist_ok=True)
    os.makedirs(f'{experiment_dir}/shap_analysis', exist_ok=True)

    prototype_df.to_csv(f'{experiment_dir}/model/initial-prototypes.csv', index=False)
    print(f"Initial prototypes saved to: {experiment_dir}/model")

    # --- Save Experiment Configuration ---
    experiment_config_dict = {
        "experiment_name": EXPERIMENT_NAME,
        "lambda_diversity": lambda_diversity_value,
        "random_seed": RANDOM_SEED,
        "shap_background_size": SHAP_BACKGROUND_SIZE,
        "shap_sample_size": SHAP_SAMPLE_SIZE,
        "model_config": base_model_config.__dict__,  # Save ModelConfig dict
        "training_config": base_training_config.__dict__  # Save TrainingConfig dict
    }
    with open(f'{experiment_dir}/experiment_config.yaml', 'w') as f:
        yaml.dump(experiment_config_dict, f, default_flow_style=False)
    with open(f'{experiment_dir}/experiment_config.json', 'w') as f:
        json.dump(experiment_config_dict, f, indent=4)

    # --- Set Random Seeds for Reproducibility ---
    np.random.seed(RANDOM_SEED)
    python_random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # --- Load Data ---
    loader = AdultIncomeDataLoader()
    X_train, X_test, y_train, y_test, input_dim, additional_info = loader.load_data()
    feature_names = additional_info['processed_column_names'] # FIX: Define feature_names here


    # --- Configure Model and Training ---
    # Create specific configs for this experiment by copying base configs
    mc = ModelConfig(**base_model_config.__dict__)  # Create new ModelConfig instance
    tc = TrainingConfig(**base_training_config.__dict__)  # Create new TrainingConfig instance

    mc.input_dim = input_dim  # Set input_dim from data
    mc.lambda_diversity = lambda_diversity_value  # Override lambda_diversity

    def build_model(mc: ModelConfig, tc: TrainingConfig, prototypes=prototype_df):
        model = ProtoModel(mc=mc, data=X_train, prototype_strategy="manual", additional_info=additional_info, custom_prototypes=prototypes)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=tc.learning_rate)
        model.compile(optimizer=optimizer, metrics=['accuracy'])
        return model

    model = build_model(mc, tc)

    # --- Define Early Stopping ---
    early_stoppers = [
        tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            patience=tc.patience,  # Use patience from TrainingConfig
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=tc.patience,  # Use patience from TrainingConfig
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"--- Training model with lambda_diversity = {lambda_diversity_value} ---")
    history = model.fit(
        X_train, y_train,
        epochs=tc.training_epochs,  # Use training_epochs from TrainingConfig
        validation_split=tc.validation_split,  # Use validation_split from TrainingConfig
        batch_size=tc.batch_size,  # Use batch_size from TrainingConfig
        validation_data=(X_test, y_test),
        callbacks=early_stoppers,
        shuffle=tc.shuffle
    )

    # --- Save Model and Prototypes ---
    model.save(f'{experiment_dir}/model/best_model')
    print(f"Model saved to {experiment_dir}/model/best_model")

    prototypes = model.retrieve_prototypes()
    prototypes.to_csv(f'{experiment_dir}/model/prototypes.csv', index=False)
    print(f"Prototypes saved to {experiment_dir}/model/prototypes.csv")

    # --- Evaluate Model on Test Set ---
    print("--- Evaluating on Test Set ---")
    y_pred_test = model.predict(X_test)
    y_pred_labels_test = np.argmax(y_pred_test['logits'], axis=1)
    report = classification_report(y_test, y_pred_labels_test, digits=4)
    conf_matrix = confusion_matrix(y_test, y_pred_labels_test)

    with open(f'{experiment_dir}/test_evaluation.txt', 'w') as f:
        f.write("Classification Report (Test Set):\n")
        f.write(report)
        f.write("\nConfusion Matrix (Test Set):\n")
        f.write(np.array_str(conf_matrix))

    print("Test Set Evaluation Metrics saved to:")
    print(f"{experiment_dir}/test_evaluation.txt")

    # --- Save Experiment Summary ---
    experiment_summary = experiment_config_dict.copy()
    experiment_summary.update({
        "validation_accuracy": history.history['val_accuracy'][-1] if history.history['val_accuracy'] else "N/A",
        "test_classification_report": report,
        "test_confusion_matrix": conf_matrix.tolist()
    })

    with open(f'{experiment_dir}/experiment_summary.json', 'w') as f:
        json.dump(experiment_summary, f, indent=4)
    print(f"Experiment summary saved to: {experiment_dir}/experiment_summary.json")

    # --- Plot Training History ---
    plot_training_history(history, experiment_dir)

    # --- SHAP Analysis ---
    perform_shap_analysis(model, X_train, X_test, additional_info, experiment_dir, feature_names) # FIX: Pass feature_names

    print(f"--- Experiment with lambda_diversity = {lambda_diversity_value} complete. ---")
    print(f"All results saved to: {experiment_dir}/")


def plot_training_history(history, experiment_dir):
    """Plots training history metrics."""
    plt.figure(figsize=(15, 10))
    metrics = ['accuracy', 'loss', 'class_error', 'vac_error', 'error_1', 'error_2', 'diversity_loss']
    for i, metric in enumerate(metrics, 1):
        if metric in history.history:
            plt.subplot(3, 3, i)
            plt.plot(history.history[metric], label=f'Training {metric}')
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                plt.plot(history.history[val_metric], label=f'Validation {metric}')
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.xlabel('Epoch')
            plt.legend()

    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/plots/training_history.png')
    plt.close()
    print(f"Training history plot saved to: {experiment_dir}/plots/training_history.png")


def perform_shap_analysis(model, X_train, X_test, additional_info, experiment_dir, feature_names): # FIX: Accept feature_names
    """Performs SHAP analysis and saves plots."""
    print("--- Performing SHAP analysis ---")

    X_train_np = X_train.numpy() if isinstance(X_train, tf.Tensor) else X_train
    X_test_np = X_test.numpy() if isinstance(X_test, tf.Tensor) else X_test

    original_cat_cols = additional_info['categorical_cols']
    numerical_cols = additional_info['numerical_cols']

    def model_predict(X):
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32) if isinstance(X, np.ndarray) else X
        predictions = model.predict(X_tensor)
        return tf.nn.softmax(predictions['logits'], axis=1).numpy()[:, 1] if isinstance(predictions,
                                                                                           dict) and 'logits' in predictions else predictions

    indices = np.random.choice(X_train_np.shape[0], SHAP_BACKGROUND_SIZE, replace=False)
    background_data = X_train_np[indices]
    explainer = shap.KernelExplainer(model_predict, background_data)
    test_indices = np.random.choice(X_test_np.shape[0], SHAP_SAMPLE_SIZE, replace=False)
    shap_sample = X_test_np[test_indices]
    shap_values = explainer.shap_values(shap_sample)

    np.save(f'{experiment_dir}/shap_analysis/shap_values.npy', shap_values)
    shap_sample_df = pd.DataFrame(shap_sample, columns=feature_names)
    shap_sample_df.to_csv(f'{experiment_dir}/shap_analysis/shap_sample_data.csv', index=False)

    feature_mapping, feature_groups = create_feature_mapping_and_groups(feature_names, original_cat_cols,
                                                                       numerical_cols)

    plot_shap_summary(shap_values, shap_sample, feature_names, feature_mapping, experiment_dir)
    plot_shap_waterfalls(shap_values, shap_sample, explainer, feature_names, feature_mapping, experiment_dir)
    plot_shap_feature_importance(shap_values, feature_groups, feature_mapping, shap_sample, numerical_cols,
                                 experiment_dir, feature_names) # FIX: Pass feature_names


    print(f"SHAP analysis complete. Results saved to: {experiment_dir}/shap_analysis/")


def create_feature_mapping_and_groups(feature_names, original_cat_cols, numerical_cols):
    """Creates feature mapping and groups for SHAP plots."""
    feature_mapping = {}
    feature_groups = {}
    for i, col in enumerate(feature_names):
        if col in numerical_cols:
            feature_mapping[i] = col
            feature_groups[col] = [i]
    for cat_col in original_cat_cols:
        cat_indices = []
        for i, col in enumerate(feature_names):
            if col.startswith(f"{cat_col}_"):
                category = col.split('_', 1)[1]
                feature_mapping[i] = f"{cat_col}={category}"
                cat_indices.append(i)
        feature_groups[cat_col] = cat_indices
    return feature_mapping, feature_groups


def plot_shap_summary(shap_values, shap_sample, feature_names, feature_mapping, experiment_dir):
    """Generates and saves SHAP summary plot."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        shap_sample,
        feature_names=[feature_mapping.get(i, f"Feature {i}") for i in range(len(feature_names))],
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/shap_analysis/shap_summary.png')
    plt.close()
    print(f"SHAP summary plot saved to: {experiment_dir}/shap_analysis/shap_summary.png")


def plot_shap_waterfalls(shap_values, shap_sample, explainer, feature_names, feature_mapping, experiment_dir):
    """Generates and saves SHAP waterfall plots for a few examples."""
    for i in range(min(5, len(shap_sample))):
        plt.figure(figsize=(12, 8))
        try:
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[i],
                    base_values=explainer.expected_value,
                    data=shap_sample[i],
                    feature_names=[feature_mapping.get(j, f"Feature {j}") for j in range(len(feature_names))],
                ),
                max_display=10,
                show=False
            )
        except Exception as e:
            print(f"Failed waterfall plot for example {i}: {e}")
            continue
        plt.tight_layout()
        plt.savefig(f'{experiment_dir}/shap_analysis/shap_waterfall_plot_{i}.png')
        plt.close()
    print(f"SHAP waterfall plots saved to: {experiment_dir}/shap_analysis/")


def plot_shap_feature_importance(shap_values, feature_groups, feature_mapping, shap_sample, numerical_cols,
                                 experiment_dir, feature_names): # FIX: Accept feature_names
    """Generates and saves SHAP feature importance plots."""
    feature_importance = np.abs(shap_values).mean(0)
    group_importance = {}
    for group, indices in feature_groups.items():
        group_importance[group] = sum(feature_importance[i] for i in indices)

    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    top_groups = [g[0] for g in sorted_groups[:5]]

    for group in top_groups:
        if group in numerical_cols:
            plot_shap_dependence(group, feature_groups, shap_values, shap_sample, feature_names,
                                 feature_mapping, experiment_dir)
        else:
            plot_shap_categorical_importance(group, feature_groups, feature_importance, feature_mapping,
                                             experiment_dir)

    plot_comprehensive_feature_importance(sorted_groups, experiment_dir)
    print(f"SHAP feature importance plots saved to: {experiment_dir}/shap_analysis/")


def plot_shap_dependence(group, feature_groups, shap_values, shap_sample, feature_names, feature_mapping,
                         experiment_dir):
    """Generates and saves SHAP dependence plot for numerical features."""
    idx = feature_groups[group][0]
    plt.figure(figsize=(10, 7))
    try:
        shap.dependence_plot(
            idx,
            shap_values,
            shap_sample,
            feature_names=[feature_mapping.get(j, f"Feature {j}") for j in range(len(feature_names))],
            show=False
        )
    except Exception as e:
        print(f"Failed dependence plot for {group}: {e}")
        return

    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/shap_analysis/shap_dependence_{group}.png')
    plt.close()
    print(f"SHAP dependence plot saved to: {experiment_dir}/shap_analysis/shap_dependence_{group}.png")


def plot_shap_categorical_importance(group, feature_groups, feature_importance, feature_mapping, experiment_dir):
    """Generates and saves SHAP categorical feature importance plot."""
    plt.figure(figsize=(10, 7))
    group_values = []
    group_labels = []
    for idx in feature_groups[group]:
        group_values.append(feature_importance[idx])
        group_labels.append(
            feature_mapping[idx].split('=')[1] if '=' in feature_mapping[idx] else feature_mapping[idx])

    sorted_indices = np.argsort(group_values)[::-1]
    sorted_values = [group_values[i] for i in sorted_indices]
    sorted_labels = [group_labels[i] for i in sorted_indices]

    plt.barh(range(len(sorted_labels)), sorted_values)
    plt.yticks(range(len(sorted_labels)), sorted_labels)
    plt.xlabel('SHAP Importance')
    plt.title(f'Importance of values for {group}')
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/shap_analysis/shap_categorical_{group}.png')
    plt.close()
    print(f"SHAP categorical importance plot saved to: {experiment_dir}/shap_analysis/shap_categorical_{group}.png")


def plot_comprehensive_feature_importance(sorted_groups, experiment_dir):
    """Generates and saves comprehensive SHAP feature importance plot."""
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_groups)), [g[1] for g in sorted_groups])
    plt.yticks(range(len(sorted_groups)), [g[0] for g in sorted_groups])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance (Original Features)')
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/shap_analysis/feature_importance_by_group.png')
    plt.close()
    print(f"SHAP comprehensive feature importance plot saved to: {experiment_dir}/shap_analysis/feature_importance_by_group.png")


if __name__ == "__main__":
    print(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")
    for lambda_diversity in LAMBDA_DIVERSITY_VALUES:
        run_lambda_diversity_experiment(lambda_diversity, base_model_config,
                                         base_training_config)  # Pass configs
    print(f"--- Experiment {EXPERIMENT_NAME} complete. ---")
    print(f"All results saved under: {EXPERIMENT_BASE_DIR}/")
