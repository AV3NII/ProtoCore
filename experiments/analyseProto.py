import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
import os
import json

EXPERIMENT_BASE_DIR_PREFIX = 'experiment'  # Prefix for experiment base directories
MODEL_DIR_NAME = 'model'
PROTOTYPES_CSV_FILENAME = 'prototypes.csv'
OUTPUT_PLOT_FILENAME = 'prototype_analysis_grid_layout.png'


def analyze_prototypes(prototypes_csv_path, output_dir):
    """
    Analyzes prototype data from a CSV, performs PCA, and generates a grid plot.

    Args:
        prototypes_csv_path (str): Path to the prototypes.csv file.
        output_dir (str): Directory to save the output plot.
    """
    print(f"Analyzing prototypes from: {prototypes_csv_path}")

    try:  # Add try-except block for file reading and processing
        # 1. Load and prepare the data
        df = pd.read_csv(prototypes_csv_path)

        # Select numerical and categorical columns
        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                            'capital-loss', 'hours-per-week']
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                              'relationship', 'race', 'sex', 'native-country']

        # Create label encoders for categorical variables
        encoders = {}
        for cat_feature in categorical_features:
            encoders[cat_feature] = LabelEncoder()
            df[cat_feature + '_encoded'] = encoders[cat_feature].fit_transform(df[cat_feature])

        # Combine numerical and encoded categorical features
        encoded_categorical_features = [f + '_encoded' for f in categorical_features]
        all_features = numerical_features + encoded_categorical_features
        X = df[all_features]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Apply PCA
        pca = PCA()
        X_pca_full = pca.fit_transform(X_scaled)
        X_pca = X_pca_full[:, :3]  # Keep first 3 components for visualization

        # 3. Create age groups based on the actual age distribution in the dataset
        age_bins = [0, 20, 30, 40, 100]
        age_labels_long = ['Young (≤20)', 'Adult (20-30)', 'Middle-aged (30-40)', 'Senior (>40)']
        age_labels_short = ['Yng', 'Adlt', 'Mid', 'Sen']  # Shortened age labels for legend
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels_long)  # Use long labels for dataframe
        y_age = pd.Categorical(df['age_group']).codes

        # 4. Visualization and Analysis - Grid Layout
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 20))  # Adjusted figure size for 2x2 layout

        categorical_attributes_for_shape = ['sex', 'education', 'relationship']
        base_filename = OUTPUT_PLOT_FILENAME.replace('.png', '')

        # Create 3 PCA plots in positions 1, 2, and 3
        for plot_index, shape_attribute in enumerate(categorical_attributes_for_shape):
            ax = fig.add_subplot(2, 2, plot_index + 1, projection='3d')
            attribute_markers = {}
            unique_attributes = df[shape_attribute].unique()
            markers = ['o', '^', 's', 'P', '*', 'X', 'D', 'v', '<', '>']
            for i, attr_val in enumerate(unique_attributes):
                attribute_markers[attr_val] = markers[i % len(markers)]

            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

            for age_group_code, (color, age_label_short) in enumerate(
                    zip(colors, age_labels_short)):
                for attr_value in unique_attributes:
                    mask = (y_age == age_group_code) & (df[shape_attribute] == attr_value)
                    marker = attribute_markers[attr_value]
                    legend_label = f'{age_label_short}, {shape_attribute[:3].upper()}={attr_value}'
                    scatter = ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                                      c=color, label=legend_label, marker=marker, alpha=0.6, s=50)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=12)
            ax.set_title(f'PCA by Age & {shape_attribute}', fontsize=14)
            ax.legend(fontsize=13)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.view_init(elev=20, azim=45)

        # Dataset Statistics
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = "Dataset Statistics:\n\n"

        # Create a dictionary to hold the statistics for JSON
        stats_dict = {}

        # Add age distribution statistics
        stats_text += "Age Distribution:\n"
        age_counts = df['age_group'].value_counts().sort_index()
        stats_dict["age_distribution"] = {}
        for age_group, count in age_counts.items():
            percentage = count / len(df) * 100
            stats_text += f"{age_group}: {count} ({percentage:.1f}%)\n"
            stats_dict["age_distribution"][str(age_group)] = {
                "count": int(count),
                "percentage": float(f"{percentage:.1f}")
            }

        stats_text += "\nNumerical Features (mean ± std):\n"
        stats_dict["numerical_features"] = {}
        for feature in numerical_features:
            mean = df[feature].mean()
            std = df[feature].std()
            stats_text += f"{feature}: {mean:.2f} ± {std:.2f}\n"
            stats_dict["numerical_features"][feature] = {
                "mean": float(f"{mean:.2f}"),
                "std": float(f"{std:.2f}")
            }

        # Add categorical feature statistics
        stats_text += "\nCategorical Features (top 2 most common):\n"
        stats_dict["categorical_features"] = {}
        for feature in categorical_features:
            value_counts = df[feature].value_counts().nlargest(2)
            stats_text += f"{feature}:\n"
            stats_dict["categorical_features"][feature] = {}
            for val, count in value_counts.items():
                percentage = count / len(df) * 100
                stats_text += f"  {val}: {percentage:.1f}%\n"
                stats_dict["categorical_features"][feature][str(val)] = {
                    "count": int(count),
                    "percentage": float(f"{percentage:.1f}")
                }

        ax4.text(0, 0.95, stats_text, va='top', fontsize=13, family='monospace')

        # Save the statistics as JSON
        json_output_path = os.path.join(output_dir, "prototype-stats.json")
        with open(json_output_path, 'w') as json_file:
            json.dump(stats_dict, json_file, indent=4)
        print(f"Statistics saved to: {json_output_path}")

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Add a suptitle for the whole figure
        fig.suptitle('Prototype Distribution Analysis by Age and Selected Attributes', fontsize=18)

        # Save the visualization
        output_filename = os.path.join(output_dir, OUTPUT_PLOT_FILENAME)
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"Visualization saved to: {output_filename}")
        plt.close(fig)

    except Exception as e:
        print(f"Error processing {prototypes_csv_path}: {e}")


if __name__ == "__main__":
    base_repo_dir = os.getcwd()  # Assuming script is run from the base repo dir

    # For direct testing with the CSV file in the current directory
    if os.path.exists(PROTOTYPES_CSV_FILENAME):
        print(f"Found {PROTOTYPES_CSV_FILENAME} in current directory, analyzing directly.")
        os.makedirs("output", exist_ok=True)
        analyze_prototypes(PROTOTYPES_CSV_FILENAME, "output")

    # Original directory scanning logic
    experiment_base_dirs = [
        os.path.join(base_repo_dir, d)
        for d in os.listdir(base_repo_dir)
        if os.path.isdir(os.path.join(base_repo_dir, d)) and d.startswith(EXPERIMENT_BASE_DIR_PREFIX)
    ]

    for experiment_base_dir in experiment_base_dirs:
        print(f"\n--- Processing experiments in: {experiment_base_dir} ---")
        experiment_dirs = [
            os.path.join(experiment_base_dir, d)
            for d in os.listdir(experiment_base_dir)
            if os.path.isdir(os.path.join(experiment_base_dir, d)) and d.startswith('lambda_diversity_')
        ]

        if not experiment_dirs:
            print(f"No experiment subdirectories (starting with 'lambda_diversity_') found in: {experiment_base_dir}")

        for experiment_dir in experiment_dirs:
            model_dir = os.path.join(experiment_dir, MODEL_DIR_NAME)
            prototypes_csv_file = os.path.join(model_dir, PROTOTYPES_CSV_FILENAME)
            plots_dir = model_dir

            if os.path.exists(prototypes_csv_file):
                os.makedirs(plots_dir, exist_ok=True)
                analyze_prototypes(prototypes_csv_file, plots_dir)
            else:
                print(f"Prototypes CSV not found in: {model_dir}")

    print("\nPrototype analysis complete for all directories.")
