
# ProtoCore: Explainable AI through Prototype-Based Neural Networks
This repository contains the code and experiments for my bachelor thesis titled "Qualitative Meets Quantitative: Feed-Forward Neural Networks with Case-Based Reasoning - An Explainability Comparison with Post-Hoc Methods."

## About This Project
This research investigates the explainability of prototype-based neural networks, an inherently interpretable approach inspired by Case-Based Reasoning (CBR), specifically when applied to tabular data. The project compares the explanations generated by prototype-based models with those from post-hoc methods like SHAP (SHapley Additive exPlanations).

Key aspects of this research:
- Adaptation of prototype-based neural networks for tabular data
- Implementation of diversity loss to improve prototype distribution
- Comparison between inherent interpretability and post-hoc explainability
- Analysis of different prototype initialization strategies
- Evaluation based on human comprehensibility, fidelity, and actionability

The experiments use the Adult Income dataset to demonstrate how prototype-based models can provide case-based explanations that complement feature attribution methods.

## Setup

### Prerequisites
- Anaconda or Miniconda

### Installation
1. Clone the repository

2. Create and activate the conda environment:
```bash

conda env create -f  requirements.yaml
conda activate protoCore
```

## Running Experiments

### Initial Experiments
Navigate to the research scripts directory and run the experiments:
```bash

cd researchScripts
python experiment1.py
python experiment2.py
```

### Analysis
Navigate to the experiments directory and run the analysis scripts:

```bash

cd ../experiments
python analyseProto.py
python analyseInitialProto.py
```


## Data
This repository contains the [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult) by Barry Becker and Ronny Kohavi (1996), extracted from the 1994 Census bureau database. The full citation is available in the [citations.bib](./citations.bib) file.