Process Overview
================

The vulnerability detection pipeline consists of three main stages: Data Processing, Model Training, and Prediction/Evaluation.

1. **Data Processing**
   
   The raw source code (C/C++) from the DiverseVul dataset is processed to generate graph representations. The code is converted into Abstract Syntax Trees (AST) and then transformed into NetworkX graphs. These graphs are then converted into PyTorch Geometric (PyG) ``Data`` objects, containing node features and edge indices suitable for GNN input. The dataset is split into training, validation, and test sets using a stratified multi-label split strategy.

   Script: ``src/process_dataset.py`` or implicitly within ``src/train.py``.

2. **Model Training**

   The processed graph data is used to train a Graph Neural Network (GNN). The system supports different GNN architectures (configured via YAML). The training process optimizes the model to classify code samples into one or more Common Weakness Enumerations (CWEs).

   Script: ``src/train.py``

3. **Prediction and Evaluation**

   The trained model checkpoint is loaded to perform inference on new data or to evaluate performance metrics (Accuracy, F1-Score, Precision, Recall, ROC-AUC) on the test set.

   Script: ``src/run_prediction.py``
