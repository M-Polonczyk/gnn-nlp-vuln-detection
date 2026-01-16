Scripts Documentation
=====================

This section documents the main Python scripts located in the ``src/`` directory.

src/process_dataset.py
----------------------

**Purpose**: 
Handles the ETL (Extract, Transform, Load) process for the vulnerability dataset. It converts raw code samples into graph structures required for GNN training.

**Key Steps**:
1. **Load Data**: Reads the DiverseVul dataset using paths defined in ``config/dataset_paths.yaml``.
2. **AST Conversion**: Parses C/C++ code to Abstract Syntax Trees (AST) and converts them to NetworkX graphs using ``DataclassToGraphConverter``.
3. **Stratified Split**: Uses ``iterative_train_test_split`` to divide the multi-label dataset into training (70%), validation (15%), and test (15%) sets, ensuring balanced label distribution.
4. **Feature Extraction**: Fits a ``CodeGraphProcessor`` on the training set to establish node vectorization (likely Word2Vec or similar embedding of AST node types).
5. **Serialization**: Converts graphs to PyTorch Geometric ``Data`` objects (tensors) and saves them as ``.pt`` files in ``data/processed/``.

**Usage**:
Run this script to regenerate the processed dataset files (``train-diversevul-c.pt``, etc.) without triggering training.

.. code-block:: bash

    python src/process_dataset.py

src/train.py
------------

**Purpose**: 
The primary entry point for training the vulnerability detection model.

**Analysis**:
This script currently performs an end-to-end workflow. It **duplicates** the data processing steps found in ``process_dataset.py`` (loading, AST conversion, feature extraction) before starting the training loop. This ensures that the model is trained on a fresh processing of the data, but means it takes longer to start if data is already processed.

**Key Features**:
- **Config Loader**: Loads usage parameters from ``config/training.yaml`` and ``config/model_params.yaml``.
- **End-to-End Execution**: Generates PyG data objects from scratch, saves them, and then immediately reloads them for training.
- **Model Training**: Initializes the GNN classifier and executes the training loop with validation monitoring.
- **Model Saving**: Saves the best model state dict to ``cwe_detector.pth``.
- **Evaluation**: Automatically runs an evaluation on the test set after training completes and prints metrics.

**Usage**:
Execute to retrain the model from scratch.

.. code-block:: bash

    python src/train.py

src/run_prediction.py
---------------------

**Purpose**: 
Standalone script for running inference using a pre-trained model checkpoint and calculating detailed performance metrics.

**Key Features**:
- **Model Loading**: Reconstructs the model architecture based on configuration and loads weights from a hardcoded path (e.g., ``checkpoints/cwe_detector_max_pooling.pth``).
- **Batch Prediction**: Can predict vulnerabilities for batches of data.
- **Metrics**: Calculates Macro-F1, Per-class F1 scores, Precision, Recall, and ROC-AUC.
- **Thresholding**: (Optional) Can load optimal decision thresholds from ``checkpoints/optimal_thresholds.csv``.
- **Visualization**: Generates confusion matrix plots.

**Usage**:
Use this for rigorous evaluation of a trained model or to debug predictions.

.. code-block:: bash

    python src/run_prediction.py

**Note**: Check the ``MODEL_PATH`` constant in the script if you want to evaluate a specific checkpoint.
