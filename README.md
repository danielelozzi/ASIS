# Real-Time Sleep Stage Prediction with LSTM

This project implements a modular system in Python for real-time sleep stage prediction using EEG data. The core idea is to predict a person's sleep stage at a future point in time, allowing for a configurable "gap" between the last available data and the prediction target. This code is designed to be used into Google Colab.

The system uses a dual-model approach:
1.  **LSTM (Long Short-Term Memory) Network**: To analyze sequences of past sleep epochs and predict a future sleep stage. This is ideal for capturing temporal dependencies in sleep patterns.
2.  **Random Forest Classifier**: As a baseline, this model classifies the *current* sleep stage based on the features of a single epoch, confirming the system's ability to process the data effectively.

This work is a practical implementation and continuation of the concepts presented in the following research paper:
> Lozzi, D., Di Matteo, A., Mattei, E., Cipriani, A., Caianiello, P., Mignosi, F., & Placidi, G. (2024, October). ASIS: A Smart Alarm Clock Based on Deep Learning for the Safety of Night Workers. In *2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)* (pp. 1129-1134). IEEE.

### BibTeX Citation
```bibtex
@inproceedings{lozzi2024asis,
  title={ASIS: A Smart Alarm Clock Based on Deep Learning for the Safety of Night Workers},
  author={Lozzi, Daniele and Di Matteo, Alessandro and Mattei, Enrico and Cipriani, Alessia and Caianiello, Pasquale and Mignosi, Filippo and Placidi, Giuseppe},
  booktitle={2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)},
  pages={1129--1134},
  year={2024},
  organization={IEEE}
}
```

---

## Features

- **Modular Design**: The code is split into logical modules for configuration, data loading, feature extraction, modeling, training, and prediction.
- **Automatic Data Fetching**: Uses the `mne` library to automatically download and cache the PhysioNet Sleep-EDF dataset, simplifying setup.
- **Configurable Prediction Gap**: Easily adjust how far into the future the model predicts.
- **Real-Time Simulation**: The `predict.py` script simulates how the system would operate overnight, making periodic predictions.

## Project Structure

The project is organized into the following Python scripts:

-   `config.py`: A central file for all global parameters.
-   `data_loader.py`: Handles the automatic download and loading of data using MNE-Python. 
-   `feature_extractor.py`: Calculates Power Spectral Density (PSD) features.
-   `models.py`: Defines the Keras LSTM and Scikit-learn Random Forest models.
-   `train.py`: Orchestrates the training pipeline.
-   `predict.py`: Loads pre-trained models to run a prediction simulation.
-   `requirements.txt`: Lists all necessary Python packages.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install the required dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** The dataset will be downloaded automatically by the `data_loader.py` script on the first run. You no longer need to download it manually.

## How to Use

The process is divided into two main steps: training and prediction.

### 1. Train the Models

First, you need to train the models. The `train.py` script is configured to use a specific number of subjects from the dataset for training. On the first run, MNE will download the necessary data, which may take some time.

```bash
python train.py
```

This will create a `models/` directory containing the trained LSTM model (`.h5`), the Random Forest model (`.pkl`), and the data scaler (`.pkl`). You can change the number of subjects to train on by editing the `subjects_to_train_on` list at the bottom of `train.py`.

### 2. Run a Prediction Simulation

Once the models are trained and saved, you can run a simulation to see the prediction system in action. The `predict.py` script will download the data for a specific subject (if not already cached) and output real-time predictions for future sleep stages.

```bash
python predict.py
```

You can customize the simulation by editing the parameters in `predict.py`, such as the subject ID, `prediction_gap_minutes`, and `update_interval_minutes`.
