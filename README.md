# Real-Time Sleep Stage Prediction with LSTM

This project implements a modular system in Python for real-time sleep stage prediction using EEG data. The core idea is to predict a person's sleep stage at a future point in time, allowing for a configurable "gap" between the last available data and the prediction target.

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
- **Configurable Prediction Gap**: Easily adjust how far into the future the model predicts (e.g., predict the sleep stage 30, 60, or 90 minutes from now).
- **Subject-Dependent Fine-Tuning (Concept)**: The structure is designed to be extendable for fine-tuning the general model on a specific subject's data.
- **Real-Time Simulation**: The `predict.py` script simulates how the system would operate overnight, making periodic predictions.
- **Standard Dataset**: Uses the well-known [PhysioNet Sleep EDFx Expanded Database](https://www.physionet.org/content/sleep-edfx/1.0.0/).

## Project Structure

The project is organized into the following Python scripts:

-   `config.py`: A central file for all global parameters, such as EEG frequency bands, model hyperparameters, and label mappings.
-   `data_loader.py`: Handles the loading and parsing of EDF files from the PhysioNet dataset. It segments the data into 30-second epochs.
-   `feature_extractor.py`: Calculates the Power Spectral Density (PSD) for different frequency bands from the raw EEG signals.
-   `models.py`: Defines the Keras LSTM model and the Scikit-learn Random Forest classifier.
-   `train.py`: Orchestrates the training pipeline. It loads data from all subjects, extracts features, and trains both the general LSTM and Random Forest models.
-   `predict.py`: Loads the pre-trained models to run a prediction simulation on a single subject's data, demonstrating the system's capabilities.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone http://github.com/danielelozzi/ASIS
    cd ASIS
    ```

2.  **Install the required dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    If a `requirements.txt` is not available, install the main packages manually:
    ```bash
    pip install tensorflow scikit-learn mne numpy joblib
    ```

3.  **Download the Dataset:**
    - Download the **Sleep EDFx Expanded** dataset from [PhysioNet](https://www.physionet.org/content/sleep-edfx/1.0.0/).
    - Extract the `sleep-cassette` folder into the root of the project directory, or update the `DATASET_PATH` variable in the `train.py` and `predict.py` scripts to point to its location.

## How to Use

The process is divided into two main steps: training and prediction.

### 1. Train the Models

First, you need to train the models on the entire dataset. This will create a `models/` directory containing the trained LSTM model (`.h5`), the Random Forest model (`.pkl`), and the data scaler (`.pkl`).

```bash
python train.py
```
This process may take some time depending on your hardware and the size of the dataset.

### 2. Run a Prediction Simulation

Once the models are trained and saved, you can run a simulation to see the prediction system in action. The script will pick a subject from the dataset and output real-time predictions for future sleep stages.

```bash
python predict.py
```

You can customize the simulation by editing the parameters in the `if __name__ == '__main__':` block of `predict.py`, such as `prediction_gap_minutes` and `update_interval_minutes`.
