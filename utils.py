# utils.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_history(history, filepath):
    """Salva la cronologia di addestramento in un file JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Salvataggio cronologia di addestramento in: {filepath}")
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)

def plot_history(history, title, filepath):
    """Crea e salva un grafico della cronologia di addestramento."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Creazione grafico di addestramento in: {filepath}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath, format='pdf')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, title, filepath):
    """Crea e salva una matrice di confusione normalizzata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Creazione matrice di confusione in: {filepath}")
    
    # Rimuovi le classi non presenti nei dati reali o predetti
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    present_class_names = [name for i, name in enumerate(class_names) if i in present_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=present_class_names, yticklabels=present_class_names)
    
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, format='pdf')
    plt.close()