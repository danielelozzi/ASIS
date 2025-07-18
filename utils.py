import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_history(history, filepath):
    """Salva la cronologia di addestramento in un file JSON."""
    print(f"Salvataggio cronologia di addestramento in: {filepath}")
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)

def plot_history(history, filepath):
    """Crea e salva un grafico della cronologia di addestramento (loss e accuracy)."""
    print(f"Creazione grafico di addestramento in: {filepath}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Grafico della Loss
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Grafico dell'Accuratezza
    ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filepath, format='pdf')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, title, filepath):
    """Crea e salva una matrice di confusione normalizzata come heatmap."""
    print(f"Creazione matrice di confusione in: {filepath}")
    cm = confusion_matrix(y_true, y_pred)
    # Normalizza la matrice di confusione per riga (cioè per classe vera)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, format='pdf')
    plt.close()