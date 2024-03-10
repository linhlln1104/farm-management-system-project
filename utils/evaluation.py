import pandas as pd
import matplotlib.pyplot as plt

def plot_training_metrics(df):
    # Create subplots with 1 row and 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Training and Validation Box Loss
    ax1.set_title('Box Loss')
    ax1.plot(df['epoch'], df['train/box_loss'], label='Training box_loss', marker='o', linestyle='-')
    ax1.plot(df['epoch'], df['val/box_loss'], label='Validation box_loss', marker='o', linestyle='-')
    ax1.set_ylabel('Box Loss')
    ax1.legend()
    ax1.grid(True)

    # Training and Validation cls_loss
    ax2.set_title('Cls Loss')
    ax2.plot(df['epoch'], df['train/cls_loss'], label='Training cls_loss', marker='o', linestyle='-')
    ax2.plot(df['epoch'], df['val/cls_loss'], label='Validation cls_loss', marker='o', linestyle='-')
    ax2.set_ylabel('cls_loss')
    ax2.legend()
    ax2.grid(True)

    # Training and Validation dfl_loss
    ax3.set_title('DFL Loss')
    ax3.plot(df['epoch'], df['train/dfl_loss'], label='Training dfl_loss', marker='o', linestyle='-')
    ax3.plot(df['epoch'], df['val/dfl_loss'], label='Validation dfl_loss', marker='o', linestyle='-')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('dfl_loss')
    ax3.legend()
    ax3.grid(True)

    plt.suptitle('Training Metrics vs. Epochs')
    plt.show()
