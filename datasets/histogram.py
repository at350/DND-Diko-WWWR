import matplotlib.pyplot as plt
import os
import yaml
import pickle as pkl
from collections import Counter

def get_labels_distribution(data_root, croptype, split):
    labels = None

    if split == 'train':
        with open(f'{croptype}\\train_labels.pkl', 'rb') as f:
            labels = pkl.load(f)
    elif split == 'val':
        with open(f'{croptype}\\val_labels.pkl', 'rb') as f:
            labels = pkl.load(f)

    return Counter(labels)


def plot_histogram(data_root, croptype):
    # Retrieve label distribution
    train_distribution = get_labels_distribution(data_root, croptype, 'train')
    val_distribution = get_labels_distribution(data_root, croptype, 'val')

    # Get unique labels (classes)
    labels = sorted(list(train_distribution.keys()))

    # Values for the bars
    train_values = [train_distribution[label] for label in labels]
    val_values = [val_distribution[label] for label in labels]

    # Bar width
    bar_width = 0.35
    r1 = range(len(train_values))
    r2 = [x + bar_width for x in r1]

    # Create bars
    plt.bar(r1, train_values, width=bar_width, color='blue', label='Train')
    plt.bar(r2, val_values, width=bar_width, color='red', label='Val')

    # Title & Subtitle
    plt.xlabel('Classes', fontweight='bold')
    plt.ylabel('Counts', fontweight='bold')

    # Rotation of the labels
    plt.xticks([r + bar_width for r in range(len(train_values))], labels, rotation=45)

    # Create legend & Show graphic
    plt.legend()
    plt.tight_layout()
    plt.show()


# To generate the histogram, call the following function with your data_root and croptype:
plot_histogram("../../DND-Diko-WWWR", "WW2020")


