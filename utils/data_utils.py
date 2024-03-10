import os
import cv2
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import seaborn as sns

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print("Error reading YAML:", e)
            return None

def print_yaml_data(data):
    formatted_yaml = yaml.dump(data, default_style=False)
    print(formatted_yaml)

def display_image(image, print_info=True, hide_axis=False):
    if isinstance(image, str):  # Check if it's a file path
        img = Image.open(image)
        plt.imshow(img)
    elif isinstance(image, np.ndarray):  # Check if it's a NumPy array
        image = image[..., ::-1]  # BGR to RGB
        img = Image.fromarray(image)
        plt.imshow(img)
    else:
        raise ValueError("Unsupported image format")

    if print_info:
        print('Type: ', type(img), '\n')
        print('Shape: ', np.array(img).shape, '\n')

    if hide_axis:
        plt.axis('off')

    plt.show()

def plot_random_images_from_folder(folder_path, num_images=20, seed=88):
    random.seed(seed)

    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]

    # Ensure that we have at least num_images files to choose from
    if len(image_files) < num_images:
        raise ValueError("Not enough images in the folder")

    # Randomly select num_images image files
    selected_files = random.sample(image_files, num_images)

    # Create a subplot grid
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    for i, file_name in enumerate(selected_files):
        # Open and display the image using PIL
        img = Image.open(os.path.join(folder_path, file_name))

        if num_rows == 1:
            ax = axes[i % num_cols]
        else:
            ax = axes[i // num_cols, i % num_cols]

        ax.imshow(img)
        ax.axis('off')

    # Remove empty subplots
    for i in range(num_images, num_rows * num_cols):
        if num_rows == 1:
            fig.delaxes(axes[i % num_cols])
        else:
            fig.delaxes(axes[i // num_cols, i % num_cols])

    plt.tight_layout()
    plt.show()

def get_image_properties(image_path):
    # Read the image file
    img = cv2.imread(image_path)

    # Check if the image file is read successfully
    if img is None:
        raise ValueError("Could not read image file")

    # Get image properties
    properties = {
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "dtype": img.dtype,
    }

    return properties

def show_image_properties(image_path):
    img_properties = get_image_properties(image_path)
    print("Image Properties:")
    for key, value in img_properties.items():
        print(f"{key}: {value}")

def plot_dataset_statistics(dataset_stats_df):
    # Create subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot vertical bar plots for each mode in subplots
    for i, mode in enumerate(['train', 'valid', 'test']):
        sns.barplot(
            data=dataset_stats_df[dataset_stats_df['Mode'] == mode].drop(columns='Mode'),
            orient='v',
            ax=axes[i],
            palette='Set2'
        )

        axes[i].set_title(f'{mode.capitalize()} Class Statistics')
        axes[i].set_xlabel('Classes')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=90)

        # Add annotations on top of each bar
        for p in axes[i].patches:
            axes[i].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                             textcoords='offset points')

    plt.tight_layout()
    plt.show()
