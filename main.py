import warnings
warnings.filterwarnings("ignore")
import os

from config import ModelConfig

from utils.create_yaml import create_yaml_file
from utils.data_utils import (read_yaml_file,
                              print_yaml_data,
                              display_image,
                              plot_random_images_from_folder,
                              get_image_properties,
                              show_image_properties,
                              plot_dataset_statistics,)

from utils.evaluation import plot_training_metrics
from utils.data_processing import analyze_dataset

from models.model_training import train_model



"""
# Analyze dataset
dataset_stats_df = analyze_dataset(ModelConfig)
plot_dataset_statistics(dataset_stats_df)


#Display image properties
example_image_path = os.path.join(ModelConfig.CUSTOM_DATASET_DIR, 'train', 'images', 'sheep-226-_jpeg_jpg.rf.c2063c20488e19a5206bbf628662157e.jpg')
img_properties = get_image_properties(example_image_path)
img_properties


# Plot random images from the training folder
folder_path = os.path.join(ModelConfig.CUSTOM_DATASET_DIR, 'train', 'images')
plot_random_images_from_folder(folder_path, num_images=20, seed=ModelConfig.SEED)


# Train the model
train_model(ModelConfig)


# Load training log
df = pd.read_csv(f'{ModelConfig.OUTPUT_DIR}runs/detect/{ModelConfig.BASE_MODEL}_{ModelConfig.EXP_NAME}/results.csv')


# Plot training metrics
plot_training_metrics(df)
"""

def main():
    create_yaml_file()

    # Load YAML file
    file_path = os.path.join(ModelConfig.OUTPUT_DIR, 'data.yaml')
    yaml_data = read_yaml_file(file_path)

    if yaml_data:
        print_yaml_data(yaml_data)

    train_model(ModelConfig)

if __name__ == '__main__':
    main()
