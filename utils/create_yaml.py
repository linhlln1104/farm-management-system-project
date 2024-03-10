import os
import yaml
from config import ModelConfig


def create_yaml_file():

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    dict_file = {
        'train': os.path.join(ModelConfig.CUSTOM_DATASET_DIR, 'train'),
        'val': os.path.join(ModelConfig.CUSTOM_DATASET_DIR, 'valid'),
        'test': os.path.join(ModelConfig.CUSTOM_DATASET_DIR, 'test'),
        'nc': ModelConfig.NUM_CLASSES_TO_TRAIN,
        'names': ModelConfig.CLASSES
    }

    with open(os.path.join(output_dir, 'data.yaml'), 'w+') as file:
        yaml.dump(dict_file, file)