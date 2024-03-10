import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def analyze_dataset(ModelConfig):
    class_idx = {str(i): ModelConfig.CLASSES[i] for i in range(ModelConfig.NUM_CLASSES_TO_TRAIN)}

    class_stat = {}
    data_len = {}
    class_info = []

    for mode in ['train', 'valid', 'test']:
        class_count = {ModelConfig.CLASSES[i]: 0 for i in range(ModelConfig.NUM_CLASSES_TO_TRAIN)}

        path = os.path.join(ModelConfig.CUSTOM_DATASET_DIR, mode, 'labels')

        for file in os.listdir(path):
            with open(os.path.join(path, file)) as f:
                lines = f.readlines()

                for cls in set([line[0] for line in lines]):
                    class_count[class_idx[cls]] += 1

        data_len[mode] = len(os.listdir(path))
        class_stat[mode] = class_count

        class_info.append({'Mode': mode, **class_count, 'Data_Volume': data_len[mode]})

    dataset_stats_df = pd.DataFrame(class_info)
    return dataset_stats_df
