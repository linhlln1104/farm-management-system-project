import os

class ModelConfig:
    DEBUG = False
    FRACTION = 0.05 if DEBUG else 1.0
    SEED = 88

    # classes
    CLASSES = ['cow', 'horse', 'sheep', 'zebra']

    NUM_CLASSES_TO_TRAIN = len(CLASSES)

    # training
    EPOCHS = 3 if DEBUG else 1  # 100
    BATCH_SIZE = 16

    BASE_MODEL = 'yolov8s'  #yolov8m, yolov8l, yolov8x
    BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'
    EXP_NAME = f'{EPOCHS}_epochs'

    OPTIMIZER = 'auto'  # SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
    LR = 1e-3
    LR_FACTOR = 0.01
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.0
    PATIENCE = 20
    PROFILE = False
    LABEL_SMOOTHING = 0.0
    DEVICE = 0 #[0, 1]

    # paths
    CUSTOM_DATASET_DIR = r'C:\Users\Administrator\Desktop\Farm\DAT_animal_dataset'
    OUTPUT_PATH = "../output/"



class TrackingConfig:
    MODEL_WEIGHTS = os.path.join("..", "model_weight", "yolov9c_50_epochs.pt")

    ### detections (YOLO)
    CONFIDENCE = 0.6
    IOU = 0.5

    ### heatmap (Supervision)
    HEATMAP_ALPHA = 0.5
    RADIUS = 30

    ### tracking (Supervision, bytetrack)
    TRACK_THRESH = 0.35
    TRACK_SECONDS = 5
    MATCH_THRESH = 0.9999

    ### paths: video file path, webcam is 0
    VIDEO_FILE = os.path.join("..", "video_demo", "video_sheep_demo.mp4")
    OUTPUT_PATH = os.path.join("..", "output", "output.mp4")
    VIDEO_FILE_ZONE = os.path.join("..", "video_demo", "tracking_in_zone_demo_video.mp4")
    ZONE_CONFIGURATION_PATH = os.path.join("..", "data", "quarters-zone-config.json")


    ### flags to turn functions on/off
    ENABLE_REALTIME_STREAMING = True
    ENABLE_HEATMAP = False
    ENABLE_COUNTING = False
    ENABLE_TRACE_ANNOTATOR = False

    DEVICE = 0

    ### 0: 'cow', 1: 'horse', 2: 'sheep', 3: 'zebra'
    CLASSES = [0, 1 ,2, 3]

    # Zone coordinates (x1, y1, x2, y2)
    ZONE_COORDINATES = (100, 100, 300, 300)

    DELAY = 20
    SCALE = 500
