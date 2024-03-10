import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from ultralytics import YOLO
import supervision as sv
import subprocess
import cv2
from config import TrackingConfig



def get_video_properties(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    properties = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    # Release the video capture object
    cap.release()

    return properties


video_properties = get_video_properties(TrackingConfig.VIDEO_FILE)
print(video_properties)

model = YOLO(TrackingConfig.MODEL_WEIGHTS)

### heatmap config
heat_map_annotator = sv.HeatMapAnnotator(
    position=sv.Position.BOTTOM_CENTER,
    opacity=TrackingConfig.HEATMAP_ALPHA,
    radius=TrackingConfig.RADIUS,
    kernel_size=25,
    top_hue=0,
    low_hue=125,
)

### annotation config
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

### tracker config
byte_tracker = sv.ByteTrack(
    track_thresh=TrackingConfig.TRACK_THRESH,
    track_buffer=TrackingConfig.TRACK_SECONDS * video_properties['fps'],
    match_thresh=TrackingConfig.MATCH_THRESH,
    frame_rate=video_properties['fps']
)

### video config
video_info = sv.VideoInfo.from_video_path(video_path=TrackingConfig.VIDEO_FILE)
frames_generator = sv.get_video_frames_generator(source_path=TrackingConfig.VIDEO_FILE, stride=1)
output_filename = f"{TrackingConfig.OUTPUT_PATH}heatmap_output_c{int(TrackingConfig.CONFIDENCE * 100)}_iou{int(TrackingConfig.IOU * 100)}.mp4"


def toggle_function(value, function_name):
    global TrackingConfig
    if function_name == "heatmap":
        TrackingConfig.ENABLE_HEATMAP = value
    elif function_name == "tracking":
        TrackingConfig.ENABLE_TRACKING = value
        if not value:
            TrackingConfig.ENABLE_COUNTING = False
    elif function_name == "counting":
        TrackingConfig.ENABLE_COUNTING = value


### Detect, track, annotate, save
unique_tracker_ids = set()

# Initialize flags
TrackingConfig.ENABLE_HEATMAP = True
TrackingConfig.ENABLE_TRACKING = True
TrackingConfig.ENABLE_COUNTING = True


def process_frame(frame):
    result = model(
        source=frame,
        classes=TrackingConfig.CLASSES,
        conf=TrackingConfig.CONFIDENCE,
        iou=TrackingConfig.IOU,
        half=True,  # Set this to False for CPU inference
        show_conf=True,
        save_txt=True,
        save_conf=True,
        save=True,
        device= TrackingConfig.DEVICE
    )[0]

    detections = sv.Detections.from_ultralytics(result)

    # Draw the line

    if TrackingConfig.ENABLE_TRACKING:
        detections = byte_tracker.update_with_detections(detections)
        if TrackingConfig.ENABLE_COUNTING:
            unique_tracker_ids.update(detections.tracker_id)

    annotated_frame = frame.copy()

    if TrackingConfig.ENABLE_HEATMAP:
        ### draw heatmap
        annotated_frame = heat_map_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

    ### draw other attributes from `detections` object
    labels = [
        f"#{tracker_id} {result.names[int(class_id)]}"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]

    label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    if TrackingConfig.ENABLE_COUNTING:
        # Draw the total count
        cv2.putText(annotated_frame, f"Total Count: {len(unique_tracker_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    return annotated_frame


cap = cv2.VideoCapture(TrackingConfig.VIDEO_FILE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        toggle_function(not TrackingConfig.ENABLE_TRACKING, "tracking")
    elif key == ord('2'):
        toggle_function(not TrackingConfig.ENABLE_HEATMAP, "heatmap")
    elif key == ord('3'):
        toggle_function(not TrackingConfig.ENABLE_COUNTING, "counting")

cap.release()
cv2.destroyAllWindows()

