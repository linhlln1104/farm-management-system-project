import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from ultralytics import YOLO
import supervision as sv
import subprocess
import cv2
import os
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

### Detect, track, annotate, save
unique_tracker_ids = set()

with sv.VideoSink(target_path=output_filename, video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(
            source=frame,
            classes=[18],  # only person class
            conf=TrackingConfig.CONFIDENCE,
            iou=TrackingConfig.IOU,
            half=False,  # Set this to False for CPU inference
            show_conf=True,
            save_txt=True,
            save_conf=True,
            save=True,
            device= 0  # Use CPU for inference
        )[0]

        detections = sv.Detections.from_ultralytics(result)

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
            f"#{tracker_id}"
            for tracker_id in set(detections.tracker_id)
        ]

        label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        if TrackingConfig.ENABLE_COUNTING:
            # Draw the total count
            cv2.putText(annotated_frame, f"Total Count: {len(unique_tracker_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        sink.write_frame(frame=annotated_frame)

IN_VIDEO_NAME = f"{TrackingConfig.OUTPUT_PATH}heatmap_output_c{int(TrackingConfig.CONFIDENCE * 100)}_iou{int(TrackingConfig.IOU * 100)}.mp4"
OUT_VIDEO_NAME = f"{TrackingConfig.OUTPUT_PATH}heatmap_and_track.mp4"

subprocess.run(
    [
        "C:\\PATH_Programs\\ffmpeg.exe", "-i", IN_VIDEO_NAME, "-crf",
        "18", "-preset", "veryfast", "-hide_banner", "-loglevel",
        "error", "-vcodec", "libx264", OUT_VIDEO_NAME
    ]
)

print(f"Output video saved as: {OUT_VIDEO_NAME}")