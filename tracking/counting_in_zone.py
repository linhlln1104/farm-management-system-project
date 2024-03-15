import json
from typing import List, Tuple
import ctypes
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv
from config import TrackingConfig

COLORS = sv.ColorPalette.default()

def load_zones_config(file_path: str) -> List[np.ndarray]:

    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]


def initiate_annotators(
    polygons: List[np.ndarray], resolution_wh: Tuple[int, int]
) -> Tuple[
    List[sv.PolygonZone], List[sv.PolygonZoneAnnotator], List[sv.BoundingBoxAnnotator]
]:
    line_thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=resolution_wh)

    zones = []
    zone_annotators = []
    box_annotators = []

    for index, polygon in enumerate(polygons):
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=resolution_wh)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=COLORS.by_idx(index),
            thickness=line_thickness,
            text_thickness=line_thickness * 2,
            text_scale=text_scale * 2,
        )
        box_annotator = sv.BoundingBoxAnnotator(
            color=COLORS.by_idx(index), thickness=line_thickness
        )
        zones.append(zone)
        zone_annotators.append(zone_annotator)
        box_annotators.append(box_annotator)

    return zones, zone_annotators, box_annotators


def detect(
    frame: np.ndarray, model: YOLO, confidence_threshold: float = 0.5
) -> sv.Detections:

    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    filter_by_class = detections.class_id == 2
    filter_by_confidence = detections.confidence > confidence_threshold
    return detections[filter_by_class & filter_by_confidence]


def annotate(
    frame: np.ndarray,
    zones: List[sv.PolygonZone],
    zone_annotators: List[sv.PolygonZoneAnnotator],
    box_annotators: List[sv.BoundingBoxAnnotator],
    detections: sv.Detections,
) -> np.ndarray:

    annotated_frame = frame.copy()
    for zone, zone_annotator, box_annotator in zip(
        zones, zone_annotators, box_annotators
    ):
        detections_in_zone = detections[zone.trigger(detections=detections)]
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections_in_zone
        )
    return annotated_frame

if __name__ == "__main__":
    zone_configuration_path = TrackingConfig.ZONE_CONFIGURATION_PATH
    source_video_path = TrackingConfig.VIDEO_FILE_ZONE
    #target_video_path = TrackingConfig.OUTPUT_PATH
    target_video_path = None
    source_weights_path = TrackingConfig.MODEL_WEIGHTS
    confidence_threshold = TrackingConfig.CONFIDENCE
    iou_threshold = TrackingConfig.IOU

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    polygons = load_zones_config(zone_configuration_path)
    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=video_info.resolution_wh
    )

    model = YOLO(source_weights_path)

    frames_generator = sv.get_video_frames_generator(source_video_path)

    if target_video_path is not None:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                detections = detect(frame, model, confidence_threshold)
                annotated_frame = annotate(
                    frame=frame,
                    zones=zones,
                    zone_annotators=zone_annotators,
                    box_annotators=box_annotators,
                    detections=detections,
                )
                sink.write_frame(annotated_frame)
    else:
        for frame in tqdm(frames_generator, total=video_info.total_frames):
            detections = detect(frame, model, confidence_threshold)
            annotated_frame = annotate(
                frame=frame,
                zones=zones,
                zone_annotators=zone_annotators,
                box_annotators=box_annotators,
                detections=detections,
            )

            user32 = ctypes.windll.user32
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)

            # Resize the frame to fit the screen
            max_width = screen_width - TrackingConfig.SCALE
            max_height = screen_height - TrackingConfig.SCALE
            frame_height, frame_width = annotated_frame.shape[:2]
            scale_factor = min(max_width / frame_width, max_height / frame_height)
            resized_frame = cv2.resize(annotated_frame, None, fx=scale_factor, fy=scale_factor)

            delay = TrackingConfig.DELAY
            # Display the resized frame
            cv2.imshow("Processed Video", resized_frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
