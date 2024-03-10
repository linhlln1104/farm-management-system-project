import argparse
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv
from config import TrackingConfig
from supervision import PolygonZoneAnnotator, BoundingBoxAnnotator, calculate_dynamic_line_thickness, calculate_dynamic_text_scale

COLORS = sv.ColorPalette.default()

ZONE_IN_POLYGONS = [
    np.array([
        [1015, 1], [1919, 1], [1919, 1079], [1015, 1079], [1015, 1]
    ]),
]

ZONE_OUT_POLYGONS = [
    np.array([
        [1015, 1], [1019, 1079], [1, 1079], [1, 1], [1015, 1]
    ]),
]


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
            self,
            detections_all: sv.Detections,
            detections_in_zones: List[sv.Detections],
            detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        detections_all.zone_id = np.full(len(detections_all), -1, dtype=int)
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
                detections_all.zone_id[detections_all.tracker_id == tracker_id] = zone_in_id

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        return detections_all


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

        line_thickness = calculate_dynamic_line_thickness(resolution_wh=self.video_info.resolution_wh)
        text_scale = calculate_dynamic_text_scale(resolution_wh=self.video_info.resolution_wh)

        self.zone_in_annotators = []
        self.box_in_annotators = []
        self.zone_out_annotators = []
        self.box_out_annotators = []

        for i in range(max(len(self.zones_in), len(self.zones_out))):
            zone_in_color = COLORS.by_idx(i) if i < len(self.zones_in) else COLORS.default()
            zone_out_color = COLORS.by_idx(i) if i < len(self.zones_out) else COLORS.default()

            if i < len(self.zones_in):
                zone_in_annotator = PolygonZoneAnnotator(
                    zone=self.zones_in[i], color=zone_in_color, thickness=line_thickness, text_thickness=line_thickness * 2, text_scale=text_scale * 2
                )
                box_in_annotator = BoundingBoxAnnotator(color=zone_in_color, thickness=line_thickness)
            else:
                zone_in_annotator = None
                box_in_annotator = None

            if i < len(self.zones_out):
                zone_out_annotator = PolygonZoneAnnotator(
                    zone=self.zones_out[i], color=zone_out_color, thickness=line_thickness, text_thickness=line_thickness * 2, text_scale=text_scale * 2
                )
                box_out_annotator = BoundingBoxAnnotator(color=zone_out_color, thickness=line_thickness)
            else:
                zone_out_annotator = None
                box_out_annotator = None

            self.zone_in_annotators.append(zone_in_annotator)
            self.box_in_annotators.append(box_in_annotator)
            self.zone_out_annotators.append(zone_out_annotator)
            self.box_out_annotators.append(box_out_annotator)

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            delay = 30  # Adjust this value to control the playback speed (higher value = slower)
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in_annotator, zone_out_annotator, box_in_annotator, box_out_annotator) in enumerate(zip(self.zone_in_annotators, self.zone_out_annotators, self.box_in_annotators, self.box_out_annotators)):
            detections_in_zone = detections[detections.zone_id == i]
            detections_out_zone = detections[detections.zone_id == -1]  # Assuming -1 means outside all zones

            annotated_frame = zone_in_annotator.annotate(scene=annotated_frame)
            annotated_frame = box_in_annotator.annotate(scene=annotated_frame, detections=detections_in_zone)
            annotated_frame = zone_out_annotator.annotate(scene=annotated_frame)
            annotated_frame = box_out_annotator.annotate(scene=annotated_frame, detections=detections_out_zone)

        labels = [
            f"#{tracker_id} {self.model.names[int(class_id)]} Zone: {zone_id}"
            for class_id, tracker_id, zone_id in zip(detections.class_id, detections.tracker_id, detections.zone_id)
        ]

        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold, classes=TrackingConfig.CLASSES
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    processor = VideoProcessor(
        source_weights_path=TrackingConfig.MODEL_WEIGHTS,
        source_video_path=TrackingConfig.VIDEO_FILE_ZONE,
        target_video_path=TrackingConfig.OUTPUT_PATH,
        #target_video_path=None,
        confidence_threshold=TrackingConfig.CONFIDENCE,
        iou_threshold=TrackingConfig.IOU,
    )
    processor.process_video()