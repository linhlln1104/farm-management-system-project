import time
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from supervision import PolygonZoneAnnotator, BoundingBoxAnnotator, calculate_dynamic_line_thickness, \
    calculate_dynamic_text_scale

from config import TrackingConfig

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
            zone_in_polygons: List[np.ndarray] = None,  # Thêm tham số mới
            zone_out_polygons: List[np.ndarray] = None,  # Thêm tham số mới
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        # Sử dụng các đa giác được cung cấp hoặc giá trị mặc định
        self.zones_in = initiate_polygon_zones(
            zone_in_polygons or ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )
        self.zones_out = initiate_polygon_zones(
            zone_out_polygons or ZONE_OUT_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

        # Heatmap configuration
        self.heat_map_annotator = sv.HeatMapAnnotator(
            position=sv.Position.BOTTOM_CENTER,
            opacity=TrackingConfig.HEATMAP_ALPHA,
            radius=TrackingConfig.RADIUS,
            kernel_size=25,
            top_hue=0,
            low_hue=125,
        )

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
                    zone=self.zones_in[i], color=zone_in_color, thickness=line_thickness,
                    text_thickness=line_thickness * 2, text_scale=text_scale * 2
                )
                box_in_annotator = BoundingBoxAnnotator(color=zone_in_color, thickness=line_thickness)
            else:
                zone_in_annotator = None
                box_in_annotator = None

            if i < len(self.zones_out):
                zone_out_annotator = PolygonZoneAnnotator(
                    zone=self.zones_out[i], color=zone_out_color, thickness=line_thickness,
                    text_thickness=line_thickness * 2, text_scale=text_scale * 2
                )
                box_out_annotator = BoundingBoxAnnotator(color=zone_out_color, thickness=line_thickness)
            else:
                zone_out_annotator = None
                box_out_annotator = None

            self.zone_in_annotators.append(zone_in_annotator)
            self.box_in_annotators.append(box_in_annotator)
            self.zone_out_annotators.append(zone_out_annotator)
            self.box_out_annotators.append(box_out_annotator)

        self.prev_time = 0

    def get_first_frame(self):
        cap = cv2.VideoCapture(self.source_video_path)
        ret, frame = cap.read()
        cap.release()
        return frame

    def process_video(self):
        import ctypes

        # Get the screen resolution
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        if TrackingConfig.ENABLE_REALTIME_STREAMING:
            cap = cv2.VideoCapture(self.source_video_path)
            delay = TrackingConfig.DELAY  # Adjust this value to control the playback speed (higher value = slower)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = self.process_frame(frame)

                # Resize the frame to fit the screen
                max_width = screen_width - TrackingConfig.SCALE
                max_height = screen_height - TrackingConfig.SCALE
                frame_height, frame_width = annotated_frame.shape[:2]
                scale_factor = min(max_width / frame_width, max_height / frame_height)
                resized_frame = cv2.resize(annotated_frame, None, fx=scale_factor, fy=scale_factor)

                current_time = time.time()
                fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time

                # Put FPS text on the frame
                cv2.putText(resized_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                # Display the resized frame
                cv2.imshow("Processed Video", resized_frame)
                key = cv2.waitKey(delay) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    TrackingConfig.ENABLE_COUNTING = not TrackingConfig.ENABLE_COUNTING
                elif key == ord("a"):
                    TrackingConfig.ENABLE_TRACE_ANNOTATOR = not TrackingConfig.ENABLE_TRACE_ANNOTATOR
                elif key == ord("h"):
                    TrackingConfig.ENABLE_HEATMAP = not TrackingConfig.ENABLE_HEATMAP

            cap.release()
            cv2.destroyAllWindows()

        else:
            frame_generator = sv.get_video_frames_generator(
                source_path=self.source_video_path
            )

            if self.target_video_path:
                with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                    for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                        annotated_frame = self.process_frame(frame)
                        sink.write_frame(annotated_frame)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()

        # Draw the heatmap
        if TrackingConfig.ENABLE_HEATMAP and detections is not None:
            annotated_frame = self.heat_map_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

        for i, (zone_in_annotator, zone_out_annotator, box_in_annotator, box_out_annotator) in enumerate(
                zip(self.zone_in_annotators, self.zone_out_annotators, self.box_in_annotators,
                    self.box_out_annotators)):
            if detections is not None:
                detections_in_zone = detections[detections.zone_id == i]
                detections_out_zone = detections[detections.zone_id == -1]  # Assuming -1 means outside all zones

                annotated_frame = zone_in_annotator.annotate(scene=annotated_frame)
                annotated_frame = box_in_annotator.annotate(scene=annotated_frame, detections=detections_in_zone)
                annotated_frame = zone_out_annotator.annotate(scene=annotated_frame)
                annotated_frame = box_out_annotator.annotate(scene=annotated_frame, detections=detections_out_zone)

        if detections is not None:
            labels = [
                f"#{tracker_id} {self.model.names[int(class_id)]} Zone: {zone_id}"
                for class_id, tracker_id, zone_id in zip(detections.class_id, detections.tracker_id, detections.zone_id)
            ]

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

        if TrackingConfig.ENABLE_TRACE_ANNOTATOR and detections is not None:
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.box_annotator.annotate(
                annotated_frame, detections, labels
            )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if TrackingConfig.ENABLE_COUNTING:
            results = self.model(
                frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold, classes=TrackingConfig.CLASSES
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = None

        detections_in_zones = []
        detections_out_zones = []

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            if detections is not None:
                detections_in_zone = detections[zone_in.trigger(detections=detections)]
                detections_in_zones.append(detections_in_zone)
                detections_out_zone = detections[zone_out.trigger(detections=detections)]
                detections_out_zones.append(detections_out_zone)

        if detections is not None:
            detections = self.detections_manager.update(
                detections, detections_in_zones, detections_out_zones
            )

        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    processor = VideoProcessor(
        source_weights_path=TrackingConfig.MODEL_WEIGHTS,
        source_video_path=TrackingConfig.VIDEO_FILE_ZONE,
        # target_video_path = TrackingConfig.OUTPUT_PATH
        target_video_path=None,
        confidence_threshold=TrackingConfig.CONFIDENCE,
        iou_threshold=TrackingConfig.IOU,
    )
    processor.process_video()

