import sys
import numpy as np
import cv2
import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPolygonItem, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

import supervision as sv
from config import TrackingConfig
from tracking_counting_in_zone import VideoProcessor

class PolygonEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polygon Editor")
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Open Video", os.getcwd(), "Video Files (*.mp4 *.avi)")

        if video_path:
            self.video_info = sv.VideoInfo.from_video_path(video_path)
        else:
            pass

        first_frame = self.get_first_frame()
        height, width, _ = first_frame.shape
        first_frame_qimage = QImage(first_frame.data, width, height, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(first_frame_qimage)
        self.scene.addPixmap(pixmap)

        self.zone_in_polygon_points = []
        self.zone_out_polygon_points = []
        self.zone_in_polygon_item = None
        self.zone_out_polygon_item = None

        self.view.setSceneRect(0, 0, 1920, 1080)  # Set the scene size to your video resolution

        self.view.mousePressEvent = self.mouse_press_event
        self.view.mouseMoveEvent = self.mouse_move_event
        self.view.mouseReleaseEvent = self.mouse_release_event

        self.zone_in_button = QPushButton("Edit Zone In Polygon")
        self.zone_out_button = QPushButton("Edit Zone Out Polygon")
        self.run_button = QPushButton("Run Video Processor")

        # Customize button styles using CSS
        button_style = "QPushButton { background-color: #4CAF50; color: white; border: none; padding: 5px 10px; " \
                       "text-align: center; text-decoration: none; display: inline-block; font-size: 12px; " \
                       "margin: 4px 2px; cursor: pointer; border-radius: 8px; }" \
                       "QPushButton:hover { background-color: #45a049; }"

        self.zone_in_button.setStyleSheet(button_style)
        self.zone_out_button.setStyleSheet(button_style)
        self.run_button.setStyleSheet(button_style)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.zone_in_button)
        button_layout.addWidget(self.zone_out_button)
        button_layout.addWidget(self.run_button)

        button_widget = QWidget()
        button_widget.setLayout(button_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.view)
        main_layout.addWidget(button_widget)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.zone_in_button.clicked.connect(self.edit_zone_in_polygon)
        self.zone_out_button.clicked.connect(self.edit_zone_out_polygon)
        self.run_button.clicked.connect(self.run_video_processor)

        self.editing_mode = None
        self.video_info = sv.VideoInfo.from_video_path(video_path)



    def get_first_frame(self):
        cap = cv2.VideoCapture(TrackingConfig.VIDEO_FILE_ZONE)
        ret, frame = cap.read()
        cap.release()
        return frame

    def edit_zone_in_polygon(self):
        self.editing_mode = "zone_in"
        self.zone_in_polygon_points = []
        if self.zone_in_polygon_item:
            self.scene.removeItem(self.zone_in_polygon_item)
            self.zone_in_polygon_item = None

    def edit_zone_out_polygon(self):
        self.editing_mode = "zone_out"
        self.zone_out_polygon_points = []
        if self.zone_out_polygon_item:
            self.scene.removeItem(self.zone_out_polygon_item)
            self.zone_out_polygon_item = None

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            if self.editing_mode == "zone_in":
                self.zone_in_polygon_points.append(self.view.mapToScene(event.pos()))
            elif self.editing_mode == "zone_out":
                self.zone_out_polygon_points.append(self.view.mapToScene(event.pos()))

    def mouse_move_event(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.editing_mode == "zone_in":
                self.zone_in_polygon_points[-1] = self.view.mapToScene(event.pos())
                self.update_polygon(self.zone_in_polygon_points, self.zone_in_polygon_item)
            elif self.editing_mode == "zone_out":
                self.zone_out_polygon_points[-1] = self.view.mapToScene(event.pos())
                self.update_polygon(self.zone_out_polygon_points, self.zone_out_polygon_item)

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            if self.editing_mode == "zone_in":
                self.zone_in_polygon_points.append(self.view.mapToScene(event.pos()))
                self.update_polygon(self.zone_in_polygon_points, self.zone_in_polygon_item)
            elif self.editing_mode == "zone_out":
                self.zone_out_polygon_points.append(self.view.mapToScene(event.pos()))
                self.update_polygon(self.zone_out_polygon_points, self.zone_out_polygon_item)

    def update_polygon(self, polygon_points, polygon_item):
        if polygon_item:
            self.scene.removeItem(polygon_item)

        polygon = QPolygonF(polygon_points)
        polygon_item = QGraphicsPolygonItem(polygon)
        polygon_item.setBrush(QBrush(QColor(255, 0, 0, 100)))
        self.scene.addItem(polygon_item)

        if self.editing_mode == "zone_in":
            self.zone_in_polygon_item = polygon_item
        elif self.editing_mode == "zone_out":
            self.zone_out_polygon_item = polygon_item

    def run_video_processor(self):
        zone_in_polygons = []
        zone_out_polygons = []

        if self.zone_in_polygon_points:
            polygon_points = []
            for p in self.zone_in_polygon_points:
                polygon_points.append((int(p.x() * self.video_info.resolution_wh[0] / self.scene.width()),
                                       int(p.y() * self.video_info.resolution_wh[1] / self.scene.height())))
            zone_in_polygons.append(np.array(polygon_points))
        else:
            zone_in_polygons = [np.array([[1015, 1], [1919, 1], [1919, 1079], [1015, 1079], [1015, 1]])]

        if self.zone_out_polygon_points:
            polygon_points = []
            for p in self.zone_out_polygon_points:
                polygon_points.append((int(p.x() * self.video_info.resolution_wh[0] / self.scene.width()),
                                       int(p.y() * self.video_info.resolution_wh[1] / self.scene.height())))
            zone_out_polygons.append(np.array(polygon_points))
        else:
            zone_out_polygons = [np.array([[1015, 1], [1019, 1079], [1, 1079], [1, 1], [1015, 1]])]

        processor = VideoProcessor(
            source_weights_path=TrackingConfig.MODEL_WEIGHTS,
            source_video_path=TrackingConfig.VIDEO_FILE_ZONE,
            target_video_path=None,
            confidence_threshold=TrackingConfig.CONFIDENCE,
            iou_threshold=TrackingConfig.IOU,
            zone_in_polygons=zone_in_polygons,
            zone_out_polygons=zone_out_polygons,
        )
        processor.process_video()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonEditorWindow()
    window.show()
    sys.exit(app.exec_())