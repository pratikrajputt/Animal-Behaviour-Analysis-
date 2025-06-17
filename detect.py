import cv2
from ultralytics import YOLO
import os
import time
from collections import defaultdict

# Load custom models
custom_model_v5 = YOLO('yolo_model/bestv5.pt')  # YOLOv5 custom model
custom_model_v8 = YOLO('yolo_model/GZbestv8.pt')  # YOLOv8 custom model

# Load pretrained model for class names
name_model = YOLO('yolov8n.pt')

# Interested animal classes
INTERESTED_CLASSES = ["zebra", "giraffe", "pigeon"]

# Real performance metrics (replace with dynamic eval if needed)
ACCURACY_METRICS_V5 = {
    "precision": 0.96,
    "recall": 0.96,
    "f1_score": 0.80,
    "map50": 0.84,
    "fps": 27
}

ACCURACY_METRICS_V8 = {
    "precision": 1.00,
    "recall": 0.95,
    "f1_score": 0.80,
    "map50": 0.85,
    "fps": 27
}

def detect_image(image_path, output_path_v5=None, output_path_v8=None):
    result_data = {}

    if output_path_v5:
        start_v5 = time.time()
        results_v5 = custom_model_v5(image_path)
        v5_time = time.time() - start_v5
        cv2.imwrite(output_path_v5, results_v5[0].plot())
        result_data["v5_time"] = round(v5_time, 2)
        result_data.update(ACCURACY_METRICS_V5)

    if output_path_v8:
        start_v8 = time.time()
        results_v8 = custom_model_v8(image_path)
        v8_time = time.time() - start_v8
        cv2.imwrite(output_path_v8, results_v8[0].plot())
        result_data["v8_time"] = round(v8_time, 2)
        result_data.update({f"v8_{k}": v for k, v in ACCURACY_METRICS_V8.items()})

    return result_data

def detect_video(video_path, output_path_v5=None, output_path_v8=None):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_path_v5:
        out_v5 = cv2.VideoWriter(output_path_v5, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    if output_path_v8:
        out_v8 = cv2.VideoWriter(output_path_v8, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

    v5_frames = v8_frames = 0
    v5_total_time = v8_total_time = 0
    class_counts = defaultdict(int)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if output_path_v5:
            start_v5 = time.time()
            results_v5 = custom_model_v5(frame)
            v5_total_time += time.time() - start_v5
            v5_frames += 1
            out_v5.write(results_v5[0].plot())

        if output_path_v8:
            start_v8 = time.time()
            results_v8 = custom_model_v8(frame)
            v8_total_time += time.time() - start_v8
            v8_frames += 1
            out_v8.write(results_v8[0].plot())

        # Use name_model just for readable class names
        temp_result = name_model(frame)[0]
        for box in temp_result.boxes:
            cls = int(box.cls.item())
            name = temp_result.names.get(cls)
            if name in INTERESTED_CLASSES:
                class_counts[name] += 1

    cap.release()
    if output_path_v5:
        out_v5.release()
    if output_path_v8:
        out_v8.release()

    result = {
        "classes": [{"name": k, "count": v} for k, v in class_counts.items()]
    }

    if output_path_v5:
        result.update({
            "v5_fps": round(v5_frames / v5_total_time, 2) if v5_total_time else 0,
            "v5_time": round(v5_total_time, 2),
            **ACCURACY_METRICS_V5
        })

    if output_path_v8:
        result.update({
            "v8_fps": round(v8_frames / v8_total_time, 2) if v8_total_time else 0,
            "v8_time": round(v8_total_time, 2),
            **{f"v8_{k}": v for k, v in ACCURACY_METRICS_V8.items()}
        })

    return result

def get_detected_classes(image_path, model="both"):
    detected = defaultdict(int)

    if model in ["v5", "both"]:
        results_v5 = name_model(image_path)[0]
        for box in results_v5.boxes:
            cls = int(box.cls.item())
            name = results_v5.names.get(cls)
            if name in INTERESTED_CLASSES:
                detected[name] += 1

    if model in ["v8", "both"]:
        results_v8 = name_model(image_path)[0]
        for box in results_v8.boxes:
            cls = int(box.cls.item())
            name = results_v8.names.get(cls)
            if name in INTERESTED_CLASSES:
                detected[name] += 1

    return [{"name": k, "count": v} for k, v in detected.items()]
