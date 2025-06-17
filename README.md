# 🐾 Real-Time Animal Behavior Analysis using YOLOv5 and YOLOv8

This project aims to automate the detection, tracking, and classification of animal behavior in real-time video feeds using deep learning models—YOLOv5 and YOLOv8. It contributes to fields such as animal welfare, livestock management, and wildlife conservation by offering a scalable, cost-effective AI solution.

---

## 📌 Features

- 🐶 Detect animals in real-time using object detection
- 🎯 Track animal movements using multi-object tracking (MOT)
- 🔍 Classify common behaviors like:
  - Walking
  - Running
  - Resting
  - Eating
- 🚨 Identify and flag abnormal behavior patterns
- 📊 Visualize detection with real-time bounding boxes and activity logs

---

## 📂 Project Structure
animal-behavior-analysis/
│


├── data/                  # Input video datasets

├── models/                # YOLOv5 and YOLOv8 configuration and weights

├── utils/                 # Helper functions for tracking, visualization

├── outputs/               # Detection and classification results

├── main.py                # Main entry point for detection & classification

├── requirements.txt       # Python dependencies

└── README.md              # Project documentation


---

## 🧠 Models Used

- **YOLOv5** – Lightweight, real-time detection with fast inference  
- **YOLOv8** – Improved accuracy, robustness in complex scenes  
- **MOT (Multi-Object Tracking)** – Maintains consistent ID of animals across frames  
- **Custom Classifier** – Classifies behavior based on movement and trajectory patterns

---

## 🔧 Technologies & Tools

- Python 3.8+
- PyTorch (YOLOv5 & YOLOv8)
- OpenCV (video processing)
- NumPy, Pandas, Scikit-learn
- CUDA-enabled GPU (recommended)

---

## 🧪 Simulation Environment

- **Hardware:** NVIDIA GPU (e.g., RTX 3060 or better)
- **Software:** Ubuntu/Windows, Python, Jupyter Notebook (optional)
- **Datasets:** Publicly available animal behavior and wildlife video datasets

---

## 🧪 Results

| Metric          | YOLOv5     | YOLOv8     |
|-----------------|------------|------------|
| Accuracy        | 87%        | **91%**     |
| Real-Time FPS   | ~28 FPS    | ~25 FPS    |
| Behavior Classification Accuracy | 84% | **89%** |

> YOLOv8 outperformed YOLOv5 in complex environments with better feature extraction and higher classification accuracy.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/animal-behavior-analysis.git
cd animal-behavior-analysis

# Install dependencies
pip install -r requirements.txt

# Run detection and behavior analysis
python main.py --model yolov8 --input data/video.mp4


