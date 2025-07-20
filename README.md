# 🪖 Military Soldier Safety and Weapon Detection using YOLOv8

This project leverages YOLOv8 and computer vision techniques to detect military soldiers and weapons in real-time images and videos. It aims to enhance soldier safety and automate surveillance using deep learning-based object detection.

---

## 📌 Project Highlights

- 🔍 **Multi-class object detection**: Detects and classifies multiple classes like soldiers, guns, tanks, helmets, etc.
- 🧠 **YOLOv8-based model**: Built using Ultralytics' YOLOv8.
- 🌐 **Streamlit Web App**: Easy interface for real-time image/video detection.
- 📊 **Performance Evaluation**: Includes metrics like mAP, precision, recall.
- 🧪 **Colab Notebook**: For model training and evaluation.
- 🖼️ **Visual Results**: Annotated detections on sample images/videos.

---

## 🗂️ Project Structure

military-object-detection/
├── 📂 military_object_dataset/
│   ├── train/images, labels/
│   ├── val/images, labels/
│   ├── test/images, labels/
│   └── military_datasetv1.yaml
├── 📂 runs/train/yolo_military2/weights/
│   └── best.pt
├── 📄 training_evaluation.ipynb
├── 📄 app.py (Streamlit)
├── 📄 README.md
└── 📂 sample_outputs/
    ├── sample_detection1.jpg
    └── sample_detection2.jpg

yaml
Copy
Edit

---

## 🔧 Setup Instructions

### ⚙️ 1. Install dependencies

```bash
pip install ultralytics opencv-python-headless streamlit

##Run the streamlit App

cd streamlit_app
streamlit run app.py

##Dataset Format(YOLO

Make sure your dataset folder is organized as:
military_object_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── military_datasetv1.yaml
)

Sample military_datasetv1.yaml:

yaml
Copy
Edit
train: /content/drive/MyDrive/Military/military_object_dataset/train
val: /content/drive/MyDrive/Military/military_object_dataset/val
test: /content/drive/MyDrive/Military/military_object_dataset/test

nc: 4
names: ['soldier', 'helmet', 'gun', 'tank']

##🧠 Model Training (YOLOv8)
In the Colab notebook:

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='military_object_dataset/military_datasetv1.yaml', epochs=50, imgsz=640)

##📊 Evaluation
Evaluate trained model using:

model = YOLO("runs/train/yolo_military2/weights/best.pt")
metrics = model.val(data='military_object_dataset/military_datasetv1.yaml')

##🚀 Streamlit App Features
Upload image or video for real-time detection.

Visualize bounding boxes, class names, confidence scores.

Option to save predictions.

##📈 Performance Metrics

| Metric         | Value   |
| -------------- | ------- |
| mAP\@0.5       | 0.89    |
| Precision      | 0.91    |
| Recall         | 0.87    |
| Inference Time | \~18 ms |

📬 Author
Eraiyanbu Arulmurugan

