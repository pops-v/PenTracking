import torch
from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8m.pt')  # Load a pre-trained YOLOv8 model
    model.train(data='C:/Users/pops/Desktop/PenDetection/dataset.yaml', epochs=50, imgsz=640)

if __name__ == '__main__':
    train_model()
